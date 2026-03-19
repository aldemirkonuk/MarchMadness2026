"""Recency Weighting Engine -- Phase 8.

Replaces flat full-season averages with exponentially-decaying weighted stats.
A game 10 games ago has 0.95^10 = 0.60 weight. A game 20 games ago has 0.36 weight.

Computes:
  - recency_rating: weighted average of per-game rating differential (PPP)
  - recency_margin: weighted average of scoring margin
  - form_trend: slope of linear fit on last 15 game ratings (rising/falling)
  - late_season_clutch: close-game win% in last 10 games only

Blends into existing momentum parameter and adds form_trend as new weight key.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from src.models import Team
from src.utils import canonical_name

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE3 = os.path.join(BASE_DIR, "archive-3")

# Exponential decay rate per game (5% decay per game back)
DECAY_LAMBDA = 0.95

# Window sizes
WINDOW_FULL = 999       # All games
WINDOW_LAST_15 = 15
WINDOW_LAST_7 = 7
WINDOW_LAST_10 = 10


def compute_recency_metrics(teams: List[Team]) -> Dict[str, dict]:
    """Compute recency-weighted metrics from game logs for all teams.

    Returns dict of team_name -> {recency_rating, recency_margin, form_trend,
                                   late_season_clutch, last7_margin, last15_margin}.
    """
    log_dir = os.path.join(ARCHIVE3, "game-logs")
    if not os.path.isdir(log_dir):
        return {}

    results = {}

    for fn in os.listdir(log_dir):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(log_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if df.empty or "team" not in df.columns:
            continue

        team_name = canonical_name(str(df.iloc[0]["team"]).strip())
        n = len(df)

        if n < 3:
            continue

        # Parse scores and ratings
        scores_t = df["score_t"].astype(float).values if "score_t" in df.columns else None
        scores_o = df["score_o"].astype(float).values if "score_o" in df.columns else None
        ratings_t = df["rating_t"].astype(float).values if "rating_t" in df.columns else None
        ratings_o = df["rating_o"].astype(float).values if "rating_o" in df.columns else None
        win_flags = (df["result"] == "W").astype(float).values if "result" in df.columns else None

        if scores_t is None or scores_o is None:
            continue

        margins = scores_t - scores_o

        # ── Exponential decay weights (most recent game = index n-1) ──
        decay_weights = np.array([DECAY_LAMBDA ** (n - 1 - i) for i in range(n)])
        decay_weights_sum = decay_weights.sum()

        # ── Recency-weighted scoring margin ──
        recency_margin = np.dot(margins, decay_weights) / decay_weights_sum

        # ── Recency-weighted rating differential (PPP) ──
        recency_rating = 0.0
        if ratings_t is not None and ratings_o is not None:
            rating_diffs = ratings_t - ratings_o
            recency_rating = np.dot(rating_diffs, decay_weights) / decay_weights_sum

        # ── Form trend: slope of linear fit on last 15 game margins ──
        last_n = min(WINDOW_LAST_15, n)
        recent_margins = margins[-last_n:]
        if len(recent_margins) >= 3:
            x = np.arange(len(recent_margins))
            # Linear regression slope
            x_mean = x.mean()
            y_mean = recent_margins.mean()
            numerator = np.sum((x - x_mean) * (recent_margins - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            form_trend = numerator / max(denominator, 1e-6)
        else:
            form_trend = 0.0

        # ── Last 7 and Last 10 averages ──
        last7_margin = margins[-min(WINDOW_LAST_7, n):].mean()
        last10_margin = margins[-min(WINDOW_LAST_10, n):].mean()
        last15_margin = margins[-min(WINDOW_LAST_15, n):].mean()

        # ── Late season clutch: close-game win% in last 10 only ──
        late_season_clutch = 0.5
        if win_flags is not None:
            last10_df = df.tail(WINDOW_LAST_10)
            if "score_t" in last10_df.columns and "score_o" in last10_df.columns:
                l10_margins = (last10_df["score_t"].astype(float) -
                               last10_df["score_o"].astype(float)).abs()
                close_mask = l10_margins <= 5
                close_games = last10_df[close_mask]
                if len(close_games) > 0:
                    late_season_clutch = (close_games["result"] == "W").mean()

        # ── Win rate windows ──
        full_wpct = win_flags.mean() if win_flags is not None else 0.5
        last10_wpct = win_flags[-min(WINDOW_LAST_10, n):].mean() if win_flags is not None else 0.5
        last7_wpct = win_flags[-min(WINDOW_LAST_7, n):].mean() if win_flags is not None else 0.5

        # ── Recency-weighted win% ──
        if win_flags is not None:
            recency_wpct = np.dot(win_flags, decay_weights) / decay_weights_sum
        else:
            recency_wpct = 0.5

        results[team_name] = {
            "recency_margin": float(recency_margin),
            "recency_rating": float(recency_rating),
            "form_trend": float(form_trend),
            "late_season_clutch": float(late_season_clutch),
            "last7_margin": float(last7_margin),
            "last10_margin": float(last10_margin),
            "last15_margin": float(last15_margin),
            "last7_wpct": float(last7_wpct),
            "last10_wpct": float(last10_wpct),
            "full_wpct": float(full_wpct),
            "recency_wpct": float(recency_wpct),
            "games_played": n,
        }

    return results


def enrich_teams_with_recency(teams: List[Team],
                               recency_data: Dict[str, dict]) -> None:
    """Enrich Team objects with recency metrics.

    Updates momentum with recency-weighted blend and adds form_trend.
    """
    if not recency_data:
        return

    # Collect form_trend values for normalization
    form_trends = [d["form_trend"] for d in recency_data.values()]
    ft_min = min(form_trends) if form_trends else -1
    ft_max = max(form_trends) if form_trends else 1
    ft_range = max(ft_max - ft_min, 0.01)

    # Collect recency_margin for normalization
    rm_values = [d["recency_margin"] for d in recency_data.values()]
    rm_min = min(rm_values) if rm_values else -10
    rm_max = max(rm_values) if rm_values else 10
    rm_range = max(rm_max - rm_min, 0.01)

    for team in teams:
        rd = recency_data.get(team.name)
        if rd is None:
            # Store defaults
            team.form_trend = 0.5
            team.recency_rating_norm = 0.5
            continue

        # ── Normalize form_trend to [0, 1] ──
        team.form_trend = (rd["form_trend"] - ft_min) / ft_range

        # ── Normalize recency_margin to [0, 1] ──
        team.recency_rating_norm = (rd["recency_margin"] - rm_min) / rm_range

        # ── Update momentum with recency-weighted blend ──
        # New momentum formula: heavier on recent performance
        old_momentum = team.momentum
        recency_wpct = rd.get("recency_wpct", 0.5)
        last7_wpct = rd.get("last7_wpct", 0.5)

        # Blend: 35% recency-weighted metrics, 25% last-7, 20% old momentum, 20% form trend
        team.momentum = (
            0.35 * recency_wpct +
            0.25 * last7_wpct +
            0.20 * old_momentum +
            0.20 * team.form_trend
        )

        # ── Update clutch with late-season data ──
        late_clutch = rd.get("late_season_clutch", 0.5)
        # Blend 70% existing clutch + 30% late-season clutch
        team.clutch_factor = 0.70 * team.clutch_factor + 0.30 * late_clutch


def recency_report(teams: List[Team], recency_data: Dict[str, dict]) -> str:
    """Generate report showing teams with biggest recency shifts."""
    lines = [
        "=" * 70,
        "  RECENCY & FORM TREND REPORT",
        "  (teams with significant late-season momentum shifts)",
        "=" * 70,
        "",
    ]

    # Find teams with biggest form trends
    team_trends = []
    for team in teams:
        rd = recency_data.get(team.name)
        if rd is None:
            continue
        team_trends.append((
            team.name, team.seed,
            rd["form_trend"],
            rd["last7_margin"],
            rd["recency_margin"],
            rd["last7_wpct"],
        ))

    # Rising teams (positive form trend)
    rising = sorted(team_trends, key=lambda x: x[2], reverse=True)[:8]
    lines.append("  RISING (improving form):")
    for name, seed, trend, l7m, rm, l7w in rising:
        if trend > 0:
            lines.append(f"    ({seed:2d}) {name:<22s}  trend: +{trend:.2f}  "
                         f"last7 margin: {l7m:+.1f}  last7 W%: {l7w:.0%}")
    lines.append("")

    # Falling teams (negative form trend)
    falling = sorted(team_trends, key=lambda x: x[2])[:8]
    lines.append("  FALLING (declining form):")
    for name, seed, trend, l7m, rm, l7w in falling:
        if trend < 0:
            lines.append(f"    ({seed:2d}) {name:<22s}  trend: {trend:.2f}  "
                         f"last7 margin: {l7m:+.1f}  last7 W%: {l7w:.0%}")
    lines.append("")

    return "\n".join(lines)
