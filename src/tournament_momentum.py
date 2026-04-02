"""Tournament Momentum: computes a per-team momentum score from live results.

Factors (each bounded independently, then summed):
  1. upset_bonus      — wins against higher-seeded opponents
  2. close_game_grit  — winning games decided by <= 5 points
  3. run_length       — consecutive tournament wins (mild compounding)
  4. margin_trend     — improving margin of victory across rounds

Returns a float in [-0.10, +0.10] suitable for the scenario engine.
"""

import csv
import os
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

ROUND_ORDER = {"FF": 0, "R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "Championship": 6}


@dataclass
class TourneyGameResult:
    round: str
    team: str
    opponent: str
    team_seed: int
    opp_seed: int
    team_score: int
    opp_score: int
    won: bool


@dataclass
class MomentumProfile:
    team: str
    wins: int = 0
    games: int = 0
    upset_bonus: float = 0.0
    close_game_grit: float = 0.0
    run_length_bonus: float = 0.0
    margin_trend: float = 0.0
    total: float = 0.0
    detail: str = ""


def _load_results(base_dir: str = None) -> Dict[str, List[TourneyGameResult]]:
    """Load all tournament results from live_results/ CSVs."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    box_path = os.path.join(base_dir, "data", "live_results", "tournament_box_scores.csv")
    if not os.path.isfile(box_path):
        return {}

    team_results: Dict[str, List[TourneyGameResult]] = {}

    with open(box_path, "r") as f:
        for row in csv.DictReader(f):
            rnd = row.get("round", "").strip()
            if rnd not in ROUND_ORDER:
                continue

            team_a = row.get("team_a", "").strip()
            team_b = row.get("team_b", "").strip()
            score_a = int(row.get("score_a", 0) or 0)
            score_b = int(row.get("score_b", 0) or 0)
            seed_a = int(row.get("seed_a", 0) or 0)
            seed_b = int(row.get("seed_b", 0) or 0)
            winner = row.get("winner", "").strip()

            if score_a == 0 and score_b == 0:
                continue

            if team_a:
                g = TourneyGameResult(
                    round=rnd, team=team_a, opponent=team_b,
                    team_seed=seed_a, opp_seed=seed_b,
                    team_score=score_a, opp_score=score_b,
                    won=(winner == team_a),
                )
                team_results.setdefault(team_a, []).append(g)

            if team_b:
                g = TourneyGameResult(
                    round=rnd, team=team_b, opponent=team_a,
                    team_seed=seed_b, opp_seed=seed_a,
                    team_score=score_b, opp_score=score_a,
                    won=(winner == team_b),
                )
                team_results.setdefault(team_b, []).append(g)

    for games in team_results.values():
        games.sort(key=lambda g: ROUND_ORDER.get(g.round, 99))

    return team_results


def compute_momentum(team_results: Dict[str, List[TourneyGameResult]]) -> Dict[str, MomentumProfile]:
    """Compute momentum profile for every team with tournament data."""
    profiles: Dict[str, MomentumProfile] = {}

    for team, games in team_results.items():
        p = MomentumProfile(team=team, games=len(games))
        wins = [g for g in games if g.won]
        p.wins = len(wins)

        if p.wins == 0:
            profiles[team] = p
            continue

        # --- Factor 1: Upset bonus ---
        # Beating a higher seed (lower number) earns a bonus proportional to seed gap
        upset_sum = 0.0
        for g in wins:
            if g.opp_seed > 0 and g.team_seed > g.opp_seed:
                seed_gap = g.team_seed - g.opp_seed
                upset_sum += min(seed_gap * 0.005, 0.025)
        p.upset_bonus = min(upset_sum, 0.04)

        # --- Factor 2: Close-game grit ---
        # Winning games decided by <= 5 points shows clutch ability
        close_wins = sum(1 for g in wins if abs(g.team_score - g.opp_score) <= 5)
        p.close_game_grit = min(close_wins * 0.008, 0.025)

        # --- Factor 3: Run length ---
        # Consecutive wins compound mildly (diminishing returns via sqrt)
        consecutive = 0
        for g in reversed(games):
            if g.won:
                consecutive += 1
            else:
                break
        if consecutive >= 2:
            p.run_length_bonus = min(math.sqrt(consecutive - 1) * 0.008, 0.02)

        # --- Factor 4: Margin trend ---
        # Are margins improving across rounds? Positive slope = team peaking
        margins = [g.team_score - g.opp_score for g in games]
        if len(margins) >= 2:
            n = len(margins)
            x_mean = (n - 1) / 2.0
            y_mean = sum(margins) / n
            num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(margins))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0.0
            p.margin_trend = max(-0.02, min(slope * 0.002, 0.02))

        p.total = max(-0.10, min(
            p.upset_bonus + p.close_game_grit + p.run_length_bonus + p.margin_trend,
            0.10
        ))

        details = []
        if p.upset_bonus > 0:
            details.append(f"upset={p.upset_bonus:+.3f}")
        if p.close_game_grit > 0:
            details.append(f"grit={p.close_game_grit:+.3f}")
        if p.run_length_bonus > 0:
            details.append(f"run={p.run_length_bonus:+.3f}")
        if abs(p.margin_trend) > 0.001:
            details.append(f"trend={p.margin_trend:+.3f}")
        p.detail = ", ".join(details) if details else "neutral"

        profiles[team] = p

    return profiles


def load_tournament_momentum(base_dir: str = None) -> Dict[str, MomentumProfile]:
    """Top-level loader: reads results, computes momentum for all teams."""
    results = _load_results(base_dir)
    return compute_momentum(results)


def momentum_report(profiles: Dict[str, MomentumProfile]) -> str:
    """Print tournament momentum summary."""
    lines = ["=" * 80, "  TOURNAMENT MOMENTUM PROFILES", "=" * 80]

    ranked = sorted(profiles.values(), key=lambda p: p.total, reverse=True)
    for p in ranked:
        if p.wins == 0:
            continue
        flag = " ** HOT **" if p.total >= 0.03 else ""
        lines.append(
            f"  {p.team:<20} {p.wins}W/{p.games}G  "
            f"momentum={p.total:+.3f}  [{p.detail}]{flag}"
        )

    return "\n".join(lines)
