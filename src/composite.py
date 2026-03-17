"""Phase 1A: Weighted Composite Engine + Logistic Win Probability.

This is the interpretable, hand-tuned prediction engine.
Every weight is a knob the user can turn.
"""

import numpy as np
from typing import List, Dict, Tuple

from src.models import Team, Matchup
from src.equations import (
    team_strength_composite,
    composite_score_differential,
    win_probability_logistic,
    upset_volatility,
    is_cinderella,
)
from src.weights import CORE_WEIGHTS, LOGISTIC_K, PARAM_KEYS, DATASET_CONFIG
from src.utils import normalize_teams
from src.wth_layer import compute_wth_adjustments, compute_total_wth_modifier


def compute_team_strengths(teams: List[Team]) -> List[Team]:
    """Normalize all parameters and compute composite TeamStrength scores."""
    normalize_teams(teams, PARAM_KEYS)

    for team in teams:
        team.team_strength = team_strength_composite(
            team.normalized_params, CORE_WEIGHTS
        )

    _detect_cinderellas(teams)
    return teams


def rank_teams(teams: List[Team]) -> List[Team]:
    """Return teams sorted by composite TeamStrength (power rankings)."""
    return sorted(teams, key=lambda t: t.team_strength, reverse=True)


def compute_win_probability(matchup: Matchup) -> float:
    """Compute Phase 1A win probability for team_a in the matchup."""
    z = composite_score_differential(
        matchup.team_a.normalized_params,
        matchup.team_b.normalized_params,
        CORE_WEIGHTS,
    )

    p_base = win_probability_logistic(z, k=LOGISTIC_K)

    wth_adj = 0.0
    if DATASET_CONFIG.get("use_wth_layer", False):
        adj_a = compute_wth_adjustments(matchup.team_a)
        adj_b = compute_wth_adjustments(matchup.team_b)
        wth_adj = compute_total_wth_modifier(adj_a, adj_b)

    p_adj = upset_volatility(
        p_base,
        matchup.team_a.chaos_index,
        matchup.team_b.chaos_index,
        pace_diff_norm=abs(matchup.team_a.pace - matchup.team_b.pace) / 20.0,
        wth_adjustment=wth_adj,
    )

    matchup.win_prob_a_1a = p_adj
    matchup.volatility = abs(p_adj - 0.5)
    return p_adj


def compute_all_matchup_probabilities(matchups: List[Matchup]) -> List[Matchup]:
    """Compute Phase 1A probabilities for all matchups."""
    for m in matchups:
        compute_win_probability(m)
    return matchups


def generate_pros_cons(matchup: Matchup) -> Matchup:
    """Generate qualitative pros/cons for both teams in a matchup."""
    a, b = matchup.team_a, matchup.team_b
    na, nb = a.normalized_params, b.normalized_params

    matchup.pros_a = []
    matchup.cons_a = []
    matchup.pros_b = []
    matchup.cons_b = []

    LABELS = {
        "adj_em": "Overall efficiency",
        "shooting_eff": "Shooting efficiency",
        "clutch_factor": "Clutch performance",
        "sos": "Strength of schedule",
        "to_pct": "Ball security (low turnovers)",
        "scoring_balance": "Inside-outside scoring balance",
        "orb_pct": "Offensive rebounding",
        "seed_score": "Seed strength",
        "top50_perf": "Elite competition record",
        "ftr": "Free throw reliability",
        "ast_pct": "Ball movement / assists",
        "spi": "Star power",
        "exp": "Experience",
        "dvi": "Defensive versatility",
        "drb_pct": "Defensive rebounding",
        "opp_to_pct": "Turnover forcing (defense)",
        "rpi_rim": "Rim protection",
        "eff_height": "Team height advantage",
        "momentum": "Momentum / hot streak",
        "ctf": "Coaching tournament pedigree",
        "fragility_score": "Resilience (low fragility)",
        "march_readiness": "March readiness",
        "msrp": "Scoring run potential",
        "barthag": "Power rating (BARTHAG)",
        "net_score": "NCAA NET ranking",
        "ppg_margin": "Scoring margin",
        "injury_health": "Roster health",
        "star_above_avg": "Star player above average",
        "z_rating": "Z Rating composite",
        "cwp_star_17_half": "Star dominance at halftime",
        "consistency": "Game-to-game consistency",
        "q1_record": "Quadrant 1 win record",
        "q34_loss_rate": "Bad loss avoidance",
        "offensive_burst": "Fast start / 1st quarter",
        "q3_adj_strength": "Halftime adjustment ability",
    }

    for param, label in LABELS.items():
        va = na.get(param, 0.5)
        vb = nb.get(param, 0.5)
        diff = va - vb

        if abs(diff) < 0.05:
            continue

        if diff > 0.15:
            matchup.pros_a.append(f"Strong edge in {label} ({va:.2f} vs {vb:.2f})")
            matchup.cons_b.append(f"Weak in {label} vs opponent")
        elif diff > 0.05:
            matchup.pros_a.append(f"Advantage in {label}")
        elif diff < -0.15:
            matchup.pros_b.append(f"Strong edge in {label} ({vb:.2f} vs {va:.2f})")
            matchup.cons_a.append(f"Weak in {label} vs opponent")
        elif diff < -0.05:
            matchup.pros_b.append(f"Advantage in {label}")

    # Key matchup narratives
    if a.three_pa_fga > 0.35 and b.opp_3p_pct < 0.32:
        matchup.cons_a.append("Three-point heavy team faces elite perimeter defense")
    if b.three_pa_fga > 0.35 and a.opp_3p_pct < 0.32:
        matchup.cons_b.append("Three-point heavy team faces elite perimeter defense")

    if a.exp < 1.5:
        matchup.cons_a.append("Very young roster -- tournament inexperience risk")
    if b.exp < 1.5:
        matchup.cons_b.append("Very young roster -- tournament inexperience risk")

    if a.is_cinderella:
        matchup.pros_a.append("CINDERELLA ALERT: underseeded by metrics")
    if b.is_cinderella:
        matchup.pros_b.append("CINDERELLA ALERT: underseeded by metrics")

    return matchup


def _detect_cinderellas(teams: List[Team]) -> None:
    """Flag teams that meet Cinderella criteria."""
    clutch_values = [t.clutch_factor for t in teams]
    median_clutch = np.median(clutch_values)

    for t in teams:
        t.is_cinderella = is_cinderella(
            t.adj_em_rank, t.seed, t.exp, t.clutch_factor, median_clutch
        )
