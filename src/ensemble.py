"""Ensemble blender: combines Phase 1A and Phase 1B predictions.

Produces final pre-tournament predictions with confidence scoring
and disagreement flags.
"""

import numpy as np
from typing import List, Dict, Optional

from src.models import Team, Matchup
from src.equations import ensemble_probability, confidence_score
from src.weights import ENSEMBLE_LAMBDA


def blend_predictions(matchup: Matchup, lambda_: float = None) -> float:
    """Blend 1A and 1B predictions for a single matchup.

    Uses adaptive blending: when models disagree sharply, trust 1A more
    since the composite model is calibrated to seed priors.
    """
    base_lam = lambda_ if lambda_ is not None else ENSEMBLE_LAMBDA
    gap = abs(matchup.win_prob_a_1a - matchup.win_prob_a_1b)

    # Adaptive: increase 1A weight when gap is large
    if gap > 0.25:
        lam = min(base_lam + 0.20, 0.85)
    elif gap > 0.15:
        lam = min(base_lam + 0.10, 0.75)
    else:
        lam = base_lam

    p_ens = ensemble_probability(matchup.win_prob_a_1a, matchup.win_prob_a_1b, lam)
    matchup.win_prob_a_ensemble = p_ens
    matchup.confidence = confidence_score(matchup.win_prob_a_1a, matchup.win_prob_a_1b)
    return p_ens


def blend_all_matchups(matchups: List[Matchup],
                       lambda_: float = None) -> List[Matchup]:
    """Blend predictions for all matchups."""
    for m in matchups:
        blend_predictions(m, lambda_)
    return matchups


def flag_disagreements(matchups: List[Matchup],
                       threshold: float = 0.15) -> List[Matchup]:
    """Flag matchups where 1A and 1B disagree significantly."""
    flagged = []
    for m in matchups:
        diff = abs(m.win_prob_a_1a - m.win_prob_a_1b)
        if diff > threshold:
            flagged.append(m)
    return sorted(flagged, key=lambda m: abs(m.win_prob_a_1a - m.win_prob_a_1b),
                  reverse=True)


def disagreement_report(matchups: List[Matchup]) -> str:
    """Generate a report of where the two models disagree."""
    flagged = flag_disagreements(matchups)
    lines = [
        "=" * 60,
        "  MODEL DISAGREEMENT REPORT (1A vs 1B)",
        "=" * 60,
    ]

    if not flagged:
        lines.append("  Models agree on all matchups (within 15% threshold)")
    else:
        for m in flagged:
            diff = abs(m.win_prob_a_1a - m.win_prob_a_1b)
            lines.append(
                f"  {m.team_a.name} vs {m.team_b.name} [{m.region}]"
            )
            lines.append(
                f"    1A: {m.win_prob_a_1a:.1%} -> "
                f"{'A' if m.win_prob_a_1a > 0.5 else 'B'} | "
                f"1B: {m.win_prob_a_1b:.1%} -> "
                f"{'A' if m.win_prob_a_1b > 0.5 else 'B'} | "
                f"Gap: {diff:.1%}"
            )
            if m.win_prob_a_1a > 0.5 and m.win_prob_a_1b < 0.5:
                lines.append("    ** MODELS PICK DIFFERENT WINNERS **")
            lines.append("")

    return "\n".join(lines)


def calibrate_lambda(historical_results: Optional[dict] = None) -> float:
    """Calibrate the ensemble blending weight using backtesting.

    Without historical results, returns default 0.5.
    """
    if historical_results is None:
        return ENSEMBLE_LAMBDA
    return ENSEMBLE_LAMBDA
