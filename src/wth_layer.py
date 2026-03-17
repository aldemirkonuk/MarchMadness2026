"""What The Hell (WTH) chaos modifier layer.

These parameters live OUTSIDE the main weighted model.
They act as multipliers/flags that adjust Monte Carlo variance.
"""

import numpy as np
from typing import Dict, Optional
from src.models import Team
from src.equations import (
    chaos_index, sightline_penalty, altitude_impact,
    ref_impact, sentiment_score, jersey_color_aggression,
)

# Known tournament venue elevations (ft above sea level)
VENUE_ELEVATIONS = {
    "Buffalo": 600,
    "Greenville": 960,
    "Oklahoma City": 1200,
    "Portland": 50,
    "Tampa": 48,
    "Philadelphia": 39,
    "San Diego": 62,
    "St. Louis": 465,
    "Houston": 80,
    "San Jose": 82,
    "Chicago": 594,
    "Washington DC": 0,
    "Indianapolis": 715,  # Final Four -- Lucas Oil Stadium
}

# Sightline penalty for NFL-stadium venues (wider sightlines, worse depth perception)
NFL_STADIUM_VENUES = {"Indianapolis"}  # Lucas Oil Stadium
SIGHTLINE_PENALTY_FACTOR = 0.03  # ~3% 3PT drop in NFL stadiums historically


def compute_wth_adjustments(team: Team, venue_city: str = "",
                            public_pick_pct: float = 0.5) -> Dict[str, float]:
    """Compute all WTH modifiers for a team at a given venue."""
    adjustments = {}

    # WTH-1: Sightline Penalty
    if venue_city in NFL_STADIUM_VENUES:
        adjustments["sightline"] = SIGHTLINE_PENALTY_FACTOR
    else:
        adjustments["sightline"] = 0.0

    # WTH-2: Altitude Impact
    venue_elev = VENUE_ELEVATIONS.get(venue_city, 200)
    team_elev = 500  # default home elevation
    elev_diff = max(0, venue_elev - team_elev)
    adjustments["altitude"] = altitude_impact(elev_diff, team.three_pa_fga)

    # WTH-4: Chaos Index (already computed on team)
    adjustments["chaos"] = team.chaos_index

    # WTH-5: Sentiment (from public pick % as proxy)
    adjustments["sentiment"] = (public_pick_pct - 0.5) * 0.02

    return adjustments


def compute_total_wth_modifier(adjustments_a: dict, adjustments_b: dict,
                               team_a: "Team" = None, team_b: "Team" = None
                               ) -> float:
    """Combine WTH adjustments into a single modifier for upset_volatility.

    Returns a value added to the volatility parameter V.
    Positive = more volatile (closer to 50/50).
    """
    sightline_effect = (adjustments_a.get("sightline", 0) +
                        adjustments_b.get("sightline", 0)) * 0.5

    altitude_effect = abs(adjustments_a.get("altitude", 0) -
                          adjustments_b.get("altitude", 0)) * 0.1

    chaos_effect = (adjustments_a.get("chaos", 0) +
                    adjustments_b.get("chaos", 0)) * 0.5

    # Volatility from scoring_margin_std (high variance = more chaos)
    variance_effect = 0.0
    if team_a is not None and team_b is not None:
        avg_std = (getattr(team_a, "scoring_margin_std", 10) +
                   getattr(team_b, "scoring_margin_std", 10)) / 2.0
        if avg_std > 12.0:
            variance_effect = min(0.04, (avg_std - 12.0) * 0.008)

    total = sightline_effect + altitude_effect + chaos_effect + variance_effect
    return np.clip(total, 0.0, 0.15)


def apply_playstyle_index(team: Team) -> float:
    """WTH-7: Playstyle Index (PSI).

    PSI > 0 = athletic/defensive/fast
    PSI < 0 = shooting/passing/halfcourt
    """
    pace_norm = (team.pace - 64) / 10  # center around average pace
    return (0.25 * pace_norm +
            0.20 * team.stl_rate * 10 +
            0.15 * team.blk_pct * 10 -
            0.20 * team.three_pa_fga * 3 -
            0.20 * team.ast_pct * 2)
