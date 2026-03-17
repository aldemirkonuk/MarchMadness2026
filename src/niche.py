"""Super niche parameters: Jersey Color, Scoring Runs, Blowout Resilience, Foul Trouble.

These are Tier 5 parameters with low weight but high insight.
"""

from typing import Dict, List
from src.models import Team


def compute_niche_metrics(team: Team) -> Dict[str, float]:
    """Compute all niche metrics for a team. Most require PBP data.

    For metrics without PBP data, we use available proxies.
    """
    metrics = {}

    # Equation 34: Max Scoring Run Potential -- using EvanMiya KILLSHOTS as proxy
    metrics["msrp"] = team.killshots_per_game - team.killshots_conceded

    # Equation 35: Blowout Resilience -- needs PBP data
    # Proxy: teams with high AdjEM and good clutch factor are more resilient
    metrics["blowout_resilience"] = team.blowout_resilience

    # Equation 36: Foul Trouble Impact -- needs PBP data
    # Proxy: high bench depth = less foul trouble impact
    metrics["foul_trouble_impact"] = team.foul_trouble_impact

    return metrics


def enrich_niche(teams: List[Team]) -> None:
    """Apply niche metric proxies to all teams.

    Note: blowout_resilience, foul_trouble_impact, and clutch_factor
    are now computed in data_loader using real TeamRankings + EvanMiya data.
    This function only enriches metrics not already set by data_loader.
    """
    pass
