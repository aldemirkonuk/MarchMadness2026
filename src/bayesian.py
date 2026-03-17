"""Phase 2: Bayesian live-updating system.

Once tournament games start, updates team strengths based on
observed results. Applies to both 1A and 1B tracks.
"""

import numpy as np
from typing import List, Dict, Optional

from src.models import Team, SimulationResult
from src.equations import posterior_update, game_likelihood, ROUND_ALPHA
from src.monte_carlo import simulate_tournament


def update_team_strength(team: Team, observed_margin: float,
                         opponent_strength: float,
                         round_name: str = "R64") -> float:
    """Update a team's strength based on observed game result.

    Args:
        team: The team to update
        observed_margin: Actual point margin (positive = team won)
        opponent_strength: Opponent's pre-game TeamStrength
        round_name: Current tournament round (affects alpha)

    Returns:
        Updated team strength
    """
    alpha = ROUND_ALPHA.get(round_name, 0.7)

    # Observed performance = how much better/worse than predicted
    predicted_margin = (team.team_strength - opponent_strength) * 20
    performance_signal = observed_margin / 20.0

    # Teams that massively outperform get a boost; underperformers get penalized
    updated = posterior_update(
        team.team_strength,
        team.team_strength + (performance_signal - predicted_margin / 20) * 0.5,
        alpha=alpha,
    )

    team.team_strength = updated
    return updated


def bayesian_round_update(teams: List[Team],
                          results: List[Dict],
                          round_name: str = "R64") -> List[Team]:
    """Update all teams after a round of results.

    Args:
        teams: All surviving teams
        results: List of dicts with keys:
            'winner': team name, 'loser': team name,
            'winner_score': int, 'loser_score': int
        round_name: Which round just completed

    Returns:
        Updated team list (losers removed)
    """
    team_dict = {t.name: t for t in teams}

    for result in results:
        winner_name = result["winner"]
        loser_name = result["loser"]
        margin = result["winner_score"] - result["loser_score"]

        winner = team_dict.get(winner_name)
        loser = team_dict.get(loser_name)

        if winner and loser:
            update_team_strength(winner, margin, loser.team_strength, round_name)
            update_team_strength(loser, -margin, winner.team_strength, round_name)

    surviving = [t for t in teams if t.name not in
                 {r["loser"] for r in results}]

    return surviving


def resimulate_remaining(teams: List[Team],
                         n_simulations: int = 100_000,
                         prob_func=None) -> SimulationResult:
    """Re-run Monte Carlo on remaining bracket after Bayesian updates."""
    from src.utils import normalize_teams
    from src.weights import PARAM_KEYS

    normalize_teams(teams, PARAM_KEYS)
    return simulate_tournament(teams, n_simulations, prob_func)


def live_update_report(pre_odds: Dict[str, float],
                       post_odds: Dict[str, float],
                       round_name: str) -> str:
    """Generate a report showing how odds changed after a round."""
    lines = [
        "=" * 60,
        f"  BAYESIAN UPDATE: After {round_name}",
        "=" * 60,
        f"{'Team':<25}{'Pre':>10}{'Post':>10}{'Change':>10}",
        "-" * 55,
    ]

    all_teams = set(list(pre_odds.keys()) + list(post_odds.keys()))
    changes = []
    for team in all_teams:
        pre = pre_odds.get(team, 0)
        post = post_odds.get(team, 0)
        changes.append((team, pre, post, post - pre))

    changes.sort(key=lambda x: x[3], reverse=True)

    for team, pre, post, change in changes[:15]:
        arrow = "+" if change > 0 else ""
        lines.append(
            f"{team:<25}{pre*100:>9.1f}%{post*100:>9.1f}%{arrow}{change*100:>8.1f}%"
        )

    return "\n".join(lines)
