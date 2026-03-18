"""Monte Carlo simulation engine.

Shared by Phase 1A, Phase 1B, and Ensemble.
Simulates the full 63-game tournament bracket N times.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from collections import defaultdict
from tqdm import tqdm

from src.models import Team, Matchup, SimulationResult
from src.equations import win_probability_logistic
from src.weights import CORE_WEIGHTS, LOGISTIC_K


# Round of 64 matchup indices within each region (seed-based bracket)
BRACKET_SEEDS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

ROUND_NAMES = ["R64", "R32", "S16", "E8", "F4", "Championship"]

REGIONS = ["East", "West", "South", "Midwest"]


def _get_win_prob(team_a: Team, team_b: Team,
                  prob_func: Optional[Callable] = None) -> float:
    """Compute win probability for team_a vs team_b.

    Uses scoring_margin_std as the sole volatility layer (WTH disabled).
    """
    if prob_func is not None:
        return prob_func(team_a, team_b)

    from src.equations import composite_score_differential
    z = composite_score_differential(
        team_a.normalized_params,
        team_b.normalized_params,
        CORE_WEIGHTS,
    )
    p = win_probability_logistic(z, k=LOGISTIC_K)

    # Volatility adjustment from scoring_margin_std:
    # high-variance favorites get pulled toward 0.5 (less reliable)
    std_a = getattr(team_a, "scoring_margin_std", 0)
    std_b = getattr(team_b, "scoring_margin_std", 0)
    combined_vol = (std_a + std_b) / 2.0
    if combined_vol > 12.0:
        vol_pull = min(0.06, (combined_vol - 12.0) * 0.01)
        p = p * (1.0 - vol_pull) + 0.5 * vol_pull

    return p


def simulate_single_game(team_a: Team, team_b: Team,
                          rng: np.random.Generator,
                          prob_func=None) -> Team:
    """Simulate one game, return the winner."""
    p = _get_win_prob(team_a, team_b, prob_func)
    return team_a if rng.random() < p else team_b


def simulate_region(teams_by_seed: Dict[int, Team],
                    rng: np.random.Generator,
                    prob_func=None) -> Tuple[Team, dict]:
    """Simulate a region bracket through Elite 8. Returns region champion + results."""
    bracket = []
    for s1, s2 in BRACKET_SEEDS:
        t1 = teams_by_seed.get(s1)
        t2 = teams_by_seed.get(s2)
        if t1 is None or t2 is None:
            bracket.append(t1 or t2)
        else:
            bracket.append(simulate_single_game(t1, t2, rng, prob_func))

    results = {t.name: "R64" for t in bracket}

    # Round of 32 (4 games)
    r32 = []
    for i in range(0, 8, 2):
        winner = simulate_single_game(bracket[i], bracket[i + 1], rng, prob_func)
        r32.append(winner)
        results[winner.name] = "R32"

    # Sweet 16 (2 games)
    s16 = []
    for i in range(0, 4, 2):
        winner = simulate_single_game(r32[i], r32[i + 1], rng, prob_func)
        s16.append(winner)
        results[winner.name] = "S16"

    # Elite 8 (1 game)
    champion = simulate_single_game(s16[0], s16[1], rng, prob_func)
    results[champion.name] = "E8"

    return champion, results


def simulate_tournament(teams: List[Team],
                        n_simulations: int = 100_000,
                        prob_func=None,
                        seed: int = 42,
                        show_progress: bool = True) -> SimulationResult:
    """Run full tournament Monte Carlo simulation.

    Args:
        teams: All 68 Team objects (with normalized_params populated)
        n_simulations: Number of tournament simulations to run
        prob_func: Optional custom probability function(team_a, team_b) -> float
        seed: Random seed for reproducibility
        show_progress: Show tqdm progress bar
    """
    rng = np.random.default_rng(seed)

    # Organize teams by region and seed
    region_teams: Dict[str, Dict[int, Team]] = defaultdict(dict)
    for t in teams:
        if t.region and t.seed:
            region_teams[t.region][t.seed] = t

    result = SimulationResult(n_simulations=n_simulations)
    result.champion_counts = defaultdict(int)
    result.final_four_counts = defaultdict(int)
    result.elite_eight_counts = defaultdict(int)
    result.sweet_sixteen_counts = defaultdict(int)
    result.round_of_32_counts = defaultdict(int)

    iterator = range(n_simulations)
    if show_progress:
        iterator = tqdm(iterator, desc="Simulating tournaments")

    for _ in iterator:
        final_four = []
        all_results = {}

        for region in REGIONS:
            rt = region_teams.get(region, {})
            if len(rt) < 2:
                continue
            champion, results = simulate_region(rt, rng, prob_func)
            final_four.append(champion)
            all_results.update(results)

            # Count per-round advancements
            # Tags: R64 = won R64 (reached R32), R32 = won R32 (reached S16),
            #        S16 = won S16 (reached E8), E8 = won E8 (reached F4)
            for team_name, round_reached in results.items():
                if round_reached in ("R64", "R32", "S16", "E8"):
                    result.round_of_32_counts[team_name] += 1
                if round_reached in ("R32", "S16", "E8"):
                    result.sweet_sixteen_counts[team_name] += 1
                if round_reached in ("S16", "E8"):
                    result.elite_eight_counts[team_name] += 1

        # Final Four
        if len(final_four) < 4:
            continue

        for t in final_four:
            result.final_four_counts[t.name] += 1

        # Semifinals (bracket order: East vs West, South vs Midwest)
        semi1 = simulate_single_game(final_four[0], final_four[1], rng, prob_func)
        semi2 = simulate_single_game(final_four[2], final_four[3], rng, prob_func)

        # Championship
        champion = simulate_single_game(semi1, semi2, rng, prob_func)
        result.champion_counts[champion.name] += 1

    return result


def print_results(result: SimulationResult, top_n: int = 20) -> None:
    """Print formatted simulation results."""
    print(f"\n{'='*70}")
    print(f"  NCAA 2026 MARCH MADNESS PREDICTIONS ({result.n_simulations:,} simulations)")
    print(f"{'='*70}\n")

    odds = result.championship_odds()

    print(f"{'Rank':<6}{'Team':<22}{'Champ %':>9}{'F4 %':>9}{'E8 %':>9}{'S16 %':>9}{'R32 %':>9}")
    print(f"{'-'*73}")

    for i, (team, prob) in enumerate(list(odds.items())[:top_n], 1):
        n = result.n_simulations
        f4  = result.final_four_counts.get(team, 0)    / n * 100
        e8  = result.elite_eight_counts.get(team, 0)   / n * 100
        s16 = result.sweet_sixteen_counts.get(team, 0) / n * 100
        r32 = result.round_of_32_counts.get(team, 0)   / n * 100
        print(f"{i:<6}{team:<22}{prob*100:>8.1f}%{f4:>8.1f}%{e8:>8.1f}%{s16:>8.1f}%{r32:>8.1f}%")

    # Biggest upset risks
    print(f"\n{'='*70}")
    print("  UPSET ALERTS (higher seed with >25% win probability)")
    print(f"{'='*70}\n")

    e8_odds = result.advancement_odds("Elite Eight")
    for team, prob in e8_odds.items():
        if prob > 0.05:
            pass
