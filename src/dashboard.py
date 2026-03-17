"""Results dashboard: generates comprehensive visual output.

Can be run standalone or imported for notebook use.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict

from src.models import Team, Matchup, SimulationResult
from src.weights import CORE_WEIGHTS


def generate_full_report(teams: List[Team],
                         matchups: List[Matchup],
                         result: SimulationResult) -> str:
    """Generate a comprehensive text-based dashboard."""
    lines = []

    # Header
    lines.append("=" * 72)
    lines.append("  NCAA 2026 MARCH MADNESS CHAMPION PREDICTOR -- FULL DASHBOARD")
    lines.append(f"  Simulations: {result.n_simulations:,} | Parameters: {len(CORE_WEIGHTS)}")
    lines.append("=" * 72)

    # Championship Odds Table
    lines.append("\n" + "=" * 72)
    lines.append("  CHAMPIONSHIP ODDS (Top 30)")
    lines.append("=" * 72)
    lines.append(f"  {'#':<4}{'Team':<25}{'Seed':<6}{'Region':<10}"
                 f"{'Champ%':>8}{'F4%':>8}{'E8%':>8}{'S16%':>8}")
    lines.append("  " + "-" * 68)

    odds = result.championship_odds()
    for i, (team, prob) in enumerate(list(odds.items())[:30], 1):
        f4 = result.final_four_counts.get(team, 0) / result.n_simulations * 100
        e8 = result.elite_eight_counts.get(team, 0) / result.n_simulations * 100
        s16 = result.sweet_sixteen_counts.get(team, 0) / result.n_simulations * 100
        t_obj = next((t for t in teams if t.name == team), None)
        seed = t_obj.seed if t_obj else 0
        region = t_obj.region if t_obj else ""
        lines.append(
            f"  {i:<4}{team:<25}{seed:<6}{region:<10}"
            f"{prob*100:>7.1f}%{f4:>7.1f}%{e8:>7.1f}%{s16:>7.1f}%"
        )

    # Region Breakdown
    for region in ["East", "West", "South", "Midwest"]:
        lines.append(f"\n  --- {region.upper()} REGION ---")
        region_teams = [t for t in teams if t.region == region]
        region_teams.sort(key=lambda t: result.champion_counts.get(t.name, 0),
                          reverse=True)
        for t in region_teams[:8]:
            champ = result.champion_counts.get(t.name, 0) / result.n_simulations * 100
            f4 = result.final_four_counts.get(t.name, 0) / result.n_simulations * 100
            lines.append(f"    ({t.seed:2d}) {t.name:<25} Champ: {champ:5.1f}% | F4: {f4:5.1f}%")

    # Round-by-Round Bracket Picks
    lines.append("\n" + "=" * 72)
    lines.append("  ROUND OF 64 PREDICTIONS (Detailed)")
    lines.append("=" * 72)

    current_region = ""
    for m in matchups:
        if m.region != current_region:
            current_region = m.region
            lines.append(f"\n  === {current_region.upper()} ===")

        winner = m.team_a if m.win_prob_a_ensemble > 0.5 else m.team_b
        prob = m.win_prob_a_ensemble if m.win_prob_a_ensemble > 0.5 else 1 - m.win_prob_a_ensemble
        confidence = m.confidence

        upset = "UPSET" if (
            (m.team_a.seed > m.team_b.seed and m.win_prob_a_ensemble > 0.5) or
            (m.team_b.seed > m.team_a.seed and m.win_prob_a_ensemble < 0.5)
        ) else ""

        conf_label = "LOCK" if confidence > 0.9 else "SOLID" if confidence > 0.8 else "LEAN" if prob > 0.6 else "COIN FLIP"

        lines.append(
            f"  ({m.team_a.seed:2d}) {m.team_a.name:<20s} vs ({m.team_b.seed:2d}) {m.team_b.name:<20s}"
        )
        lines.append(
            f"       PICK: {winner.name} ({prob:.1%}) [{conf_label}]"
            f"  1A:{m.win_prob_a_1a:.1%} | 1B:{m.win_prob_a_1b:.1%} | ENS:{m.win_prob_a_ensemble:.1%}"
            f"  {upset}"
        )

        # Key matchup factors
        factors = []
        for p in (m.pros_a if m.win_prob_a_ensemble > 0.5 else m.pros_b)[:2]:
            factors.append(f"+ {p}")
        for c in (m.cons_a if m.win_prob_a_ensemble > 0.5 else m.cons_b)[:1]:
            factors.append(f"- {c}")
        for f in factors:
            lines.append(f"         {f}")
        lines.append("")

    # Weight Configuration
    lines.append("\n" + "=" * 72)
    lines.append("  WEIGHT CONFIGURATION")
    lines.append("=" * 72)
    sorted_weights = sorted(CORE_WEIGHTS.items(), key=lambda x: x[1], reverse=True)
    for param, weight in sorted_weights:
        bar = "█" * int(weight * 200)
        lines.append(f"  {param:<25s} {weight:>5.1%} {bar}")

    # Cinderella Alerts
    cinderellas = [t for t in teams if t.is_cinderella]
    if cinderellas:
        lines.append("\n" + "=" * 72)
        lines.append("  CINDERELLA ALERTS")
        lines.append("=" * 72)
        for t in cinderellas:
            champ = result.champion_counts.get(t.name, 0) / result.n_simulations * 100
            lines.append(
                f"  * {t.name} ({t.seed}-seed, {t.region})"
                f" -- AdjEM Rank #{t.adj_em_rank}, Exp={t.exp:.1f}, Champ={champ:.2f}%"
            )

    return "\n".join(lines)


def save_dashboard(teams, matchups, result, output_dir=None):
    """Save all dashboard outputs to files."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "results"
        )
    os.makedirs(output_dir, exist_ok=True)

    # Full text report
    report = generate_full_report(teams, matchups, result)
    with open(os.path.join(output_dir, "full_report.txt"), "w") as f:
        f.write(report)

    # Team parameters export
    params_data = []
    for t in sorted(teams, key=lambda x: x.team_strength, reverse=True):
        row = {
            "team": t.name, "seed": t.seed, "region": t.region,
            "team_strength": round(t.team_strength, 4),
            "adj_em": round(t.adj_em, 2),
            "shooting_eff": round(t.shooting_eff, 4),
            "scoring_balance": round(t.scoring_balance, 4),
            "clutch_factor": round(t.clutch_factor, 3),
            "sos": round(t.sos, 2),
            "to_pct_adj": round(t.to_pct, 4),
            "orb_pct": round(t.orb_pct, 3),
            "drb_pct": round(t.drb_pct, 3),
            "seed_score": round(t.seed_score, 3),
            "top50_perf": round(t.top50_perf, 3),
            "ftr": round(t.ftr, 3),
            "ast_pct": round(t.ast_pct, 3),
            "spi": round(t.spi, 3),
            "exp": round(t.exp, 2),
            "dvi": round(t.dvi, 3),
            "opp_to_pct_adj": round(t.opp_to_pct, 4),
            "rpi_rim": round(t.rpi_rim, 3),
            "eff_height_m": round(t.eff_height, 2),
            "momentum": round(t.momentum, 3),
            "ctf": round(t.ctf, 3),
            "q1_record": round(t.q1_record, 3),
            "q34_loss_rate": round(t.q34_loss_rate, 3),
            "msrp": round(t.msrp, 2),
            "fragility": round(t.fragility_score, 3),
            "march_readiness": round(t.march_readiness, 3),
            "legacy_factor": round(t.legacy_factor, 1),
            "net_rating": t.net_rating,
            "injury_rank": t.injury_rank,
            "barthag": round(t.barthag, 3),
            "ppg": round(t.ppg, 1),
            "opp_ppg": round(t.opp_ppg, 1),
            "z_rating": round(t.z_rating, 2),
            "star_above_avg": round(t.star_above_avg, 3),
            "consistency": round(t.consistency, 1),
            "cinderella": t.is_cinderella,
            "champ_pct": round(
                result.champion_counts.get(t.name, 0) / result.n_simulations * 100, 2
            ),
        }
        params_data.append(row)

    pd.DataFrame(params_data).to_csv(
        os.path.join(output_dir, "team_parameters.csv"), index=False
    )

    print(f"  Dashboard saved to {output_dir}/")
    print(f"    - full_report.txt")
    print(f"    - team_parameters.csv")
    print(f"    - power_rankings.csv")
    print(f"    - championship_odds.csv")
    print(f"    - matchup_predictions.csv")
