"""NCAA 2026 March Madness Champion Predictor -- Main Entry Point.

Runs the full prediction pipeline:
  Phase 1A: Weighted Composite + Logistic + Monte Carlo
  Phase 1B: XGBoost ML Pipeline
  Ensemble: Blend 1A + 1B
  Output: Power rankings, bracket picks, championship odds
"""

import sys
import os
import time
import numpy as np
from collections import defaultdict

from src.data_loader import load_all_teams, load_matchups, load_historical_games, build_season_h2h
from src.composite import (
    compute_team_strengths, rank_teams,
    compute_all_matchup_probabilities, generate_pros_cons,
)
from src.niche import enrich_niche
from src.cinderella import cinderella_report
from src.monte_carlo import simulate_tournament, print_results
from src.weights import CORE_WEIGHTS, MONTE_CARLO_SIMS, ENSEMBLE_LAMBDA
from src.ensemble import blend_all_matchups, disagreement_report


def main():
    print("=" * 70)
    print("  NCAA 2026 MARCH MADNESS CHAMPION PREDICTOR")
    print("  Dual-Track Engine: Weighted Composite + XGBoost ML")
    print("=" * 70)

    # ── Step 1: Load Data ─────────────────────────────────────────────
    print("\n[1/7] Loading team data from archives...")
    teams = load_all_teams()
    print(f"  Loaded {len(teams)} teams")

    # ── Step 2: Enrich with Niche Metrics ─────────────────────────────
    print("[2/7] Computing niche metrics (scoring runs, resilience, foul trouble)...")
    enrich_niche(teams)

    # ── Step 3: Phase 1A -- Composite Scoring ─────────────────────────
    print("[3/7] Phase 1A: Computing team strengths (31 parameters, weighted composite)...")
    teams = compute_team_strengths(teams)
    ranked = rank_teams(teams)

    print("\n  POWER RANKINGS (Top 25)")
    print(f"  {'Rank':<6}{'Seed':<6}{'Team':<25}{'Strength':>10}{'AdjEM':>8}{'NET':>5}{'BARTH':>7}{'PPG':>6}{'Ht(m)':>6}{'Region':<10}")
    print(f"  {'-'*86}")
    for i, t in enumerate(ranked[:25], 1):
        print(f"  {i:<6}{t.seed:<6}{t.name:<25}{t.team_strength:>10.4f}{t.adj_em:>8.1f}{t.net_rating:>5d}{t.barthag:>7.3f}{t.ppg:>6.1f}{t.eff_height:>6.2f}  {t.region}")

    # ── Step 3b: P&R Proxy Metrics ───────────────────────────────────
    try:
        from src.player_matchup import load_player_data, compute_pnr_metrics
        _player_df = load_player_data()
        if not _player_df.empty:
            compute_pnr_metrics(teams, _player_df)
            pnr_teams = [(t.name, t.big_man_offense, t.rim_defense_bpr)
                         for t in teams if t.big_man_offense > 0 or t.rim_defense_bpr > 0]
            print(f"  P&R proxy metrics computed for {len(pnr_teams)} teams")
    except Exception:
        pass

    # ── Step 4: Cinderella Detection ──────────────────────────────────
    print(f"\n{cinderella_report(teams)}")

    # ── Step 5: Matchup Analysis ──────────────────────────────────────
    print("\n[4/7] Loading matchups and computing win probabilities...")
    h2h_lookup = build_season_h2h()
    matchups = load_matchups(teams, h2h_lookup)
    matchups = compute_all_matchup_probabilities(matchups)

    # Generate pros/cons
    for m in matchups:
        generate_pros_cons(m)

    # ── Step 6: Phase 1B -- XGBoost ───────────────────────────────────
    print("[5/7] Phase 1B: Training XGBoost on historical tournament data...")
    xgb_model = _run_phase_1b(teams, matchups)

    # ── Step 7: Ensemble ──────────────────────────────────────────────
    print("[6/7] Blending Phase 1A + 1B predictions (ensemble)...")
    matchups = blend_all_matchups(matchups, ENSEMBLE_LAMBDA)

    # Print matchup predictions
    _print_matchup_predictions(matchups)

    # Disagreement report
    print(f"\n{disagreement_report(matchups)}")

    # ── Step 8: Monte Carlo Simulation ────────────────────────────────
    n_sims = MONTE_CARLO_SIMS
    print(f"\n[7/7] Running Monte Carlo simulation ({n_sims:,} tournaments)...")
    start = time.time()

    # Build ensemble prob_func: blends 1A + 1B per matchup using calibrated lambda
    ensemble_prob_func = _build_ensemble_prob_func(xgb_model, h2h_lookup)
    result_ensemble = simulate_tournament(teams, n_simulations=n_sims,
                                           prob_func=ensemble_prob_func,
                                           seed=42, show_progress=True)

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    # ── Output Results ────────────────────────────────────────────────
    print_results(result_ensemble, top_n=25)

    # Detailed bracket picks
    _print_bracket_picks(matchups, result_ensemble)

    # Upset probabilities
    _print_upset_analysis(matchups, result_ensemble)

    # Upset detection model flags
    try:
        from src.upset_model import print_upset_flags
        print(f"\n{print_upset_flags(matchups)}")
    except Exception:
        pass

    # Player matchup sandbox
    try:
        from src.player_matchup import player_matchup_report
        print(f"\n{player_matchup_report(matchups, teams)}")
    except Exception:
        pass

    # Save results and dashboard
    _save_results(teams, matchups, result_ensemble)

    from src.dashboard import save_dashboard, generate_full_report
    save_dashboard(teams, matchups, result_ensemble)

    print("\n" + "=" * 70)
    print("  PREDICTION COMPLETE")
    print("  Results saved to data/results/")
    print("=" * 70)


def _run_phase_1b(teams, matchups):
    """Run Phase 1B XGBoost pipeline."""
    try:
        from src.xgboost_model import (
            prepare_historical_features, train_xgboost, predict_matchup,
        )
        from src.data_loader import load_historical_games

        hist_df, matchups_df = load_historical_games()
        X, y = prepare_historical_features(hist_df, matchups_df)

        if len(X) == 0:
            print("  WARNING: No historical features generated. Using 1A only.")
            for m in matchups:
                m.win_prob_a_1b = m.win_prob_a_1a
            return None

        print(f"  Training on {len(X)} historical matchup samples...")
        model, params = train_xgboost(X, y)

        if model is None:
            print("  WARNING: XGBoost not available. Using 1A only.")
            for m in matchups:
                m.win_prob_a_1b = m.win_prob_a_1a
            return None

        # Compute 1B probabilities for all matchups (pass 1A as soft anchor)
        for m in matchups:
            m.win_prob_a_1b = predict_matchup(model, m.team_a, m.team_b,
                                              p_1a=m.win_prob_a_1a)

        print(f"  XGBoost trained successfully. Phase 1B predictions computed.")
        return model

    except Exception as e:
        import traceback
        print(f"  WARNING: Phase 1B failed ({e}). Falling back to Phase 1A only.")
        traceback.print_exc()
        for m in matchups:
            m.win_prob_a_1b = m.win_prob_a_1a
        return None


def _build_ensemble_prob_func(xgb_model, h2h_lookup):
    """Build a probability function that blends 1A + 1B per matchup.

    This replaces the old dual-simulation approach: instead of running
    separate 1A and 1B MC simulations and merging counts 50/50, we
    run a single simulation where each game uses the sigmoid-blended
    ensemble probability (calibrated lambda=0.65).
    """
    from src.equations import (
        composite_score_differential, win_probability_logistic,
    )
    from src.ensemble import _sigmoid

    _cache = {}

    def prob_func(team_a, team_b):
        key = (team_a.name, team_b.name)
        if key in _cache:
            return _cache[key]

        # Phase 1A: composite + logistic + scoring_margin_std volatility
        z = composite_score_differential(
            team_a.normalized_params,
            team_b.normalized_params,
            CORE_WEIGHTS,
        )

        # H2H season tiebreaker
        h2h_margin = h2h_lookup.get((team_a.name, team_b.name), 0.0)
        if h2h_margin != 0.0:
            z += np.clip(h2h_margin / 30.0, -0.05, 0.05) * 0.5

        # P&R tactical counter: only penalize teams that actually rely on
        # big-man P&R offense (BMO >= 5.0) when facing strong rim defense.
        # Guard-driven teams (low BMO) are unaffected.
        bmo_a = getattr(team_a, "big_man_offense", 0.0)
        bmo_b = getattr(team_b, "big_man_offense", 0.0)
        rd_a = getattr(team_a, "rim_defense_bpr", 0.0)
        rd_b = getattr(team_b, "rim_defense_bpr", 0.0)
        pnr_a_countered = max(0.0, rd_b - bmo_a) if bmo_a >= 5.0 else 0.0
        pnr_b_countered = max(0.0, rd_a - bmo_b) if bmo_b >= 5.0 else 0.0
        pnr_net = (pnr_b_countered - pnr_a_countered) * 0.015
        z += np.clip(pnr_net, -0.025, 0.025)

        p_1a = win_probability_logistic(z, k=14.0)

        # scoring_margin_std volatility (sole volatility layer)
        std_a = getattr(team_a, "scoring_margin_std", 0)
        std_b = getattr(team_b, "scoring_margin_std", 0)
        combined_vol = (std_a + std_b) / 2.0
        if combined_vol > 12.0:
            vol_pull = min(0.06, (combined_vol - 12.0) * 0.01)
            p_1a = p_1a * (1.0 - vol_pull) + 0.5 * vol_pull

        # Phase 1B: XGBoost
        if xgb_model is not None:
            try:
                from src.xgboost_model import predict_matchup
                p_1b = predict_matchup(xgb_model, team_a, team_b)
            except Exception:
                p_1b = p_1a
        else:
            p_1b = p_1a

        # Sigmoid-weighted ensemble blend (same as ensemble.py)
        conf_1a = abs(p_1a - 0.5)
        conf_1b = abs(p_1b - 0.5)
        conf_diff = conf_1a - conf_1b
        shift = 0.20 * (2.0 * _sigmoid(conf_diff, scale=8.0) - 1.0)
        lam = np.clip(ENSEMBLE_LAMBDA + shift, 0.20, 0.85)
        p_ens = lam * p_1a + (1.0 - lam) * p_1b

        _cache[key] = p_ens
        _cache[(team_b.name, team_a.name)] = 1.0 - p_ens
        return p_ens

    return prob_func


def _print_matchup_predictions(matchups):
    """Print Round of 64 predictions."""
    print(f"\n{'='*70}")
    print("  ROUND OF 64 PREDICTIONS")
    print(f"{'='*70}\n")

    current_region = ""
    for m in matchups:
        if m.region != current_region:
            current_region = m.region
            print(f"\n  --- {current_region.upper()} REGION ---\n")

        # Determine predicted winner
        if m.win_prob_a_ensemble > 0.5:
            winner = m.team_a.name
            prob = m.win_prob_a_ensemble
        else:
            winner = m.team_b.name
            prob = 1 - m.win_prob_a_ensemble

        conf_label = "LOCK" if m.confidence > 0.9 else "LEAN" if m.confidence > 0.7 else "TOSS-UP"

        upset = ""
        if m.team_a.seed > m.team_b.seed and m.win_prob_a_ensemble > 0.5:
            upset = " ** UPSET **"
        elif m.team_b.seed > m.team_a.seed and m.win_prob_a_ensemble < 0.5:
            upset = " ** UPSET **"

        print(f"  ({m.team_a.seed:2d}) {m.team_a.name:<22s} vs ({m.team_b.seed:2d}) {m.team_b.name:<22s}")
        print(f"       -> {winner} ({prob:.1%}) [{conf_label}]{upset}")

        # Top pros/cons
        if m.pros_a[:2] or m.pros_b[:2]:
            if m.win_prob_a_ensemble > 0.5:
                for p in m.pros_a[:2]:
                    print(f"          + {p}")
                for c in m.cons_a[:1]:
                    print(f"          - {c}")
            else:
                for p in m.pros_b[:2]:
                    print(f"          + {p}")
                for c in m.cons_b[:1]:
                    print(f"          - {c}")
        print()


def _print_bracket_picks(matchups, result):
    """Print recommended bracket picks."""
    print(f"\n{'='*70}")
    print("  RECOMMENDED BRACKET PICKS")
    print(f"{'='*70}\n")

    odds = result.championship_odds()
    f4 = result.advancement_odds("Final Four")

    print("  FINAL FOUR:")
    f4_teams = list(f4.items())[:8]
    for team, prob in f4_teams[:4]:
        print(f"    {team}: {prob*100:.1f}%")

    print("\n  CHAMPIONSHIP GAME:")
    top2 = list(odds.items())[:2]
    for team, prob in top2:
        print(f"    {team}: {prob*100:.1f}%")

    print(f"\n  PREDICTED CHAMPION: {list(odds.items())[0][0]}")
    print(f"  Championship probability: {list(odds.items())[0][1]*100:.1f}%")


def _print_upset_analysis(matchups, result):
    """Print upset probability analysis with volatile-favorite flagging."""
    print(f"\n{'='*70}")
    print("  UPSET WATCH")
    print(f"{'='*70}\n")

    upsets = []
    for m in matchups:
        seed_diff = abs(m.team_a.seed - m.team_b.seed)
        if seed_diff < 3:
            continue

        higher_seed = m.team_a if m.team_a.seed > m.team_b.seed else m.team_b
        lower_seed = m.team_b if m.team_a.seed > m.team_b.seed else m.team_a

        if m.team_a.seed > m.team_b.seed:
            upset_prob = m.win_prob_a_ensemble
        else:
            upset_prob = 1 - m.win_prob_a_ensemble

        if upset_prob > 0.20:
            upsets.append((higher_seed, lower_seed, upset_prob, m))

    upsets.sort(key=lambda x: x[2], reverse=True)

    for hs, ls, prob, m in upsets:
        danger = "HIGH" if prob > 0.40 else "MEDIUM" if prob > 0.30 else "LOW"
        print(f"  ({hs.seed}) {hs.name} over ({ls.seed}) {ls.name}")
        print(f"       Upset probability: {prob:.1%} [{danger} RISK]")

        # Flag volatile favorites
        fav = ls  # lower seed = expected favorite
        if fav.scoring_margin_std > 13.0:
            print(f"       !! VOLATILE FAVORITE: {fav.name} margin_std={fav.scoring_margin_std:.1f} (high variance)")
        print()

    # Volatile favorites summary
    print(f"  --- VOLATILE FAVORITES (high scoring_margin_std) ---\n")
    volatile = [(m, m.team_a if m.team_a.seed <= m.team_b.seed else m.team_b)
                for m in matchups]
    volatile = [(m, fav) for m, fav in volatile
                if fav.scoring_margin_std > 12.0 and fav.seed <= 6]
    volatile.sort(key=lambda x: -x[1].scoring_margin_std)

    if volatile:
        for m, fav in volatile:
            underdog = m.team_b if fav == m.team_a else m.team_a
            print(f"  ({fav.seed}) {fav.name:<22s} std={fav.scoring_margin_std:5.1f}  "
                  f"vs ({underdog.seed}) {underdog.name}")
    else:
        print("  No volatile favorites detected.")


def _save_results(teams, matchups, result):
    """Save results to CSV files."""
    import pandas as pd

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Power rankings
    ranked = sorted(teams, key=lambda t: t.team_strength, reverse=True)
    rankings_data = []
    for i, t in enumerate(ranked, 1):
        rankings_data.append({
            "rank": i,
            "team": t.name,
            "seed": t.seed,
            "region": t.region,
            "team_strength": round(t.team_strength, 4),
            "adj_em": round(t.adj_em, 2),
            "net_rating": t.net_rating,
            "barthag": round(t.barthag, 3),
            "ppg": round(t.ppg, 1),
            "eff_height_m": round(t.eff_height, 2),
            "shooting_eff": round(t.shooting_eff, 4),
            "scoring_balance": round(t.scoring_balance, 4),
            "q1_record": round(t.q1_record, 3),
            "clutch": round(t.clutch_factor, 3),
            "experience": round(t.exp, 2),
            "legacy_factor": round(t.legacy_factor, 1),
            "z_rating": round(t.z_rating, 2),
            "injury_rank": t.injury_rank,
            "star_above_avg": round(t.star_above_avg, 3),
            "cinderella": t.is_cinderella,
        })
    pd.DataFrame(rankings_data).to_csv(
        os.path.join(results_dir, "power_rankings.csv"), index=False
    )

    # Championship odds
    odds = result.championship_odds()
    odds_data = [{"team": t, "championship_pct": round(p * 100, 2),
                  "final_four_pct": round(result.final_four_counts.get(t, 0) / result.n_simulations * 100, 2),
                  "elite_eight_pct": round(result.elite_eight_counts.get(t, 0) / result.n_simulations * 100, 2)}
                 for t, p in odds.items()]
    pd.DataFrame(odds_data).to_csv(
        os.path.join(results_dir, "championship_odds.csv"), index=False
    )

    # Matchup predictions
    matchup_data = []
    for m in matchups:
        matchup_data.append({
            "region": m.region,
            "team_a": m.team_a.name,
            "seed_a": m.team_a.seed,
            "team_b": m.team_b.name,
            "seed_b": m.team_b.seed,
            "prob_a_1a": round(m.win_prob_a_1a, 3),
            "prob_a_1b": round(m.win_prob_a_1b, 3),
            "prob_a_ensemble": round(m.win_prob_a_ensemble, 3),
            "confidence": round(m.confidence, 3),
            "predicted_winner": m.team_a.name if m.win_prob_a_ensemble > 0.5 else m.team_b.name,
        })
    pd.DataFrame(matchup_data).to_csv(
        os.path.join(results_dir, "matchup_predictions.csv"), index=False
    )

    print(f"\n  Saved: power_rankings.csv, championship_odds.csv, matchup_predictions.csv")


if __name__ == "__main__":
    main()
