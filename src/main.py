"""NCAA 2026 March Madness Champion Predictor -- Main Entry Point.

Runs the full prediction pipeline:
  Phase 1A: Weighted Composite + Logistic + Monte Carlo
  Phase 1B: XGBoost ML Pipeline
  Ensemble: Blend 1A + 1B
  Phase 7: Round-Specific Chaos Adjustments
  Phase 8: Recency Weighting
  Phases 2-5: Injury-Adjusted Predictions
  Phase 9: Dual Bracket Output (Healthy + Injury-Adjusted)
  Output: Power rankings, bracket picks, championship odds
"""

import sys
import os
import time
import copy
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
from src.weights import CORE_WEIGHTS, ACTIVE_WEIGHTS, MONTE_CARLO_SIMS, ENSEMBLE_LAMBDA, TOURNAMENT_CHAOS, DATASET_CONFIG
from src.ensemble import blend_all_matchups, disagreement_report


def main():
    print("=" * 70)
    print("  NCAA 2026 MARCH MADNESS CHAMPION PREDICTOR")
    print("  Dual-Track Engine: Weighted Composite + XGBoost ML")
    print("  + Injury Model + Recency Weighting + Round-Specific Chaos")
    print("=" * 70)

    # ── Step 1: Load Data ─────────────────────────────────────────────
    print("\n[1/9] Loading team data from archives...")
    teams = load_all_teams()
    print(f"  Loaded {len(teams)} teams")

    # ── Step 2: Enrich with Niche Metrics ─────────────────────────────
    print("[2/9] Computing niche metrics (scoring runs, resilience, foul trouble)...")
    enrich_niche(teams)

    # ── Step 2b: Phase 8 -- Recency Weighting ─────────────────────────
    if DATASET_CONFIG.get("use_recency_weighting", False):
        print("[2b/9] Computing recency-weighted metrics from game logs...")
        try:
            from src.recency import compute_recency_metrics, enrich_teams_with_recency, recency_report
            recency_data = compute_recency_metrics(teams)
            enrich_teams_with_recency(teams, recency_data)
            print(f"  Recency metrics computed for {len(recency_data)} teams")
            print(f"  Momentum now blended with recency weighting (decay λ=0.95)")
        except Exception as e:
            import traceback
            print(f"  WARNING: Recency weighting failed ({e}). Using original momentum.")
            traceback.print_exc()
            recency_data = {}
    else:
        recency_data = {}

    # ── Save healthy (pre-injury) team copies for dual bracket ────────
    if DATASET_CONFIG.get("use_dual_brackets", False):
        teams_healthy = copy.deepcopy(teams)
    else:
        teams_healthy = None

    # ── Step 3: Phase 2-3 -- Injury Degradation (Pre-Composite) ──────
    injury_profiles = {}
    if DATASET_CONFIG.get("use_injury_model", False):
        print("[3/9] Loading injuries and computing player impact...")
        try:
            from src.injury_model import (
                load_injuries, quantify_player_impacts,
                apply_injury_degradation, injury_impact_report,
                compute_star_isolation,
            )
            injuries = load_injuries()
            if injuries:
                # Load player data for BPR analysis
                from src.player_matchup import load_player_data
                player_df = load_player_data()
                if player_df.empty:
                    print("  WARNING: EvanMiya_Players.csv not loaded — using notes-based fallback")

                injury_profiles = quantify_player_impacts(injuries, player_df)

                # Star Isolation Index: detect teams where one player IS the offense
                sii_results = compute_star_isolation(injury_profiles, player_df)
                if sii_results:
                    severe = {t: s for t, s in sii_results.items() if s >= 0.12}
                    mild = {t: s for t, s in sii_results.items() if 0.08 <= s < 0.12}
                    if severe:
                        print(f"  STAR ISOLATION (severe): {severe}")
                    if mild:
                        print(f"  STAR ISOLATION (mild): {mild}")

                teams = apply_injury_degradation(teams, injury_profiles, round_name="R64")
                print(f"  {len(injuries)} injuries loaded across {len(injury_profiles)} teams")
                print(f"  Team parameters degraded for R64 availability")

                # Print injury report
                print(f"\n{injury_impact_report(injury_profiles, 'R64')}")
            else:
                print("  No injuries to process")
        except Exception as e:
            import traceback
            print(f"  WARNING: Injury model failed ({e}). Proceeding without adjustments.")
            traceback.print_exc()
    else:
        print("[3/9] Injury model disabled -- skipping")

    # ── Step 4: Phase 1A -- Composite Scoring ─────────────────────────
    print("[4/9] Phase 1A: Computing team strengths (32 parameters, weighted composite)...")
    teams = compute_team_strengths(teams)
    ranked = rank_teams(teams)

    print("\n  POWER RANKINGS (Top 25)")
    print(f"  {'Rank':<6}{'Seed':<6}{'Team':<25}{'Strength':>10}{'AdjEM':>8}{'NET':>5}{'BARTH':>7}{'PPG':>6}{'Ht(m)':>6}{'Region':<10}")
    print(f"  {'-'*86}")
    for i, t in enumerate(ranked[:25], 1):
        print(f"  {i:<6}{t.seed:<6}{t.name:<25}{t.team_strength:>10.4f}{t.adj_em:>8.1f}{t.net_rating:>5d}{t.barthag:>7.3f}{t.ppg:>6.1f}{t.eff_height:>6.2f}  {t.region}")

    # ── Step 4b: P&R Proxy Metrics ───────────────────────────────────
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

    # ── Step 5: Cinderella Detection ──────────────────────────────────
    print(f"\n{cinderella_report(teams)}")

    # ── Step 6: Matchup Analysis ──────────────────────────────────────
    print("\n[5/9] Loading matchups and computing win probabilities...")
    h2h_lookup = build_season_h2h()
    matchups = load_matchups(teams, h2h_lookup)
    matchups = compute_all_matchup_probabilities(matchups)

    # Generate pros/cons
    for m in matchups:
        generate_pros_cons(m)

    # ── Step 7: Phase 1B -- XGBoost ───────────────────────────────────
    print("[6/9] Phase 1B: Training XGBoost on historical tournament data...")
    xgb_model = _run_phase_1b(teams, matchups)

    # ── Step 8: Ensemble ──────────────────────────────────────────────
    print("[7/9] Blending Phase 1A + 1B predictions (ensemble)...")
    matchups = blend_all_matchups(matchups, ENSEMBLE_LAMBDA)

    # ── Step 8b: Phase 5 -- Post-ensemble injury vacuum penalty ──────
    if DATASET_CONFIG.get("use_injury_model", False) and injury_profiles:
        try:
            from src.injury_model import apply_star_vacuum_penalty, injury_matchup_flags
            matchups = apply_star_vacuum_penalty(matchups, injury_profiles, round_name="R64")
            print("  Post-ensemble star vacuum penalties applied")
        except Exception as e:
            print(f"  WARNING: Star vacuum penalty failed ({e})")

    # ── Step 8c: Load tournament box scores & blend upstream ──────────
    tourney_profiles = {}
    try:
        from src.tournament_loader import load_tournament_box_scores, tournament_profile_report
        tourney_profiles = load_tournament_box_scores()
        if tourney_profiles:
            print(f"  Tournament box scores loaded for {len(tourney_profiles)} teams")
            for t in teams:
                tp = tourney_profiles.get(t.name)
                if tp and tp.n_games >= 1:
                    t.tourney_efg = tp.efg
                    t.tourney_orb_pct = tp.orb_pct
                    t.tourney_paint_pct = tp.paint_pct
                    t.tourney_ast_rate = tp.ast_rate
                    t.tourney_tov_pct = tp.tov_pct
                    t.tourney_bench_pct = tp.bench_pct
                    t.tourney_games = tp.n_games
                    t.traj_fg_pct = tp.traj_fg_pct
                    t.traj_ast = tp.traj_ast
                    t.traj_tov = tp.traj_tov
                    t.traj_paint = tp.traj_paint
                    t.traj_bench = tp.traj_bench
                    t.n_accelerating = tp.n_accelerating
            accel_teams = [(t.name, t.n_accelerating) for t in teams if t.n_accelerating >= 2]
            if accel_teams:
                print(f"  ACCELERATING teams (2+ improving stats): {accel_teams}")
    except Exception as e:
        import traceback
        print(f"  WARNING: Tournament box-score loader failed ({e})")
        traceback.print_exc()

    # ── Step 8c2: Load tournament momentum ─────────────────────────────
    tourney_momentum = {}
    try:
        from src.tournament_momentum import load_tournament_momentum, momentum_report
        tourney_momentum = load_tournament_momentum()
        if tourney_momentum:
            hot = [(t, p.total) for t, p in tourney_momentum.items() if p.total >= 0.03]
            if hot:
                print(f"  Tournament momentum loaded for {len(tourney_momentum)} teams")
                print(f"  HOT momentum: {hot}")
    except Exception as e:
        import traceback
        print(f"  WARNING: Tournament momentum loader failed ({e})")
        traceback.print_exc()

    # ── Step 8d: Scenario Engine — evidence-based 4-category reasoning ──
    scenario_results = {}
    if DATASET_CONFIG.get("use_branch_engine", True):
        try:
            from src.scenario_engine import ScenarioContext, evaluate_scenario, scenario_report

            # BUG FIX #1: use explicit variable reference instead of 'sii_results' in dir()
            _sii = sii_results if injury_profiles else {}

            print("  [SCENARIO ENGINE] 4-category evidence analysis per matchup...")
            for m in matchups:
                ctx = _build_scenario_context(m, injury_profiles, _sii, tourney_profiles, tourney_momentum)
                result = evaluate_scenario(ctx)
                scenario_results[(m.team_a.name, m.team_b.name)] = result

                old_p = m.win_prob_a_ensemble
                m.win_prob_a_ensemble = result.p_final

                if abs(result.total_shift) > 0.005:
                    fired_cats = sum(1 for c in result.categories if abs(c.final_shift) > 0.001)
                    print(f"    {m.team_a.name} vs {m.team_b.name}: "
                          f"{old_p:.1%} -> {m.win_prob_a_ensemble:.1%} "
                          f"({result.total_shift:+.1%}) "
                          f"[{fired_cats} categories, coherence={result.coherence:.0%}]")

            print(f"  Scenario engine applied to {len(matchups)} matchups")
        except Exception as e:
            import traceback
            print(f"  WARNING: Scenario engine failed ({e}). Using ensemble probabilities.")
            traceback.print_exc()

    # Print matchup predictions
    _print_matchup_predictions(matchups)

    # Disagreement report
    print(f"\n{disagreement_report(matchups)}")

    # Injury matchup flags
    if DATASET_CONFIG.get("use_injury_model", False) and injury_profiles:
        try:
            from src.injury_model import injury_matchup_flags
            print(f"\n{injury_matchup_flags(matchups, injury_profiles, 'R64')}")
        except Exception:
            pass

    # ── Step 9: Monte Carlo Simulation ────────────────────────────────
    n_sims = MONTE_CARLO_SIMS
    print(f"\n[8/9] Running Monte Carlo simulation ({n_sims:,} tournaments)...")
    start = time.time()

    # Build ensemble prob_func: blends 1A + 1B per matchup using calibrated lambda
    ensemble_prob_func = _build_ensemble_prob_func(xgb_model, h2h_lookup)

    # ── Narrative Verification Layer (post-ensemble, pre-MC) ──────────
    narrative_data = {}
    if DATASET_CONFIG.get("use_narrative_layer", False):
        try:
            from src.narrative_layer import load_narratives, build_narrative_prob_func
            narrative_data = load_narratives()
            if narrative_data:
                ensemble_prob_func = build_narrative_prob_func(
                    ensemble_prob_func, narrative_data, injury_profiles
                )
                print(f"  Narrative adjustments loaded for {len(narrative_data)} teams")
            else:
                print("  No narrative data found — skipping")
        except Exception as e:
            import traceback
            print(f"  WARNING: Narrative layer failed ({e}). Proceeding without.")
            traceback.print_exc()
    else:
        print("  Narrative layer disabled — skipping")

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

    # Recency report
    if recency_data:
        try:
            from src.recency import recency_report as rec_report
            print(f"\n{rec_report(teams, recency_data)}")
        except Exception:
            pass

    # Narrative audit report
    if narrative_data:
        try:
            from src.narrative_layer import narrative_audit_report
            print(f"\n{narrative_audit_report(narrative_data, matchups)}")
        except Exception:
            pass

    # Scenario engine fired-only report
    if scenario_results:
        try:
            from src.scenario_engine import scenario_report
            print(f"\n{'='*80}")
            print("  SCENARIO ENGINE: FIRED-ONLY REPORTS")
            print(f"{'='*80}")
            for (a_name, b_name), result in scenario_results.items():
                if abs(result.total_shift) > 0.005:
                    print(scenario_report(result))
        except Exception:
            pass

    # Tournament box-score profiles
    if tourney_profiles:
        try:
            from src.tournament_loader import tournament_profile_report
            print(f"\n{tournament_profile_report(tourney_profiles)}")
        except Exception:
            pass

    # Social media verification (diagnostic only — does NOT touch probabilities)
    social_report_text = ""
    try:
        from src.social_validation import (
            load_social_signals, verify_signals, social_verification_report,
        )
        social_signals = load_social_signals()
        if social_signals:
            matchup_results = {
                f"{m.team_a.name} vs {m.team_b.name}": m.win_prob_a_ensemble
                for m in matchups
            }
            teams_dict = {t.name: t for t in teams}
            verifications = verify_signals(
                social_signals, matchup_results,
                injury_profiles=injury_profiles,
                teams_dict=teams_dict,
                round_name="R64",
            )
            if verifications:
                social_report_text = social_verification_report(verifications)
                print(f"\n{social_report_text}")
    except Exception:
        pass

    # Save results and dashboard
    _save_results(teams, matchups, result_ensemble, injury_profiles)

    from src.dashboard import save_dashboard, generate_full_report
    save_dashboard(teams, matchups, result_ensemble, prob_func=ensemble_prob_func)

    # Append social verification to full_report.txt (diagnostic section)
    if social_report_text:
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "results", "full_report.txt"
        )
        with open(report_path, "a") as f:
            f.write("\n\n" + social_report_text + "\n")

    # ── Step 10: Dual Bracket (Healthy vs Injury-Adjusted) ───────────
    if teams_healthy is not None and DATASET_CONFIG.get("use_dual_brackets", False):
        print(f"\n[9/9] Generating healthy bracket for side-by-side comparison...")
        _run_healthy_bracket(teams_healthy, h2h_lookup, n_sims, result_ensemble, injury_profiles)
    else:
        print(f"\n[9/9] Dual brackets disabled -- skipping healthy bracket")

    print("\n" + "=" * 70)
    print("  PREDICTION COMPLETE")
    print("  Results saved to data/results/")
    print("=" * 70)


def _run_healthy_bracket(teams_healthy, h2h_lookup, n_sims, result_injury, injury_profiles):
    """Run the model WITHOUT injury adjustments for side-by-side comparison."""
    from src.composite import compute_team_strengths
    from src.recency import compute_recency_metrics, enrich_teams_with_recency

    # Apply recency but NOT injuries
    if DATASET_CONFIG.get("use_recency_weighting", False):
        try:
            recency_data = compute_recency_metrics(teams_healthy)
            enrich_teams_with_recency(teams_healthy, recency_data)
        except Exception:
            pass

    teams_healthy = compute_team_strengths(teams_healthy)

    # Build healthy prob_func (no injury adjustments)
    from src.equations import composite_score_differential, win_probability_logistic
    from src.ensemble import _sigmoid
    from src.weights import ROUND_CHAOS

    def healthy_prob_func(team_a, team_b):
        z = composite_score_differential(
            team_a.normalized_params, team_b.normalized_params, ACTIVE_WEIGHTS
        )
        h2h_margin = h2h_lookup.get((team_a.name, team_b.name), 0.0)
        if h2h_margin != 0.0:
            z += np.clip(h2h_margin / 30.0, -0.05, 0.05) * 0.5
        return win_probability_logistic(z, k=14.0)

    result_healthy = simulate_tournament(teams_healthy, n_simulations=n_sims,
                                          prob_func=healthy_prob_func,
                                          seed=42, show_progress=False)

    # Compare and print delta report
    _print_dual_bracket_comparison(result_healthy, result_injury, injury_profiles)


def _print_dual_bracket_comparison(result_healthy, result_injury, injury_profiles):
    """Print side-by-side comparison of healthy vs injury-adjusted brackets."""
    print(f"\n{'='*80}")
    print("  DUAL BRACKET COMPARISON: HEALTHY vs INJURY-ADJUSTED")
    print(f"{'='*80}\n")

    odds_h = result_healthy.championship_odds()
    odds_i = result_injury.championship_odds()

    # Get all teams that appear in either
    all_teams = set(list(odds_h.keys())[:30]) | set(list(odds_i.keys())[:30])

    # Sort by injury-adjusted championship odds
    team_data = []
    for team in all_teams:
        h_pct = odds_h.get(team, 0) * 100
        i_pct = odds_i.get(team, 0) * 100
        delta = i_pct - h_pct
        team_data.append((team, h_pct, i_pct, delta))

    team_data.sort(key=lambda x: x[2], reverse=True)

    print(f"  {'Team':<22}{'Healthy %':>10}{'Injury %':>10}{'Delta':>10}  {'Impact'}")
    print(f"  {'-'*70}")

    for team, h_pct, i_pct, delta in team_data[:25]:
        impact = ""
        if abs(delta) > 2.0:
            impact = "*** MAJOR SHIFT ***"
        elif abs(delta) > 0.5:
            impact = "* notable *"

        # Check if team has injuries
        profile = injury_profiles.get(team)
        if profile and profile.has_star_carrier_out:
            impact += " [STAR OUT]"

        delta_str = f"{delta:+.1f}%" if delta != 0 else "  --"
        print(f"  {team:<22}{h_pct:>9.1f}%{i_pct:>9.1f}%{delta_str:>10}  {impact}")

    # Biggest movers
    print(f"\n  BIGGEST INJURY BENEFICIARIES (opponents weakened):")
    risers = sorted(team_data, key=lambda x: x[3], reverse=True)[:5]
    for team, h_pct, i_pct, delta in risers:
        if delta > 0.1:
            print(f"    {team:<22} {h_pct:.1f}% -> {i_pct:.1f}% ({delta:+.1f}%)")

    print(f"\n  BIGGEST INJURY CASUALTIES:")
    fallers = sorted(team_data, key=lambda x: x[3])[:5]
    for team, h_pct, i_pct, delta in fallers:
        if delta < -0.1:
            print(f"    {team:<22} {h_pct:.1f}% -> {i_pct:.1f}% ({delta:+.1f}%)")

    # Save comparison CSV
    import pandas as pd
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "results")
    comp_data = [{"team": t, "healthy_champ_pct": h, "injury_champ_pct": i, "delta_pct": d}
                 for t, h, i, d in team_data]
    pd.DataFrame(comp_data).to_csv(
        os.path.join(results_dir, "dual_bracket_comparison.csv"), index=False
    )
    print(f"\n  Saved: dual_bracket_comparison.csv")


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
            ACTIVE_WEIGHTS,
        )

        # H2H season tiebreaker
        h2h_margin = h2h_lookup.get((team_a.name, team_b.name), 0.0)
        if h2h_margin != 0.0:
            z += np.clip(h2h_margin / 30.0, -0.05, 0.05) * 0.5

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

        # NOTE: Tournament chaos is now applied per-round in _get_win_prob()
        # so we do NOT apply it here (would double-apply).

        _cache[key] = p_ens
        _cache[(team_b.name, team_a.name)] = 1.0 - p_ens
        return p_ens

    # BUG FIX #2-6: Wrap with scenario engine for MC simulation
    # - Full context from Team objects (not just subset)
    # - round_name threaded from MC caller
    # - Cache keyed by (team_a, team_b, round_name) to prevent staleness
    if DATASET_CONFIG.get("use_branch_engine", True):
        _raw_prob_func = prob_func
        _scenario_cache = {}

        def prob_func_with_scenarios(team_a, team_b, round_name="R64"):
            key = (team_a.name, team_b.name, round_name)
            if key in _scenario_cache:
                return _scenario_cache[key]

            p_base = _raw_prob_func(team_a, team_b)

            try:
                from src.scenario_engine import ScenarioContext, evaluate_scenario
                ctx = ScenarioContext(
                    team_a_name=team_a.name,
                    team_b_name=team_b.name,
                    p_base=p_base,
                    round_name=round_name,
                    seed_a=team_a.seed,
                    seed_b=team_b.seed,
                    bench_depth_a=team_a.bds,
                    bench_depth_b=team_b.bds,
                    orb_pct_a=team_a.orb_pct,
                    orb_pct_b=team_b.orb_pct,
                    drb_pct_a=team_a.drb_pct,
                    drb_pct_b=team_b.drb_pct,
                    rbm_a=team_a.rbm,
                    rbm_b=team_b.rbm,
                    conference_a=team_a.conference,
                    conference_b=team_b.conference,
                    three_pt_share_a=team_a.three_pa_fga,
                    three_pt_share_b=team_b.three_pa_fga,
                    three_pt_pct_a=team_a.three_p_pct,
                    three_pt_pct_b=team_b.three_p_pct,
                    three_pt_std_a=team_a.three_pt_std,
                    three_pt_std_b=team_b.three_pt_std,
                    pace_a=team_a.pace,
                    pace_b=team_b.pace,
                    tourney_orb_pct_a=team_a.tourney_orb_pct,
                    tourney_orb_pct_b=team_b.tourney_orb_pct,
                    tourney_paint_pct_a=team_a.tourney_paint_pct,
                    tourney_paint_pct_b=team_b.tourney_paint_pct,
                    tourney_ast_rate_a=team_a.tourney_ast_rate,
                    tourney_ast_rate_b=team_b.tourney_ast_rate,
                    tourney_games_a=team_a.tourney_games,
                    tourney_games_b=team_b.tourney_games,
                    form_trend_a=team_a.form_trend,
                    form_trend_b=team_b.form_trend,
                    momentum_a=team_a.momentum,
                    momentum_b=team_b.momentum,
                    offensive_burst_a=team_a.offensive_burst,
                    offensive_burst_b=team_b.offensive_burst,
                    q3_adj_a=team_a.q3_adj_strength,
                    q3_adj_b=team_b.q3_adj_strength,
                    trajectory_fg_pct_a=team_a.traj_fg_pct,
                    trajectory_fg_pct_b=team_b.traj_fg_pct,
                    trajectory_ast_a=team_a.traj_ast,
                    trajectory_ast_b=team_b.traj_ast,
                    trajectory_tov_a=team_a.traj_tov,
                    trajectory_tov_b=team_b.traj_tov,
                    trajectory_paint_a=team_a.traj_paint,
                    trajectory_paint_b=team_b.traj_paint,
                    trajectory_bench_a=team_a.traj_bench,
                    trajectory_bench_b=team_b.traj_bench,
                    team_exp_a=team_a.exp,
                    team_exp_b=team_b.exp,
                    star_ppg_a=team_a.best_player_above_avg_pts + 12.0 if team_a.best_player_above_avg_pts > 0 else 0.0,
                    star_ppg_b=team_b.best_player_above_avg_pts + 12.0 if team_b.best_player_above_avg_pts > 0 else 0.0,
                    first_tourney_a=team_a.exp < 1.5,
                    first_tourney_b=team_b.exp < 1.5,
                    foul_trouble_rate_a=team_a.foul_trouble_impact,
                    foul_trouble_rate_b=team_b.foul_trouble_impact,
                )
                result = evaluate_scenario(ctx)
                p_final = result.p_final
            except Exception:
                p_final = p_base

            _scenario_cache[key] = p_final
            _scenario_cache[(team_b.name, team_a.name, round_name)] = 1.0 - p_final
            return p_final

        return prob_func_with_scenarios

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


_RECENT_FINAL_FOURS = {
    "Connecticut": 2,   # 2023 champion, 2024 champion
    "Alabama": 1,       # 2024 Final Four
    "Purdue": 1,        # 2024 championship game
    "Houston": 1,       # 2024 Sweet 16, 2023 Final Four
    "Duke": 1,          # 2022 Final Four
    "Michigan State": 1, # deep tournament runs, Izzo DNA
    "Florida": 1,       # 2023 R32 exit but 2024 run
}


def _build_scenario_context(matchup, injury_profiles, sii_results, tourney_profiles=None, tourney_momentum=None):
    """Build ScenarioContext from a Matchup + all available data sources."""
    from src.scenario_engine import ScenarioContext
    a = matchup.team_a
    b = matchup.team_b

    prof_a = injury_profiles.get(a.name)
    prof_b = injury_profiles.get(b.name)

    star_lost_a = prof_a.has_star_carrier_out if prof_a else False
    star_lost_b = prof_b.has_star_carrier_out if prof_b else False
    star_bpr_a = prof_a.top_player_bpr_share if prof_a and hasattr(prof_a, 'top_player_bpr_share') else 0.0
    star_bpr_b = prof_b.top_player_bpr_share if prof_b and hasattr(prof_b, 'top_player_bpr_share') else 0.0
    sii_a = sii_results.get(a.name, 0.0) if sii_results else 0.0
    sii_b = sii_results.get(b.name, 0.0) if sii_results else 0.0

    inj_pen_a = getattr(prof_a, 'total_adj_em_penalty', 0.0) if prof_a else 0.0
    inj_pen_b = getattr(prof_b, 'total_adj_em_penalty', 0.0) if prof_b else 0.0

    tp_a = tourney_profiles.get(a.name) if tourney_profiles else None
    tp_b = tourney_profiles.get(b.name) if tourney_profiles else None

    ctx = ScenarioContext(
        team_a_name=a.name,
        team_b_name=b.name,
        p_base=matchup.win_prob_a_ensemble,
        round_name=matchup.round_name,
        seed_a=a.seed,
        seed_b=b.seed,

        # ROSTER STATE
        injury_penalty_a=abs(inj_pen_a),
        injury_penalty_b=abs(inj_pen_b),
        star_lost_a=star_lost_a,
        star_lost_b=star_lost_b,
        star_bpr_share_a=star_bpr_a,
        star_bpr_share_b=star_bpr_b,
        sii_a=sii_a,
        sii_b=sii_b,
        crippled_roster_a=getattr(prof_a, 'has_crippled_roster', False) if prof_a else False,
        crippled_roster_b=getattr(prof_b, 'has_crippled_roster', False) if prof_b else False,
        crippled_weeks_out_a=getattr(prof_a, 'crippled_weeks_out', 0.0) if prof_a else 0.0,
        crippled_weeks_out_b=getattr(prof_b, 'crippled_weeks_out', 0.0) if prof_b else 0.0,
        bench_depth_a=a.bds,
        bench_depth_b=b.bds,
        foul_trouble_rate_a=a.foul_trouble_impact,
        foul_trouble_rate_b=b.foul_trouble_impact,

        # MATCHUP STYLE
        orb_pct_a=a.orb_pct,
        orb_pct_b=b.orb_pct,
        drb_pct_a=a.drb_pct,
        drb_pct_b=b.drb_pct,
        rbm_a=a.rbm,
        rbm_b=b.rbm,
        conference_a=a.conference,
        conference_b=b.conference,
        three_pt_share_a=a.three_pa_fga,
        three_pt_share_b=b.three_pa_fga,
        three_pt_pct_a=a.three_p_pct,
        three_pt_pct_b=b.three_p_pct,
        three_pt_std_a=a.three_pt_std,
        three_pt_std_b=b.three_pt_std,
        pace_a=a.pace,
        pace_b=b.pace,
        tourney_orb_pct_a=a.tourney_orb_pct,
        tourney_orb_pct_b=b.tourney_orb_pct,
        tourney_paint_pct_a=a.tourney_paint_pct,
        tourney_paint_pct_b=b.tourney_paint_pct,
        tourney_ast_rate_a=a.tourney_ast_rate,
        tourney_ast_rate_b=b.tourney_ast_rate,
        tourney_games_a=a.tourney_games,
        tourney_games_b=b.tourney_games,

        # FORM & TRAJECTORY
        form_trend_a=a.form_trend,
        form_trend_b=b.form_trend,
        momentum_a=a.momentum,
        momentum_b=b.momentum,
        tourney_momentum_a=tourney_momentum.get(a.name).total if tourney_momentum and tourney_momentum.get(a.name) else 0.0,
        tourney_momentum_b=tourney_momentum.get(b.name).total if tourney_momentum and tourney_momentum.get(b.name) else 0.0,
        offensive_burst_a=a.offensive_burst,
        offensive_burst_b=b.offensive_burst,
        q3_adj_a=a.q3_adj_strength,
        q3_adj_b=b.q3_adj_strength,
        trajectory_fg_pct_a=a.traj_fg_pct,
        trajectory_fg_pct_b=b.traj_fg_pct,
        trajectory_ast_a=a.traj_ast,
        trajectory_ast_b=b.traj_ast,
        trajectory_tov_a=a.traj_tov,
        trajectory_tov_b=b.traj_tov,
        trajectory_paint_a=a.traj_paint,
        trajectory_paint_b=b.traj_paint,
        trajectory_bench_a=a.traj_bench,
        trajectory_bench_b=b.traj_bench,
        tourney_efg_a=a.tourney_efg,
        tourney_efg_b=b.tourney_efg,

        # INTANGIBLES
        team_exp_a=a.exp,
        team_exp_b=b.exp,
        star_ppg_a=a.best_player_above_avg_pts + 12.0 if a.best_player_above_avg_pts > 0 else 0.0,
        star_ppg_b=b.best_player_above_avg_pts + 12.0 if b.best_player_above_avg_pts > 0 else 0.0,
        first_tourney_a=a.exp < 1.5,
        first_tourney_b=b.exp < 1.5,
        recent_final_fours_a=_RECENT_FINAL_FOURS.get(a.name, 0),
        recent_final_fours_b=_RECENT_FINAL_FOURS.get(b.name, 0),
        sos_rank_a=getattr(a, 'sos_rank', 50),
        sos_rank_b=getattr(b, 'sos_rank', 50),
        q1_wins_a=int(getattr(a, 'q1_record', 0.5) * 10),
        q1_wins_b=int(getattr(b, 'q1_record', 0.5) * 10),
    )

    return ctx


def _print_bracket_picks(matchups, result):
    """Print recommended bracket picks with full round-by-round progression."""
    print(f"\n{'='*70}")
    print("  RECOMMENDED BRACKET PICKS")
    print(f"{'='*70}\n")

    odds = result.championship_odds()

    rounds = [
        ("ROUND OF 32 (Top 16 most likely to advance)", "Round of 32", 16),
        ("SWEET SIXTEEN (Top 16)", "Sweet Sixteen", 16),
        ("ELITE EIGHT (Top 12)", "Elite Eight", 12),
        ("FINAL FOUR (Top 8)", "Final Four", 8),
        ("CHAMPIONSHIP GAME (Top 4)", "Champion", 4),
    ]

    for title, round_name, show_n in rounds:
        adv = result.advancement_odds(round_name)
        print(f"  {title}:")
        for team, prob in list(adv.items())[:show_n]:
            print(f"    {team:<24s} {prob*100:5.1f}%")
        print()

    champ_name = list(odds.items())[0][0]
    champ_prob = list(odds.items())[0][1]
    print(f"  PREDICTED CHAMPION: {champ_name}")
    print(f"  Championship probability: {champ_prob*100:.1f}%")


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


def _save_results(teams, matchups, result, injury_profiles=None):
    """Save results to CSV files."""
    import pandas as pd

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Power rankings
    ranked = sorted(teams, key=lambda t: t.team_strength, reverse=True)
    rankings_data = []
    for i, t in enumerate(ranked, 1):
        row = {
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
            "form_trend": round(getattr(t, "form_trend", 0.5), 3),
            "momentum": round(t.momentum, 3),
        }
        # Add injury penalty if available
        if injury_profiles:
            profile = injury_profiles.get(t.name)
            if profile:
                from src.injury_model import _get_round_penalty
                row["injury_penalty_r64"] = round(_get_round_penalty(t.name, injury_profiles, "R64"), 3)
                row["star_carrier_out"] = profile.has_star_carrier_out
            else:
                row["injury_penalty_r64"] = 0.0
                row["star_carrier_out"] = False
        rankings_data.append(row)
    pd.DataFrame(rankings_data).to_csv(
        os.path.join(results_dir, "power_rankings.csv"), index=False
    )

    # Championship odds (all rounds)
    odds = result.championship_odds()
    n = result.n_simulations
    odds_data = [{"team": t,
                  "championship_pct": round(p * 100, 2),
                  "final_four_pct": round(result.final_four_counts.get(t, 0) / n * 100, 2),
                  "elite_eight_pct": round(result.elite_eight_counts.get(t, 0) / n * 100, 2),
                  "sweet_sixteen_pct": round(result.sweet_sixteen_counts.get(t, 0) / n * 100, 2),
                  "round_of_32_pct": round(result.round_of_32_counts.get(t, 0) / n * 100, 2)}
                 for t, p in odds.items()]
    pd.DataFrame(odds_data).to_csv(
        os.path.join(results_dir, "championship_odds.csv"), index=False
    )

    # Matchup predictions
    matchup_data = []
    for m in matchups:
        row = {
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
        }
        # Add injury flags
        if injury_profiles:
            from src.injury_model import _get_round_penalty
            pen_a = _get_round_penalty(m.team_a.name, injury_profiles, "R64")
            pen_b = _get_round_penalty(m.team_b.name, injury_profiles, "R64")
            row["injury_penalty_a"] = round(pen_a, 3)
            row["injury_penalty_b"] = round(pen_b, 3)
            row["injury_shift"] = round(abs(pen_a - pen_b) * 0.02, 3)
            row["injury_flag"] = abs(pen_a - pen_b) * 0.02 > 0.03
        matchup_data.append(row)
    pd.DataFrame(matchup_data).to_csv(
        os.path.join(results_dir, "matchup_predictions.csv"), index=False
    )

    print(f"\n  Saved: power_rankings.csv, championship_odds.csv, matchup_predictions.csv")


if __name__ == "__main__":
    main()
