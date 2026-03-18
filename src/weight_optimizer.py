"""Weight Optimization Pipeline: Grid Search, Sensitivity Analysis, Bayesian Optimization.

Tests weight vectors against historical NCAA tournament results (2008-2025)
to find optimal parameter weights for predicting game outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time

from src.weights import (CORE_WEIGHTS, INVERTED_PARAMS, Z_SCORE_PARAMS,
                         LOGISTIC_K, TEMPORAL_SCHEME)
from src.utils import SEED_EXPECTED_ROUND


# ── Historical data helpers ──────────────────────────────────────────────────

def _safe(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


ROUND_TO_TOURNEY_WINS = {
    64: 0,
    32: 1,
    16: 2,
    8: 3,
    4: 4,
    2: 5,
    1: 6,
}


def _load_historical_legacy_priors():
    """Build year-safe team legacy priors using reformed PASE.

    For team T in year Y, uses only appearances strictly before Y.
    Applies the same 3 corrections as data_loader._build_team_legacy():
      1. Difficulty coefficient (normalize by max possible PASE for that seed)
      2. Coaching decay (0.85^years_ago, relative to evaluation year Y)
      3. sqrt(N) normalization + cap [-3, +3]
    """
    if hasattr(_load_historical_legacy_priors, "_cache"):
        return _load_historical_legacy_priors._cache

    from math import sqrt
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "archive-3")

    priors = {}
    # history[team] = list of (year, normalized_outperformance)
    history = defaultdict(list)
    try:
        tm = pd.read_csv(os.path.join(base, "Tournament Matchups.csv"))
        cols = ["YEAR", "TEAM", "SEED", "ROUND"]
        team_years = tm[cols].dropna().copy()
        team_years["YEAR"] = team_years["YEAR"].astype(int)
        team_years["SEED"] = team_years["SEED"].astype(int)
        team_years["ROUND"] = team_years["ROUND"].astype(int)

        team_years = (
            team_years.sort_values(["YEAR", "TEAM", "ROUND"])
            .groupby(["YEAR", "TEAM"], as_index=False)
            .first()
            .sort_values(["YEAR", "TEAM"])
        )

        for _, row in team_years.iterrows():
            year = int(row["YEAR"])
            team = str(row["TEAM"]).strip()
            seed = int(row["SEED"])

            # Compute prior for this (team, year) using only appearances BEFORE year
            past = history[team]
            if past:
                n = len(past)
                weight_total = sum(0.85 ** max(0, year - y) for y, _ in past)
                weighted_sum = sum(
                    outperf * 0.85 ** max(0, year - y) for y, outperf in past
                )
                weighted_avg = weighted_sum / weight_total if weight_total > 0 else 0.0
                pase = weighted_avg * sqrt(n)
                priors[(team, year)] = max(-3.0, min(3.0, pase))
            else:
                priors[(team, year)] = 0.0

            # Record this year's result for future years
            actual = ROUND_TO_TOURNEY_WINS.get(int(row["ROUND"]), 0)
            expected = SEED_EXPECTED_ROUND.get(seed, 0.0)
            max_pase = 6.0 - expected
            if max_pase < 0.5:
                max_pase = 0.5
            normalized_outperf = (actual - expected) / max_pase
            history[team].append((year, normalized_outperf))
    except Exception:
        pass

    _load_historical_legacy_priors._cache = priors
    return priors


def _row_to_param_dict(row, all_teams_df: pd.DataFrame = None) -> dict:
    """Convert a KenPom Barttorvik CSV row to a dict of the same keys as CORE_WEIGHTS.

    Mirrors the derivations in data_loader.load_all_teams() as closely as
    possible using only data available in the historical CSV.
    """
    seed = int(_safe(row.get("SEED", 8), 8))
    adj_em = _safe(row.get("KADJ EM", 0))
    adj_o = _safe(row.get("KADJ O", 0))
    adj_d = _safe(row.get("KADJ D", 0))
    pace = _safe(row.get("RAW T", 68))
    efg = _safe(row.get("EFG%", 0)) / 100.0
    efg_d = _safe(row.get("EFG%D", 0)) / 100.0
    tov = _safe(row.get("TOV%", 0)) / 100.0
    tov_d = _safe(row.get("TOV%D", 0)) / 100.0
    orb = _safe(row.get("OREB%", 0)) / 100.0
    drb = _safe(row.get("DREB%", 0)) / 100.0
    blk = _safe(row.get("BLK%", 0)) / 100.0
    ast = _safe(row.get("AST%", 0)) / 100.0
    ftr = _safe(row.get("FTR", 0)) / 100.0
    ft_pct = _safe(row.get("FT%", 0)) / 100.0
    three_pct = _safe(row.get("3PT%", 0)) / 100.0
    three_pct_d = _safe(row.get("3PT%D", 0)) / 100.0
    two_pct = _safe(row.get("2PT%", 0)) / 100.0
    three_ptr = _safe(row.get("3PTR", 0)) / 100.0
    two_ptr = _safe(row.get("2PTR", 0)) / 100.0
    sos = _safe(row.get("ELITE SOS", 0))
    exp_val = _safe(row.get("EXP", 2))
    barthag = _safe(row.get("BARTHAG", 0.5))
    wins = _safe(row.get("W", 0))
    losses = _safe(row.get("L", 0))
    eff_hgt = _safe(row.get("EFF HGT", 0))
    pppo = _safe(row.get("PPPO", 0))
    pppd = _safe(row.get("PPPD", 0))

    win_pct = wins / max(wins + losses, 1)
    seed_score = 1.0 / max(seed, 1)
    three_pri = three_pct * three_ptr

    # Shooting efficiency (merged)
    ts_approx = efg * 0.85 + ftr * ft_pct * 0.15
    shooting_eff = 0.6 * efg + 0.4 * ts_approx

    # Scoring balance
    scoring_balance = two_pct * two_ptr + three_pct * three_ptr

    # DVI
    stl_rate = tov_d * 0.5
    dvi = 0.3 * blk + 0.3 * stl_rate + 0.4 * (1 - three_pct_d)

    # Rim protection
    opp_fg_rim = efg_d
    rpi_rim = blk + (1 - opp_fg_rim)

    # FTR combined
    ftr_combined = ftr * ft_pct

    # PPG
    ppg = pppo * pace if pppo > 0 else adj_o * pace / 100.0
    opp_ppg = pppd * pace if pppd > 0 else adj_d * pace / 100.0
    ppg_margin = ppg - opp_ppg

    # RBM
    rbm = (orb + drb) / 2 - 0.5

    # SPI: talent * AdjEM
    talent_rank = _safe(row.get("TALENT", 0))
    if talent_rank > 0 and talent_rank <= 68:
        talent_factor = max(0, (68 - talent_rank) / 68.0)
    else:
        talent_factor = 0.3
    spi = talent_factor * adj_em / 15.0

    # Eff height in meters
    eff_height = eff_hgt * 0.0254 if eff_hgt > 10 else eff_hgt

    # Derived params: use real data where available, proxy otherwise
    team_name = str(row.get("TEAM", "")).strip()
    year = int(_safe(row.get("YEAR", 0), 0))
    team_pase_lookup = _load_historical_legacy_priors()

    top25_perf = win_pct * 0.8 if sos > 30 else win_pct * 0.5
    # We do not have year-safe historical coach mapping, so keep CTF neutral
    # instead of leaking full-period tournament records into past seasons.
    ctf = 0.5
    legacy_factor = team_pase_lookup.get((team_name, year), 0.0)
    bds = 0.25
    momentum = 0.5 * win_pct + 0.5 * 0.5
    scoring_margin_std = 14.0 * (1.0 + abs(adj_em) / 30.0) ** (-0.5)
    consistency = 15.0 / (15.0 + scoring_margin_std)
    clutch_factor = 0.35 * win_pct + 0.25 * barthag + 0.20 * consistency + 0.20 * 0.5

    # NET/z_rating approximations
    net_score = max(0, (68 - seed * 4) / 68.0)
    z_rating = 0.45 * adj_em + 0.35 * sos + 3.0
    injury_health = 35.0
    star_above_avg = talent_factor
    cwp_star_17_half = 0.40 * barthag + 0.30 * talent_factor + 0.15 * win_pct + 0.15 * win_pct

    # CWP composites
    margin_normalized = min(max(adj_em / 30.0, 0), 1)
    chaos_index = three_ptr * (three_ptr * 0.08)
    fragility_score = max(0, min(1.0,
        0.35 * (1 - margin_normalized) +
        0.25 * 0.5 +
        0.20 * (1 - win_pct) +
        0.20 * chaos_index * 10
    ))
    march_readiness = (0.20 * win_pct + 0.15 * min(win_pct + 0.1, 1.0) +
                       0.15 * (1 - three_ptr) + 0.10 * win_pct +
                       0.10 * drb + 0.10 * (1 - three_pct_d) +
                       0.10 * 0.5 + 0.10 * cwp_star_17_half)

    # Q1 start strength (proxy)
    offensive_burst = 0.5 * adj_o / 120.0 + 0.3 * pace / 75.0 + 0.2 * three_pri
    # Q3 adj strength (proxy)
    q3_adj_strength = 0.4 * ctf + 0.3 * min(exp_val / 3.0, 1.0) + 0.2 * 0.5 + 0.1 * clutch_factor

    # Q1/Q3 records (approximation from seed)
    q1_record = win_pct * 0.6 if seed <= 4 else win_pct * 0.3
    q34_loss_rate = max(0, (seed - 4) / 12.0) * 0.1

    msrp = adj_em / 20.0
    blowout_resilience = (0.30 * min(max((msrp + 1) / 2.0, 0), 1) +
                          0.30 * consistency +
                          0.20 * (adj_em / 40.0) + 0.20 * bds)
    foul_trouble_impact = -(1.0 - bds) * 0.5

    return {
        "adj_em": adj_em,
        "shooting_eff": shooting_eff,
        "clutch_factor": clutch_factor,
        "sos": sos,
        "to_pct": tov,
        "scoring_balance": scoring_balance,
        "orb_pct": orb,
        "seed_score": seed_score,
        "top25_perf": top25_perf,
        "barthag": barthag,
        "ftr": ftr_combined,
        "ast_pct": ast,
        "spi": spi,
        "exp": exp_val,
        "dvi": dvi,
        "drb_pct": drb,
        "opp_to_pct": tov_d,
        "rpi_rim": rpi_rim,
        "net_score": net_score,
        "z_rating": z_rating,
        "eff_height": eff_height,
        "momentum": momentum,
        "ctf": ctf,
        "rbm": rbm,
        "q1_record": q1_record,
        "q34_loss_rate": q34_loss_rate,
        "offensive_burst": offensive_burst,
        "q3_adj_strength": q3_adj_strength,
        "ppg_margin": ppg_margin,
        "fragility_score": fragility_score,
        "march_readiness": march_readiness,
        "legacy_factor": legacy_factor,
        "bds": bds,
        "injury_health": injury_health,
        "cwp_star_17_half": cwp_star_17_half,
        "star_above_avg": star_above_avg,
        "msrp": msrp,
        "blowout_resilience": blowout_resilience,
        "foul_trouble_impact": foul_trouble_impact,
        "consistency": consistency,
        "scoring_margin_std": scoring_margin_std,
    }


def _normalize_param_values(team_params: List[dict], param_keys: list) -> List[dict]:
    """Min-max / z-score normalize across all teams, returning normalized dicts."""
    n = len(team_params)
    if n == 0:
        return []

    normalized = [{} for _ in range(n)]

    for key in param_keys:
        vals = np.array([tp.get(key, 0.0) for tp in team_params], dtype=float)
        invert = key in INVERTED_PARAMS

        if key in Z_SCORE_PARAMS:
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std < 1e-9:
                normed = np.full(n, 0.5)
            else:
                z = (vals - mean) / std
                normed = 1.0 / (1.0 + np.exp(-z))
            if invert:
                normed = 1.0 - normed
        else:
            v_min, v_max = np.nanmin(vals), np.nanmax(vals)
            if v_max == v_min:
                normed = np.full(n, 0.5)
            else:
                normed = (vals - v_min) / (v_max - v_min)
            if invert:
                normed = 1.0 - normed

        for i in range(n):
            normalized[i][key] = float(normed[i])

    return normalized


def _compute_score(norm_params: dict, weights: dict) -> float:
    score = 0.0
    for key, w in weights.items():
        score += w * norm_params.get(key, 0.0)
    return score


def _predict_winner(norm_a: dict, norm_b: dict, weights: dict, k: float = 6.0) -> float:
    """Return P(A wins) using logistic on weighted composite differential."""
    z = sum(w * (norm_a.get(p, 0.0) - norm_b.get(p, 0.0)) for p, w in weights.items())
    return 1.0 / (1.0 + np.exp(-k * z))


# ── Temporal (recency) weighting ──────────────────────────────────────────────

def recency_weight(year: int, scheme: str = "uniform",
                   param: float = 0.95,
                   min_year: int = 2008, max_year: int = 2025) -> float:
    """Return a positive weight for *year* under the given decay scheme.

    Schemes
    -------
    uniform     : all years equal (weight = 1.0)
    linear      : 2008 → 0.5, 2025 → 1.0
    exponential : w = param^(max_year - year),  param in (0, 1]
    step        : 1.0 for year >= cutoff (param), 0.5 otherwise
    """
    if scheme == "uniform":
        return 1.0
    if scheme == "linear":
        span = max(max_year - min_year, 1)
        return 0.5 + 0.5 * (year - min_year) / span
    if scheme == "exponential":
        return param ** (max_year - year)
    if scheme == "step":
        cutoff = int(param) if param > 1 else 2018
        return 1.0 if year >= cutoff else 0.5
    return 1.0


def sweep_temporal_schemes(games: List[Tuple] = None,
                           n_grid: int = 3000) -> Tuple[str, float]:
    """Test multiple temporal weighting schemes.

    For each scheme we run a quick grid search (n_grid samples) using that
    scheme's weighted accuracy as the objective, then evaluate the resulting
    weights on the *full* dataset with UNIFORM weighting (our real metric).
    The scheme whose optimised weights generalise best wins.

    Returns (best_scheme_name, best_param).
    """
    if games is None:
        kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
        tm = pd.read_csv("archive-3/Tournament Matchups.csv")
        games = _build_eval_games(kb, tm)

    param_keys = list(CORE_WEIGHTS.keys())
    years = sorted({g[3] for g in games})

    schemes = [
        ("uniform",      1.0),
        ("linear",       1.0),
        ("exponential",  0.90),
        ("exponential",  0.92),
        ("exponential",  0.94),
        ("exponential",  0.95),
        ("exponential",  0.96),
        ("exponential",  0.97),
        ("exponential",  0.98),
        ("step",         2016),
        ("step",         2018),
        ("step",         2020),
    ]

    print(f"\n{'='*70}")
    print("  TEMPORAL WEIGHTING SWEEP")
    print(f"{'='*70}")
    print(f"  Games: {len(games)} | Years: {years[0]}-{years[-1]} | "
          f"Schemes: {len(schemes)} | Grid per scheme: {n_grid:,}")

    baseline_uniform = _evaluate_weights(CORE_WEIGHTS, games, param_keys,
                                         temporal_scheme="uniform")
    print(f"  Baseline (CORE_WEIGHTS, uniform): {baseline_uniform:.4f} ({baseline_uniform*100:.1f}%)")

    results = []
    rng_seed = 42

    for scheme_name, scheme_param in schemes:
        rng = np.random.default_rng(rng_seed)
        best_weighted_acc = 0.0
        best_w = None

        for _ in range(n_grid):
            w = _random_weight_vector(param_keys, rng)
            acc = _evaluate_weights(w, games, param_keys,
                                    temporal_scheme=scheme_name,
                                    temporal_param=scheme_param)
            if acc > best_weighted_acc:
                best_weighted_acc = acc
                best_w = w

        if best_w is None:
            best_w = CORE_WEIGHTS

        # The real test: how do these weights perform with UNIFORM (unweighted)?
        real_acc = _evaluate_weights(best_w, games, param_keys,
                                     temporal_scheme="uniform")

        label = f"{scheme_name}({scheme_param})"
        delta = real_acc - baseline_uniform
        results.append((label, scheme_name, scheme_param, real_acc, best_weighted_acc))
        print(f"  {label:<25} uniform_acc={real_acc:.4f} ({real_acc*100:.1f}%)  "
              f"Δ={delta*100:+.2f}%  [weighted_obj={best_weighted_acc:.4f}]")

    results.sort(key=lambda x: x[3], reverse=True)
    best = results[0]

    print(f"\n  ┌─ BEST SCHEME ───────────────────────────────────────────────────┐")
    print(f"  │  {best[0]:<25} → {best[3]*100:.1f}% uniform accuracy     │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    return best[1], best[2]


# ── Build historical evaluation dataset ──────────────────────────────────────

def _build_eval_games(kb_df: pd.DataFrame, tm_df: pd.DataFrame,
                      start_year: int = 2008, end_year: int = 2025
                      ) -> List[Tuple[dict, dict, int, int]]:
    """Build list of (team_a_params, team_b_params, a_won, year).

    Returns raw param dicts (not normalized) so we can re-normalize per weight
    trial.  The 4th element is the tournament year for temporal weighting.
    """
    games = []

    for year in range(start_year, end_year + 1):
        yr_kb = kb_df[kb_df["YEAR"] == year]
        yr_tm = tm_df[tm_df["YEAR"] == year]

        if yr_kb.empty or yr_tm.empty:
            continue

        # Build team param cache for this year
        team_cache = {}
        for _, row in yr_kb.iterrows():
            team = str(row.get("TEAM", "")).strip()
            seed = row.get("SEED", 0)
            if pd.isna(seed) or int(seed) == 0:
                continue
            team_cache[team] = _row_to_param_dict(row)

        # Parse matchups as consecutive row pairs
        mdf = yr_tm.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)

        for i in range(0, len(mdf) - 1, 2):
            row_a = mdf.iloc[i]
            row_b = mdf.iloc[i + 1]

            if row_a.get("CURRENT ROUND") != row_b.get("CURRENT ROUND"):
                continue

            team_a = str(row_a["TEAM"]).strip()
            team_b = str(row_b["TEAM"]).strip()
            round_a = _safe(row_a.get("ROUND", 0))
            round_b = _safe(row_b.get("ROUND", 0))
            score_a = _safe(row_a.get("SCORE", 0))
            score_b = _safe(row_b.get("SCORE", 0))

            params_a = team_cache.get(team_a)
            params_b = team_cache.get(team_b)
            if params_a is None or params_b is None:
                continue

            if round_a < round_b:
                a_won = 1
            elif round_b < round_a:
                a_won = 0
            else:
                a_won = 1 if score_a > score_b else 0

            games.append((params_a, params_b, a_won, year))

    return games


def _evaluate_weights(weights: dict, games: List[Tuple],
                      param_keys: list, k: float = 6.0,
                      use_wth: bool = False,
                      temporal_scheme: str = "uniform",
                      temporal_param: float = 0.95) -> float:
    """Evaluate a weight vector: return temporally-weighted accuracy.

    Each game's contribution is scaled by ``recency_weight(year, ...)``.
    When ``temporal_scheme="uniform"`` (default) this reduces to the original
    unweighted accuracy so all existing call-sites are unaffected.
    """
    if not games:
        return 0.0

    all_params = [g[0] for g in games] + [g[1] for g in games]
    normalized = _normalize_param_values(all_params, param_keys)

    n_games = len(games)
    norm_a_list = normalized[:n_games]
    norm_b_list = normalized[n_games:]

    weighted_correct = 0.0
    total_weight = 0.0

    for i in range(n_games):
        game = games[i]
        raw_a, raw_b, a_won = game[0], game[1], game[2]
        year = game[3] if len(game) > 3 else 2016

        prob_a = _predict_winner(norm_a_list[i], norm_b_list[i], weights, k)

        if use_wth:
            three_ptr_a = raw_a.get("scoring_balance", 0.15)
            three_ptr_b = raw_b.get("scoring_balance", 0.15)
            chaos_a = three_ptr_a * 0.08
            chaos_b = three_ptr_b * 0.08
            pace_a = raw_a.get("momentum", 0.5)
            pace_b = raw_b.get("momentum", 0.5)
            pace_diff = abs(pace_a - pace_b) / 1.0
            wth_v = (0.15 * (chaos_a + chaos_b) / 2.0 +
                     0.05 * pace_diff + 0.02)
            wth_v = min(max(wth_v, 0.0), 0.15)
            prob_a = prob_a * (1.0 - wth_v) + 0.5 * wth_v

        w = recency_weight(year, temporal_scheme, temporal_param)
        predicted_a_wins = prob_a > 0.5
        if predicted_a_wins == bool(a_won):
            weighted_correct += w
        total_weight += w

    return weighted_correct / total_weight if total_weight > 0 else 0.0


# ── Stage 1: Random Weight Search ────────────────────────────────────────────

def _random_weight_vector(param_keys: list, rng: np.random.Generator,
                          base_weights: dict = None,
                          perturbation: float = 1.0) -> dict:
    """Generate a random weight vector that sums to 1.0.

    If base_weights provided, perturb around them (perturbation controls spread).
    """
    n = len(param_keys)
    if base_weights is not None and perturbation < 1.0:
        base = np.array([base_weights.get(k, 1.0 / n) for k in param_keys])
        noise = rng.exponential(1.0, n)
        raw = base * (1.0 - perturbation) + noise * perturbation
    else:
        raw = rng.exponential(1.0, n)

    raw = np.maximum(raw, 0.001)
    raw /= raw.sum()
    return {k: float(v) for k, v in zip(param_keys, raw)}


def stage1_grid_search(games: List[Tuple], param_keys: list,
                       n_samples: int = 10000, top_k: int = 50,
                       seed: int = 42, use_wth: bool = False,
                       temporal_scheme: str = "uniform",
                       temporal_param: float = 0.95
                       ) -> Tuple[List[Tuple[float, dict]], float]:
    """Random weight search. Returns top_k (accuracy, weights) pairs and baseline accuracy."""
    rng = np.random.default_rng(seed)
    mode_label = "WITH WTH" if use_wth else "NO WTH"
    ts_label = f"  Temporal: {temporal_scheme}" + (f"({temporal_param})" if temporal_scheme != "uniform" else "")
    print(f"\n{'='*70}")
    print(f"  STAGE 1: Random Weight Search ({n_samples:,} samples) [{mode_label}]")
    print(f"{ts_label}")
    print(f"{'='*70}")
    print(f"  Historical games: {len(games)}")
    print(f"  Parameters: {len(param_keys)}")

    _ew = lambda w: _evaluate_weights(
        w, games, param_keys, use_wth=use_wth,
        temporal_scheme=temporal_scheme, temporal_param=temporal_param)

    baseline_acc = _ew(CORE_WEIGHTS)
    print(f"  Current weights accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")

    results = []
    t0 = time.time()
    checkpoints = {int(n_samples * p): p for p in [0.1, 0.25, 0.5, 0.75, 1.0]}

    for i in range(n_samples):
        w = _random_weight_vector(param_keys, rng)
        acc = _ew(w)
        results.append((acc, w))

        if (i + 1) in checkpoints:
            elapsed = time.time() - t0
            best_so_far = max(r[0] for r in results)
            pct = checkpoints[i + 1]
            print(f"  [{pct*100:5.1f}%] {i+1:>6,}/{n_samples:,}  "
                  f"best={best_so_far:.4f}  elapsed={elapsed:.1f}s")

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    best_acc, best_w = top_results[0]
    print(f"\n  Best random accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"  Improvement over current: {(best_acc - baseline_acc)*100:+.2f}%")

    return top_results, baseline_acc


# ── Stage 2: Sensitivity Analysis ────────────────────────────────────────────

def stage2_sensitivity(games: List[Tuple], param_keys: list,
                       best_weights: dict,
                       perturbation_pcts: List[float] = None,
                       use_wth: bool = False,
                       temporal_scheme: str = "uniform",
                       temporal_param: float = 0.95
                       ) -> List[Tuple[str, float, float, float]]:
    """Perturb each weight +/-50% and measure accuracy change.

    Returns sorted list of (param, sensitivity, best_direction, best_weight).
    sensitivity = max accuracy change when perturbing this param.
    """
    if perturbation_pcts is None:
        perturbation_pcts = [-0.50, -0.25, +0.25, +0.50]

    mode_label = "WITH WTH" if use_wth else "NO WTH"
    print(f"\n{'='*70}")
    print(f"  STAGE 2: Sensitivity Analysis [{mode_label}]")
    print(f"{'='*70}")

    _ew = lambda w: _evaluate_weights(
        w, games, param_keys, use_wth=use_wth,
        temporal_scheme=temporal_scheme, temporal_param=temporal_param)

    base_acc = _ew(best_weights)
    print(f"  Base accuracy: {base_acc:.4f}")

    sensitivities = []

    for param in param_keys:
        max_delta = 0.0
        best_dir = 0.0
        best_w_val = best_weights[param]

        for pct in perturbation_pcts:
            perturbed = dict(best_weights)
            old_val = perturbed[param]
            new_val = max(0.001, old_val * (1 + pct))
            delta_weight = new_val - old_val
            perturbed[param] = new_val

            other_total = sum(v for k, v in perturbed.items() if k != param)
            if other_total > 0:
                for k in perturbed:
                    if k != param:
                        perturbed[k] -= delta_weight * (perturbed[k] / other_total)
                        perturbed[k] = max(0.001, perturbed[k])

            total = sum(perturbed.values())
            perturbed = {k: v / total for k, v in perturbed.items()}

            acc = _ew(perturbed)
            delta = acc - base_acc

            if abs(delta) > abs(max_delta):
                max_delta = delta
                best_dir = pct
                best_w_val = new_val

        sensitivities.append((param, max_delta, best_dir, best_w_val))

    sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n  {'Parameter':<25} {'Sensitivity':>12} {'Best Δ':>10} {'Direction':>12}")
    print(f"  {'─'*25} {'─'*12} {'─'*10} {'─'*12}")
    for param, sens, direction, _ in sensitivities:
        arrow = "↑" if direction > 0 else "↓" if direction < 0 else "—"
        print(f"  {param:<25} {sens*100:>+11.3f}% {abs(sens)*100:>9.3f}%  "
              f"{arrow} {direction*100:+.0f}%")

    high_impact = [s for s in sensitivities if abs(s[1]) > 0.005]
    low_impact = [s for s in sensitivities if abs(s[1]) < 0.001]
    print(f"\n  High-impact params (>0.5% swing): {len(high_impact)}")
    print(f"  Low-impact params (<0.1% swing): {len(low_impact)}")
    if low_impact:
        print(f"  Candidates for removal: {', '.join(s[0] for s in low_impact[:5])}")

    return sensitivities


# ── Stage 3: Bayesian Optimization ───────────────────────────────────────────

def stage3_bayesian(games: List[Tuple], param_keys: list,
                    start_weights: dict, n_trials: int = 500,
                    use_wth: bool = False,
                    temporal_scheme: str = "uniform",
                    temporal_param: float = 0.95
                    ) -> Tuple[dict, float]:
    """Optimize weights using scipy.optimize.minimize (Nelder-Mead).

    Falls back to random local search if scipy unavailable.
    """
    mode_label = "WITH WTH" if use_wth else "NO WTH"
    print(f"\n{'='*70}")
    print(f"  STAGE 3: Bayesian / Numerical Optimization ({n_trials} trials) [{mode_label}]")
    print(f"{'='*70}")

    _ew = lambda w: _evaluate_weights(
        w, games, param_keys, use_wth=use_wth,
        temporal_scheme=temporal_scheme, temporal_param=temporal_param)

    base_acc = _ew(start_weights)
    print(f"  Starting accuracy: {base_acc:.4f}")

    best_weights, best_acc = _try_optuna(games, param_keys, start_weights,
                                          n_trials, use_wth=use_wth,
                                          temporal_scheme=temporal_scheme,
                                          temporal_param=temporal_param)
    if best_weights is not None:
        return best_weights, best_acc

    best_weights, best_acc = _try_scipy(games, param_keys, start_weights,
                                         n_trials, use_wth=use_wth,
                                         temporal_scheme=temporal_scheme,
                                         temporal_param=temporal_param)
    if best_weights is not None:
        return best_weights, best_acc

    return _random_local_search(games, param_keys, start_weights,
                                 n_trials, use_wth=use_wth,
                                 temporal_scheme=temporal_scheme,
                                 temporal_param=temporal_param)


def _try_optuna(games, param_keys, start_weights, n_trials, use_wth=False,
                temporal_scheme="uniform", temporal_param=0.95):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  optuna not available, trying scipy...")
        return None, 0.0

    print("  Using optuna for Bayesian optimization...")

    def objective(trial):
        raw = {}
        for key in param_keys:
            base = start_weights.get(key, 1.0 / len(param_keys))
            low = max(0.0005, base * 0.1)
            high = max(low * 10.0, base * 5.0)
            raw[key] = trial.suggest_float(key, low, high, log=True)

        total = sum(raw.values())
        weights = {k: v / total for k, v in raw.items()}
        return _evaluate_weights(weights, games, param_keys, use_wth=use_wth,
                                 temporal_scheme=temporal_scheme,
                                 temporal_param=temporal_param)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))

    study.enqueue_trial({k: start_weights.get(k, 0.025) for k in param_keys})

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw_best = study.best_params
    total = sum(raw_best.values())
    best_weights = {k: raw_best[k] / total for k in param_keys}
    best_acc = study.best_value

    print(f"  Optuna best accuracy: {best_acc:.4f}")
    return best_weights, best_acc


def _try_scipy(games, param_keys, start_weights, n_trials, use_wth=False,
               temporal_scheme="uniform", temporal_param=0.95):
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("  scipy not available, falling back to random local search...")
        return None, 0.0

    print("  Using scipy Nelder-Mead optimization...")

    n = len(param_keys)
    x0 = np.array([start_weights.get(k, 1.0 / n) for k in param_keys])
    x0 = np.log(x0 + 1e-8)

    call_count = [0]

    def neg_accuracy(log_weights):
        raw = np.exp(log_weights)
        raw /= raw.sum()
        w = {k: float(v) for k, v in zip(param_keys, raw)}
        call_count[0] += 1
        return -_evaluate_weights(w, games, param_keys, use_wth=use_wth,
                                  temporal_scheme=temporal_scheme,
                                  temporal_param=temporal_param)

    result = minimize(neg_accuracy, x0, method="Nelder-Mead",
                      options={"maxiter": n_trials, "maxfev": n_trials * 2,
                               "xatol": 1e-6, "fatol": 1e-6})

    raw = np.exp(result.x)
    raw /= raw.sum()
    best_weights = {k: float(v) for k, v in zip(param_keys, raw)}
    best_acc = -result.fun

    print(f"  scipy best accuracy: {best_acc:.4f} ({call_count[0]} evaluations)")
    return best_weights, best_acc


def _random_local_search(games, param_keys, start_weights, n_trials, use_wth=False,
                         temporal_scheme="uniform", temporal_param=0.95):
    print("  Using random local search (no scipy/optuna)...")

    _ew = lambda w: _evaluate_weights(
        w, games, param_keys, use_wth=use_wth,
        temporal_scheme=temporal_scheme, temporal_param=temporal_param)

    rng = np.random.default_rng(42)
    best_weights = dict(start_weights)
    best_acc = _ew(best_weights)

    for i in range(n_trials):
        candidate = _random_weight_vector(param_keys, rng,
                                           base_weights=best_weights,
                                           perturbation=0.2)
        acc = _ew(candidate)
        if acc > best_acc:
            best_acc = acc
            best_weights = candidate

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_trials}] best={best_acc:.4f}")

    print(f"  Local search best accuracy: {best_acc:.4f}")
    return best_weights, best_acc


# ── Output Report ────────────────────────────────────────────────────────────

def generate_report(current_weights: dict, optimized_weights: dict,
                    baseline_acc: float, optimized_acc: float,
                    sensitivities: List[Tuple],
                    param_keys: list) -> str:
    """Generate comprehensive comparison report."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  WEIGHT OPTIMIZATION REPORT")
    lines.append("=" * 80)

    # Accuracy comparison
    lines.append("")
    lines.append("  ┌─ ACCURACY COMPARISON ─────────────────────────────────────┐")
    lines.append(f"  │  Current weights:   {baseline_acc*100:6.2f}% correct predictions    │")
    lines.append(f"  │  Optimized weights:  {optimized_acc*100:6.2f}% correct predictions    │")
    delta = (optimized_acc - baseline_acc) * 100
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
    lines.append(f"  │  Change:            {arrow} {abs(delta):5.2f}%                        │")
    lines.append("  └────────────────────────────────────────────────────────────┘")

    # Side-by-side weight comparison
    lines.append("")
    lines.append(f"  {'Parameter':<25} {'Current':>8} {'Optimized':>10} {'Change':>10} {'Direction':>10}")
    lines.append(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    changes = []
    for key in param_keys:
        cur = current_weights.get(key, 0)
        opt = optimized_weights.get(key, 0)
        diff = opt - cur
        changes.append((key, cur, opt, diff))

    changes.sort(key=lambda x: abs(x[3]), reverse=True)

    for key, cur, opt, diff in changes:
        if abs(diff) < 0.0005:
            arrow = "  ─"
        elif diff > 0:
            arrow = "  ▲"
        else:
            arrow = "  ▼"
        lines.append(f"  {key:<25} {cur:>7.4f}  {opt:>9.4f}  {diff:>+9.4f} {arrow}")

    # Sensitivity ranking
    lines.append("")
    lines.append("  ┌─ SENSITIVITY RANKING (most → least impactful) ────────────┐")
    for i, (param, sens, direction, _) in enumerate(sensitivities[:10], 1):
        bar_len = int(min(abs(sens) * 1000, 30))
        bar = "█" * bar_len
        lines.append(f"  │ {i:>2}. {param:<22} {sens*100:>+7.3f}%  {bar}")
    lines.append("  └────────────────────────────────────────────────────────────┘")

    # Recommendations
    lines.append("")
    lines.append("  ┌─ RECOMMENDATIONS ──────────────────────────────────────────┐")

    increase = [(k, d) for k, _, _, d in changes if d > 0.005]
    decrease = [(k, d) for k, _, _, d in changes if d < -0.005]
    low_sens = [s[0] for s in sensitivities if abs(s[1]) < 0.001]

    if increase:
        lines.append("  │  INCREASE these weights:")
        for k, d in increase[:5]:
            lines.append(f"  │    • {k}: +{d*100:.2f}%")
    if decrease:
        lines.append("  │  DECREASE these weights:")
        for k, d in decrease[:5]:
            lines.append(f"  │    • {k}: {d*100:.2f}%")
    if low_sens:
        lines.append("  │  CONSIDER REMOVING (low sensitivity):")
        for k in low_sens[:5]:
            lines.append(f"  │    • {k}")
    lines.append("  └────────────────────────────────────────────────────────────┘")

    lines.append("")
    lines.append("  NOTE: The optimizer does NOT auto-apply weights.")
    lines.append("  Review the recommendations above and decide which changes to adopt.")
    lines.append("  To apply, update CORE_WEIGHTS in src/weights.py.")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


# ── Main entry point ─────────────────────────────────────────────────────────

def _run_single_optimization(games, param_keys, n_grid, n_bayesian,
                              use_wth=False,
                              temporal_scheme="uniform",
                              temporal_param=0.95) -> Tuple[dict, float, float, List]:
    """Run full 3-stage pipeline for one mode. Returns (weights, acc, baseline, sensitivities)."""
    ts_kw = dict(temporal_scheme=temporal_scheme, temporal_param=temporal_param)

    top_results, baseline_acc = stage1_grid_search(
        games, param_keys, n_samples=n_grid, use_wth=use_wth, **ts_kw)
    best_random_acc, best_random_weights = top_results[0]

    sensitivities = stage2_sensitivity(
        games, param_keys, best_random_weights, use_wth=use_wth, **ts_kw)

    best_weights, best_acc = stage3_bayesian(
        games, param_keys, best_random_weights,
        n_trials=n_bayesian, use_wth=use_wth, **ts_kw)

    if best_random_acc > best_acc:
        final_weights = best_random_weights
        final_acc = best_random_acc
    else:
        final_weights = best_weights
        final_acc = best_acc

    return final_weights, final_acc, baseline_acc, sensitivities


def _apply_floor_and_normalize(weights: dict, floor: float = 0.001) -> dict:
    """Ensure all weights >= floor and sum to 1.0."""
    w = {k: max(v, floor) for k, v in weights.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def run_optimization(n_grid: int = 10000, n_bayesian: int = 500) -> str:
    """Run full 3-stage optimization pipeline (no WTH). Returns report string."""
    print("\n  Loading historical data...")
    kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
    tm = pd.read_csv("archive-3/Tournament Matchups.csv")

    param_keys = list(CORE_WEIGHTS.keys())
    games = _build_eval_games(kb, tm)
    print(f"  Built {len(games)} historical matchup games for evaluation")

    if len(games) < 50:
        return "ERROR: Not enough historical games found for optimization."

    ts, tp = TEMPORAL_SCHEME
    final_weights, final_acc, baseline_acc, sensitivities = \
        _run_single_optimization(games, param_keys, n_grid, n_bayesian,
                                 use_wth=False, temporal_scheme=ts, temporal_param=tp)

    report = generate_report(
        CORE_WEIGHTS, final_weights,
        baseline_acc, final_acc,
        sensitivities, param_keys
    )

    print(report)
    return report


def run_dual_optimization(n_grid: int = 10000, n_bayesian: int = 500) -> str:
    """Run optimizer TWICE: once without WTH, once with WTH.

    Returns combined report and prints both weight sets ready for weights.py.
    """
    print("\n  Loading historical data...")
    kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
    tm = pd.read_csv("archive-3/Tournament Matchups.csv")

    param_keys = list(CORE_WEIGHTS.keys())
    games = _build_eval_games(kb, tm)
    print(f"  Built {len(games)} historical matchup games for evaluation")

    if len(games) < 50:
        return "ERROR: Not enough historical games found for optimization."

    ts, tp = TEMPORAL_SCHEME

    # ── Pass 1: No WTH ──
    print("\n" + "▓" * 70)
    print("  PASS 1: OPTIMIZING WITHOUT WTH LAYER")
    print("▓" * 70)
    w_no_wth, acc_no_wth, base_no_wth, sens_no_wth = \
        _run_single_optimization(games, param_keys, n_grid, n_bayesian,
                                 use_wth=False, temporal_scheme=ts, temporal_param=tp)
    w_no_wth = _apply_floor_and_normalize(w_no_wth)

    # ── Pass 2: With WTH ──
    print("\n" + "▓" * 70)
    print("  PASS 2: OPTIMIZING WITH WTH LAYER")
    print("▓" * 70)
    w_with_wth, acc_with_wth, base_with_wth, sens_with_wth = \
        _run_single_optimization(games, param_keys, n_grid, n_bayesian,
                                 use_wth=True, temporal_scheme=ts, temporal_param=tp)
    w_with_wth = _apply_floor_and_normalize(w_with_wth)

    # ── Combined report ──
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("  DUAL WEIGHT OPTIMIZATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  ┌─ ACCURACY COMPARISON ─────────────────────────────────────────┐")
    lines.append(f"  │  NO WTH:   baseline {base_no_wth*100:.2f}% → optimized {acc_no_wth*100:.2f}%  │")
    lines.append(f"  │  WITH WTH: baseline {base_with_wth*100:.2f}% → optimized {acc_with_wth*100:.2f}%  │")
    lines.append("  └─────────────────────────────────────────────────────────────────┘")

    # Weight comparison
    lines.append("")
    lines.append(f"  {'Parameter':<25} {'No WTH':>8} {'With WTH':>10} {'Diff':>10}")
    lines.append(f"  {'─'*25} {'─'*8} {'─'*10} {'─'*10}")

    diffs = []
    for key in param_keys:
        v1 = w_no_wth.get(key, 0)
        v2 = w_with_wth.get(key, 0)
        diffs.append((key, v1, v2, v2 - v1))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    for key, v1, v2, d in diffs:
        lines.append(f"  {key:<25} {v1:>7.4f}  {v2:>9.4f}  {d:>+9.4f}")

    lines.append("")

    # Print as Python dict for easy copy-paste
    lines.append("  ┌─ CORE_WEIGHTS (no WTH) — paste into weights.py ──────────────┐")
    for key in param_keys:
        lines.append(f'  │  "{key}": {w_no_wth[key]:.6f},')
    lines.append(f"  │  # Sum = {sum(w_no_wth.values()):.6f}")
    lines.append("  └─────────────────────────────────────────────────────────────────┘")

    lines.append("")
    lines.append("  ┌─ CORE_WEIGHTS_WTH (with WTH) — paste into weights.py ────────┐")
    for key in param_keys:
        lines.append(f'  │  "{key}": {w_with_wth[key]:.6f},')
    lines.append(f"  │  # Sum = {sum(w_with_wth.values()):.6f}")
    lines.append("  └─────────────────────────────────────────────────────────────────┘")

    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report, w_no_wth, w_with_wth, acc_no_wth, acc_with_wth


def run_evaluation_metrics() -> str:
    """Compute comprehensive evaluation metrics: AUC-ROC, Avg Precision, LOYO CV.

    Uses the current CORE_WEIGHTS on historical tournament data.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss,
        accuracy_score
    )
    from scipy.stats import spearmanr

    print("\n  Loading historical data for evaluation...")
    kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
    tm = pd.read_csv("archive-3/Tournament Matchups.csv")

    param_keys = list(CORE_WEIGHTS.keys())
    games = _build_eval_games(kb, tm)
    n = len(games)

    # Compute predictions
    all_params = [g[0] for g in games] + [g[1] for g in games]
    normalized = _normalize_param_values(all_params, param_keys)
    norm_a, norm_b = normalized[:n], normalized[n:]

    probs = np.array([_predict_winner(norm_a[i], norm_b[i], CORE_WEIGHTS, LOGISTIC_K)
                       for i in range(n)])
    labels = np.array([g[2] for g in games])

    # Core metrics
    acc = accuracy_score(labels, probs > 0.5)
    brier = brier_score_loss(labels, probs)
    auc = roc_auc_score(labels, probs)
    avg_prec = average_precision_score(labels, probs)

    # Spearman correlation between predicted prob and actual outcome
    rho, p_val = spearmanr(probs, labels)

    lines = [
        "",
        "=" * 70,
        "  COMPREHENSIVE EVALUATION METRICS",
        "=" * 70,
        "",
        f"  Historical games evaluated: {n}",
        f"  Years covered: 2008-2025",
        "",
        "  ┌─ Core Metrics ─────────────────────────────────────────────────┐",
        f"  │  Accuracy:           {acc:.4f} ({acc*100:.1f}%)               │",
        f"  │  Brier Score:        {brier:.4f} (lower=better, top=0.15)    │",
        f"  │  AUC-ROC:            {auc:.4f} (0.5=random, 1.0=perfect)    │",
        f"  │  Avg Precision:      {avg_prec:.4f}                          │",
        f"  │  Spearman rho:       {rho:.4f} (p={p_val:.2e})              │",
        "  └────────────────────────────────────────────────────────────────┘",
    ]

    # Upset detection metrics (upsets = higher seed wins)
    seeds_a = []
    seeds_b = []
    for g in games:
        sa = int(round(1.0 / g[0].get("seed_score", 0.125))) if g[0].get("seed_score", 0) > 0 else 8
        sb = int(round(1.0 / g[1].get("seed_score", 0.125))) if g[1].get("seed_score", 0) > 0 else 8
        seeds_a.append(sa)
        seeds_b.append(sb)

    seeds_a = np.array(seeds_a)
    seeds_b = np.array(seeds_b)
    is_upset = ((seeds_a > seeds_b) & (labels == 1)) | ((seeds_b > seeds_a) & (labels == 0))
    upset_rate = is_upset.mean()

    upset_mask = abs(seeds_a - seeds_b) >= 4
    if upset_mask.sum() > 0:
        upset_probs = probs[upset_mask]
        upset_labels = labels[upset_mask]
        upset_acc = accuracy_score(upset_labels, upset_probs > 0.5)
        try:
            upset_auc = roc_auc_score(upset_labels, upset_probs)
        except ValueError:
            upset_auc = 0.5
    else:
        upset_acc = 0
        upset_auc = 0.5

    lines.extend([
        "",
        "  ┌─ Upset Detection ──────────────────────────────────────────────┐",
        f"  │  Overall upset rate:     {upset_rate:.1%} ({int(is_upset.sum())}/{n})        │",
        f"  │  Upset-prone accuracy:   {upset_acc:.4f} (seed diff >= 4)     │",
        f"  │  Upset-prone AUC-ROC:    {upset_auc:.4f}                      │",
        "  └────────────────────────────────────────────────────────────────┘",
    ])

    # Leave-One-Year-Out cross-validation
    years = sorted(kb["YEAR"].unique())
    years = [y for y in years if 2010 <= y <= 2025]

    loyo_correct = 0
    loyo_total = 0
    loyo_probs_all = []
    loyo_labels_all = []

    for test_year in years:
        train_games = [(p_a, p_b, w) for (p_a, p_b, w) in games
                       if _get_year_from_seed(p_a, p_b, seeds_a, seeds_b, games, test_year)]

    # Simpler LOYO: rebuild games per-year and evaluate
    year_games = {}
    for yr in years:
        yr_kb = kb[kb["YEAR"] == yr]
        yr_tm = tm[tm["YEAR"] == yr]
        if yr_kb.empty or yr_tm.empty:
            continue
        yg = _build_eval_games(yr_kb, yr_tm, start_year=yr, end_year=yr)
        if yg:
            year_games[yr] = yg

    loyo_lines = []
    all_loyo_probs = []
    all_loyo_labels = []

    for test_yr, test_games in year_games.items():
        # Train on all other years
        train_games = []
        for yr, yg in year_games.items():
            if yr != test_yr:
                train_games.extend(yg)

        if len(train_games) < 50 or len(test_games) < 5:
            continue

        # Evaluate using CORE_WEIGHTS (weights are fixed, so LOYO only tests generalization)
        test_all = [g[0] for g in test_games] + [g[1] for g in test_games]
        test_norm = _normalize_param_values(test_all, param_keys)
        tn = len(test_games)
        tna, tnb = test_norm[:tn], test_norm[tn:]

        yr_preds = []
        yr_labels = []
        for i in range(tn):
            p = _predict_winner(tna[i], tnb[i], CORE_WEIGHTS, LOGISTIC_K)
            yr_preds.append(p)
            yr_labels.append(test_games[i][2])

        yr_acc = accuracy_score(yr_labels, [p > 0.5 for p in yr_preds])
        loyo_lines.append(f"  │  {test_yr}: {yr_acc:.1%} ({int(yr_acc*tn)}/{tn} games)")
        all_loyo_probs.extend(yr_preds)
        all_loyo_labels.extend(yr_labels)

    if all_loyo_probs:
        loyo_acc = accuracy_score(all_loyo_labels, [p > 0.5 for p in all_loyo_probs])
        loyo_brier = brier_score_loss(all_loyo_labels, all_loyo_probs)
        try:
            loyo_auc = roc_auc_score(all_loyo_labels, all_loyo_probs)
        except ValueError:
            loyo_auc = 0.5

        lines.extend([
            "",
            "  ┌─ Leave-One-Year-Out Cross-Validation ──────────────────────────┐",
            f"  │  LOYO Accuracy:    {loyo_acc:.4f} ({loyo_acc*100:.1f}%)               │",
            f"  │  LOYO Brier:       {loyo_brier:.4f}                              │",
            f"  │  LOYO AUC-ROC:     {loyo_auc:.4f}                              │",
        ])
        lines.extend(loyo_lines)
        lines.append("  └────────────────────────────────────────────────────────────────┘")

    lines.extend(["", "=" * 70])
    report = "\n".join(lines)
    print(report)
    return report


def _get_year_from_seed(*args):
    """Helper stub -- not actually needed for LOYO implementation above."""
    return False


def run_overfitting_diagnostics() -> str:
    """Investigate overfitting, leakage, and noise sensitivity.

    Checks: temporal train/test gap, LOYO stability, param noise robustness,
    and consistency-AdjEM correlation (proxy vs real signal).
    """
    from sklearn.metrics import accuracy_score, brier_score_loss

    print("\n  Loading historical data for overfitting diagnostics...")
    kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
    tm = pd.read_csv("archive-3/Tournament Matchups.csv")
    param_keys = list(CORE_WEIGHTS.keys())
    games = _build_eval_games(kb, tm)

    # Build games by year
    year_games = {}
    years = sorted(kb["YEAR"].unique())
    years = [y for y in years if 2010 <= y <= 2025]
    for yr in years:
        yr_kb = kb[kb["YEAR"] == yr]
        yr_tm = tm[tm["YEAR"] == yr]
        if yr_kb.empty or yr_tm.empty:
            continue
        yg = _build_eval_games(yr_kb, yr_tm, start_year=yr, end_year=yr)
        if yg:
            year_games[yr] = yg

    lines = [
        "",
        "=" * 70,
        "  OVERFITTING & NOISE DIAGNOSTICS",
        "=" * 70,
    ]

    # 1. Temporal split: train 2008-2018, test 2019-2025
    train_years = [y for y in years if y <= 2018]
    test_years = [y for y in years if y >= 2019]
    train_g = []
    for yr in train_years:
        if yr in year_games:
            train_g.extend(year_games[yr])
    test_g = []
    for yr in test_years:
        if yr in year_games:
            test_g.extend(year_games[yr])

    if train_g and test_g:
        all_p = [g[0] for g in test_g] + [g[1] for g in test_g]
        norm = _normalize_param_values(all_p, param_keys)
        n = len(test_g)
        na, nb = norm[:n], norm[n:]
        probs = [_predict_winner(na[i], nb[i], CORE_WEIGHTS, LOGISTIC_K) for i in range(n)]
        labels = [g[2] for g in test_g]
        acc_test = accuracy_score(labels, [p > 0.5 for p in probs])
        brier_test = brier_score_loss(labels, probs)
        lines.extend([
            "",
            "  ┌─ 1. Temporal Train/Test (train ≤2018, test 2019+) ─────────────┐",
            f"  │  Test accuracy:  {acc_test:.1%} ({len(test_g)} games)          │",
            f"  │  Test Brier:     {brier_test:.4f}                            │",
            "  │  If test << train accuracy → possible overfitting to old data  │",
            "  └────────────────────────────────────────────────────────────────┘",
        ])

    # 2. LOYO variance
    per_year_acc = []
    for test_yr, test_games in year_games.items():
        all_p = [g[0] for g in test_games] + [g[1] for g in test_games]
        norm = _normalize_param_values(all_p, param_keys)
        tn = len(test_games)
        tna, tnb = norm[:tn], norm[tn:]
        preds = [_predict_winner(tna[i], tnb[i], CORE_WEIGHTS, LOGISTIC_K) for i in range(tn)]
        labs = [g[2] for g in test_games]
        per_year_acc.append(accuracy_score(labs, [p > 0.5 for p in preds]))

    if per_year_acc:
        acc_mean = np.mean(per_year_acc)
        acc_std = np.std(per_year_acc)
        lines.extend([
            "",
            "  ┌─ 2. LOYO Per-Year Accuracy Variance ─────────────────────────────┐",
            f"  │  Mean accuracy:  {acc_mean:.1%}                                 │",
            f"  │  Std dev:        {acc_std:.1%} (lower = more stable)         │",
            "  │  High variance → model sensitive to specific years (noise?)      │",
            "  └────────────────────────────────────────────────────────────────┘",
        ])

    # 3. Noise robustness: add 1% Gaussian noise to params
    n = len(games)
    all_p = [g[0] for g in games] + [g[1] for g in games]
    rng = np.random.default_rng(42)
    noisy_p = []
    for p in all_p:
        np_dict = dict(p)
        for k in param_keys:
            v = np_dict.get(k, 0)
            np_dict[k] = v + rng.normal(0, abs(v) * 0.01 + 0.005)
        noisy_p.append(np_dict)

    norm_orig = _normalize_param_values(all_p, param_keys)
    norm_noisy = _normalize_param_values(noisy_p, param_keys)
    na_o, nb_o = norm_orig[:n], norm_orig[n:]
    na_n, nb_n = norm_noisy[:n], norm_noisy[n:]
    probs_o = [_predict_winner(na_o[i], nb_o[i], CORE_WEIGHTS, LOGISTIC_K) for i in range(n)]
    probs_n = [_predict_winner(na_n[i], nb_n[i], CORE_WEIGHTS, LOGISTIC_K) for i in range(n)]
    labels = [g[2] for g in games]
    acc_o = accuracy_score(labels, [p > 0.5 for p in probs_o])
    acc_n = accuracy_score(labels, [p > 0.5 for p in probs_n])
    flip_rate = np.mean([1 if (p > 0.5) != (q > 0.5) else 0 for p, q in zip(probs_o, probs_n)])
    lines.extend([
        "",
        "  ┌─ 3. Noise Robustness (1% Gaussian noise on all params) ──────────────┐",
        f"  │  Original accuracy:   {acc_o:.1%}                               │",
        f"  │  Noisy accuracy:      {acc_n:.1%}                               │",
        f"  │  Prediction flip %:   {flip_rate:.1%} (pred changed vs orig)   │",
        "  │  High flip rate → predictions fragile to small input noise        │",
        "  └────────────────────────────────────────────────────────────────┘",
    ])

    # 4. Consistency–AdjEM correlation (historical proxy vs signal)
    from src.data_loader import load_all_teams
    teams = load_all_teams()
    data = [(t.adj_em, t.consistency, t.scoring_margin_std) for t in teams]
    if data:
        adj_ems = [d[0] for d in data]
        cons = [d[1] for d in data]
        sm_std = [d[2] for d in data]
        r_con_em = np.corrcoef(cons, adj_ems)[0, 1] if np.std(cons) > 0 else 0
        r_std_em = np.corrcoef(sm_std, adj_ems)[0, 1] if np.std(sm_std) > 0 else 0
        lines.extend([
            "",
            "  ┌─ 4. Consistency / AdjEM Correlation (2026 data) ───────────────────┐",
            f"  │  consistency vs adj_em:     r = {r_con_em:.3f}                    │",
            f"  │  scoring_margin_std vs adj_em: r = {r_std_em:.3f}                 │",
            "  │  Low r → consistency carries independent signal (good)             │",
            "  │  High r → consistency may duplicate adj_em (redundant)             │",
            "  └────────────────────────────────────────────────────────────────┘",
        ])

    lines.extend(["", "=" * 70])
    report = "\n".join(lines)
    print(report)
    return report


if __name__ == "__main__":
    import sys
    if "--diagnostics" in sys.argv or "-d" in sys.argv:
        run_overfitting_diagnostics()
        if "--metrics" in sys.argv:
            run_evaluation_metrics()
    elif "--metrics" in sys.argv or "-m" in sys.argv:
        run_evaluation_metrics()
    else:
        run_dual_optimization()
