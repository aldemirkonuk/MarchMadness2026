"""Ensemble blender: combines Phase 1A and Phase 1B predictions.

Produces final pre-tournament predictions with confidence scoring
and disagreement flags.
"""

import numpy as np
from typing import List, Dict, Optional

from src.models import Team, Matchup
from src.equations import ensemble_probability, confidence_score
from src.weights import ENSEMBLE_LAMBDA


def _sigmoid(x: float, scale: float = 10.0) -> float:
    """Logistic sigmoid mapping (-inf,+inf) -> (0,1)."""
    return 1.0 / (1.0 + np.exp(-scale * x))


def blend_predictions(matchup: Matchup, lambda_: float = None) -> float:
    """Blend 1A and 1B predictions for a single matchup.

    Uses smooth confidence-weighted blending instead of hard-coded gap
    thresholds.  When 1A is more confident (further from 0.5) than 1B,
    lambda increases toward 1A; when 1B is more confident, lambda
    decreases toward 1B.
    """
    base_lam = lambda_ if lambda_ is not None else ENSEMBLE_LAMBDA

    p1a = matchup.win_prob_a_1a
    p1b = matchup.win_prob_a_1b

    conf_1a = abs(p1a - 0.5)
    conf_1b = abs(p1b - 0.5)

    # Sigmoid blending: confidence difference drives how far from base_lam
    # Positive conf_diff → 1A more confident → shift toward 1A
    conf_diff = conf_1a - conf_1b
    shift = 0.20 * (2.0 * _sigmoid(conf_diff, scale=8.0) - 1.0)
    lam = np.clip(base_lam + shift, 0.20, 0.85)

    p_ens = ensemble_probability(p1a, p1b, lam)
    matchup.win_prob_a_ensemble = p_ens
    matchup.confidence = confidence_score(p1a, p1b)
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


def calibrate_lambda() -> float:
    """Sweep ensemble lambda on historical data to minimise Brier score.

    Replays all 2008-2025 tournament games through both the 1A weighted
    composite and a LOYO XGBoost, then picks the lambda that yields the
    lowest Brier score on the ensemble.

    Returns the best lambda found (or ENSEMBLE_LAMBDA if insufficient data).
    """
    try:
        import pandas as pd
        from sklearn.metrics import brier_score_loss
        import xgboost as xgb
    except ImportError:
        return ENSEMBLE_LAMBDA

    from src.weight_optimizer import (
        _build_eval_games, _evaluate_weights, _normalize_param_values,
        _predict_winner
    )
    from src.weights import CORE_WEIGHTS, LOGISTIC_K, PARAM_KEYS
    from src.xgboost_model import (
        prepare_historical_features, ML_FEATURE_KEYS
    )
    from src.equations import build_ml_features

    kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
    tm = pd.read_csv("archive-3/Tournament Matchups.csv")

    games = _build_eval_games(kb, tm)
    n = len(games)
    if n < 100:
        return ENSEMBLE_LAMBDA

    # 1A predictions (weighted composite)
    all_params = [g[0] for g in games] + [g[1] for g in games]
    norm = _normalize_param_values(all_params, list(CORE_WEIGHTS.keys()))
    na, nb = norm[:n], norm[n:]
    p1a_all = [_predict_winner(na[i], nb[i], CORE_WEIGHTS, LOGISTIC_K) for i in range(n)]

    # 1B predictions (XGBoost, LOYO to avoid leakage)
    from collections import defaultdict
    year_indices = defaultdict(list)
    for i, g in enumerate(games):
        year_indices[g[3]].append(i)

    X_full, y_full = prepare_historical_features(kb, tm)
    p1b_all = [0.5] * n

    # XGBoost features are symmetrised (2x matchups), so we use the
    # _row_to_stats approach per game from our weight_optimizer games.
    from src.xgboost_model import _row_to_stats
    team_stats_cache = {}
    for _, row in kb.iterrows():
        yr = row.get("YEAR", 0)
        team = str(row.get("TEAM", "")).strip()
        seed = row.get("SEED", 0)
        if pd.isna(seed) or seed == 0 or pd.isna(yr):
            continue
        team_stats_cache[(int(yr), team)] = _row_to_stats(row)

    # LOYO XGBoost
    years = sorted(year_indices.keys())
    for test_yr in years:
        test_idx = year_indices[test_yr]
        train_idx = [i for yr, idxs in year_indices.items()
                     if yr != test_yr for i in idxs]

        if len(train_idx) < 50:
            continue

        # Build train features from weight_optimizer games
        X_train_list, y_train_list = [], []
        for idx in train_idx:
            g = games[idx]
            pa, pb = g[0], g[1]
            feats = build_ml_features(pa, pb, ML_FEATURE_KEYS)
            X_train_list.append(feats)
            y_train_list.append(g[2])
            # Symmetrise
            feats_rev = []
            for j in range(0, len(feats), 3):
                feats_rev.extend([feats[j+1], feats[j], -feats[j+2]])
            X_train_list.append(feats_rev)
            y_train_list.append(1 - g[2])

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            eval_metric="logloss", random_state=42)
        clf.fit(X_train, y_train, verbose=False)

        for idx in test_idx:
            g = games[idx]
            feats = build_ml_features(g[0], g[1], ML_FEATURE_KEYS)
            X_test = np.array([feats])
            p1b_all[idx] = float(clf.predict_proba(X_test)[0][1])

    labels = [g[2] for g in games]

    # Sweep lambda
    print("\n  ┌─ ENSEMBLE LAMBDA CALIBRATION ────────────────────────────────┐")
    print(f"  │  Games: {n}  |  Lambda sweep: 0.30 → 0.80                  │")
    print(f"  │  {'Lambda':>8}  {'Accuracy':>10}  {'Brier':>10}              │")
    print(f"  │  {'─'*8}  {'─'*10}  {'─'*10}              │")

    best_lam = ENSEMBLE_LAMBDA
    best_brier = 1.0

    for lam_int in range(30, 82, 5):
        lam = lam_int / 100.0
        p_ens = [lam * p1a_all[i] + (1 - lam) * p1b_all[i] for i in range(n)]
        acc = sum(1 for p, l in zip(p_ens, labels) if (p > 0.5) == l) / n
        brier = brier_score_loss(labels, p_ens)
        marker = "  <-- best" if brier < best_brier else ""
        print(f"  │  {lam:>8.2f}  {acc*100:>9.1f}%  {brier:>10.4f}{marker:>14}│")
        if brier < best_brier:
            best_brier = brier
            best_lam = lam

    print(f"  │                                                            │")
    print(f"  │  Best: lambda={best_lam:.2f}  Brier={best_brier:.4f}               │")
    print(f"  └────────────────────────────────────────────────────────────┘")

    return best_lam
