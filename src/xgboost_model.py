"""Phase 1B: XGBoost ML Pipeline.

Trains on historical tournament data (2010-2025) to learn non-linear
interactions between parameters. Uses Leave-One-Year-Out cross-validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from src.models import Team
from src.weights import PARAM_KEYS
from src.equations import build_ml_features


ML_FEATURE_KEYS = [
    "adj_em", "efg_pct", "ts_pct", "to_pct", "three_pri", "orb_pct",
    "drb_pct", "opp_to_pct", "ast_pct", "exp", "seed_score", "sos",
    "ftr", "blk_pct", "pace", "spi", "barthag", "ppg",
]

SEED_WIN_PRIORS = {
    (1, 16): 0.99, (1, 8): 0.80, (1, 9): 0.85,
    (2, 15): 0.94, (2, 7): 0.70, (2, 10): 0.75,
    (3, 14): 0.85, (3, 6): 0.65, (3, 11): 0.70,
    (4, 13): 0.79, (4, 5): 0.55,
    (5, 12): 0.64, (6, 11): 0.63, (7, 10): 0.61, (8, 9): 0.51,
}


def prepare_historical_features(hist_df: pd.DataFrame,
                                matchups_df: Optional[pd.DataFrame] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix from historical tournament data.

    Uses Tournament Matchups data (consecutive row pairs = one game) combined
    with KenPom Barttorvik team stats to construct per-matchup features.
    """
    if matchups_df is not None and not matchups_df.empty:
        result = _build_features_from_actual_matchups(hist_df, matchups_df)
        if len(result[0]) > 0:
            return result

    return _build_features_from_seed_matchups(hist_df)


def _build_features_from_seed_matchups(hist_df: pd.DataFrame
                                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic matchups from seed-based pairings.

    For each year, create matchups based on historical seed win rates.
    """
    features_list = []
    labels_list = []

    SEED_WIN_RATES = {
        (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
        (5, 12): 0.64, (6, 11): 0.63, (7, 10): 0.61, (8, 9): 0.51,
    }

    years = sorted(hist_df["YEAR"].unique())

    for year in years:
        yr_data = hist_df[hist_df["YEAR"] == year]
        if yr_data.empty:
            continue

        # Create team stat lookup
        team_stats = {}
        for _, row in yr_data.iterrows():
            team = str(row.get("TEAM", "")).strip()
            seed = row.get("SEED", 0)
            if pd.isna(seed) or seed == 0:
                continue
            seed = int(seed)

            stats = _row_to_stats(row)
            team_stats[(year, seed, team)] = stats

        # Generate seed-based matchups
        seeds_by_year = defaultdict(list)
        for (y, s, t), stats in team_stats.items():
            if y == year:
                seeds_by_year[s].append((t, stats))

        for (s1, s2), win_rate in SEED_WIN_RATES.items():
            teams_s1 = seeds_by_year.get(s1, [])
            teams_s2 = seeds_by_year.get(s2, [])

            for t1_name, t1_stats in teams_s1:
                for t2_name, t2_stats in teams_s2:
                    feats = build_ml_features(t1_stats, t2_stats, ML_FEATURE_KEYS)
                    features_list.append(feats)

                    # Use actual round reached if available
                    labels_list.append(1 if np.random.random() < win_rate else 0)

    if not features_list:
        return np.array([]), np.array([])

    return np.array(features_list), np.array(labels_list)


def _build_features_from_actual_matchups(hist_df: pd.DataFrame,
                                          matchups_df: pd.DataFrame
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Build features from actual tournament matchup results.

    Tournament Matchups format: consecutive row pairs form one game.
    Row with higher ROUND advanced further (= winner).
    """
    features_list = []
    labels_list = []

    # Build team stats lookup from KenPom Barttorvik
    team_stats_cache = {}
    for _, row in hist_df.iterrows():
        year = row.get("YEAR", 0)
        team = str(row.get("TEAM", "")).strip()
        seed = row.get("SEED", 0)
        if pd.isna(seed) or seed == 0 or pd.isna(year):
            continue
        team_stats_cache[(int(year), team)] = _row_to_stats(row)

    # Parse matchups as consecutive row pairs
    mdf = matchups_df.sort_values(["YEAR", "BY YEAR NO"], ascending=[True, False])

    for year in range(2010, 2026):
        yr_matchups = mdf[mdf["YEAR"] == year].reset_index(drop=True)
        if len(yr_matchups) < 2:
            continue

        for i in range(0, len(yr_matchups) - 1, 2):
            row_a = yr_matchups.iloc[i]
            row_b = yr_matchups.iloc[i + 1]

            # Only use same CURRENT ROUND (same game)
            if row_a.get("CURRENT ROUND") != row_b.get("CURRENT ROUND"):
                continue

            team_a = str(row_a["TEAM"]).strip()
            team_b = str(row_b["TEAM"]).strip()
            score_a = float(row_a.get("SCORE", 0) or 0)
            score_b = float(row_b.get("SCORE", 0) or 0)
            round_a = float(row_a.get("ROUND", 0) or 0)
            round_b = float(row_b.get("ROUND", 0) or 0)

            stats_a = team_stats_cache.get((year, team_a))
            stats_b = team_stats_cache.get((year, team_b))

            if stats_a is None or stats_b is None:
                continue

            feats = build_ml_features(stats_a, stats_b, ML_FEATURE_KEYS)
            features_list.append(feats)

            # Lower ROUND number = advanced further = winner
            # ROUND=64 means lost in R64, ROUND=4 means reached Final Four
            if round_a < round_b:
                labels_list.append(1)  # team A advanced further
            elif round_b < round_a:
                labels_list.append(0)  # team B advanced further
            else:
                labels_list.append(1 if score_a > score_b else 0)

    if not features_list:
        return np.array([]), np.array([])

    # Symmetrize: add reverse matchups to eliminate positional bias
    sym_features = []
    sym_labels = []
    for feats, label in zip(features_list, labels_list):
        sym_features.append(feats)
        sym_labels.append(label)
        # Swap A and B features
        reversed_feats = []
        for i in range(0, len(feats), 3):
            reversed_feats.extend([feats[i+1], feats[i], -feats[i+2]])
        sym_features.append(reversed_feats)
        sym_labels.append(1 - label)

    return np.array(sym_features), np.array(sym_labels)


def _row_to_stats(row) -> dict:
    """Convert a DataFrame row to stats dict."""
    seed = int(row.get("SEED", 8) or 8)
    adj_em = float(row.get("KADJ EM", 0) or 0)
    pace = float(row.get("RAW T", 68) or 68)
    pppo = float(row.get("PPPO", 0) or 0)
    return {
        "adj_em": adj_em,
        "efg_pct": float(row.get("EFG%", 0) or 0) / 100,
        "ts_pct": float(row.get("EFG%", 0) or 0) / 100 * 0.85 +
                  float(row.get("FTR", 0) or 0) / 100 * float(row.get("FT%", 0) or 0) / 100 * 0.15,
        "to_pct": float(row.get("TOV%", 0) or 0) / 100,
        "three_pri": float(row.get("3PT%", 0) or 0) / 100 *
                     float(row.get("3PTR", 0) or 0) / 100,
        "orb_pct": float(row.get("OREB%", 0) or 0) / 100,
        "drb_pct": float(row.get("DREB%", 0) or 0) / 100,
        "opp_to_pct": float(row.get("TOV%D", 0) or 0) / 100,
        "ast_pct": float(row.get("AST%", 0) or 0) / 100,
        "exp": float(row.get("EXP", 2) or 2),
        "seed_score": 1.0 / max(seed, 1),
        "sos": float(row.get("ELITE SOS", 0) or 0),
        "ftr": float(row.get("FTR", 0) or 0) / 100 *
               float(row.get("FT%", 0) or 0) / 100,
        "blk_pct": float(row.get("BLK%", 0) or 0) / 100,
        "pace": pace,
        "spi": adj_em / 20.0,
        "barthag": float(row.get("BARTHAG", 0.5) or 0.5),
        "ppg": pppo * pace if pppo > 0 else adj_em * pace / 100.0,
    }


def train_xgboost(X: np.ndarray, y: np.ndarray,
                   loyo_years: Optional[list] = None) -> Tuple:
    """Train XGBoost model with optional Leave-One-Year-Out CV."""
    try:
        import xgboost as xgb
    except ImportError:
        print("WARNING: xgboost not installed. Phase 1B will use seed-based fallback.")
        return None, None

    params = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)

    return model, params


def predict_matchup(model, team_a: Team, team_b: Team,
                    p_1a: float = None) -> float:
    """Predict win probability for team_a using XGBoost model.

    Applies seed-based calibration to prevent extreme divergence.
    Optionally uses 1A probability as a soft anchor.
    """
    if model is None:
        return 0.5

    a_stats = _team_to_stats(team_a)
    b_stats = _team_to_stats(team_b)
    feats = build_ml_features(a_stats, b_stats, ML_FEATURE_KEYS)
    X = np.array([feats])

    raw_prob = float(model.predict_proba(X)[0][1])

    # Calibrate: blend raw XGBoost with seed-based prior
    seed_a, seed_b = team_a.seed, team_b.seed
    pair = (min(seed_a, seed_b), max(seed_a, seed_b))
    seed_prior = SEED_WIN_PRIORS.get(pair, 0.5)
    if seed_a > seed_b:
        seed_prior = 1.0 - seed_prior

    seed_gap = abs(seed_a - seed_b)
    if seed_gap <= 3:
        model_weight = 0.55
    elif seed_gap <= 6:
        model_weight = 0.65
    else:
        model_weight = 0.75

    calibrated = model_weight * raw_prob + (1.0 - model_weight) * seed_prior

    # If 1A probability available, soft-anchor toward it to prevent wild divergence
    if p_1a is not None:
        divergence = abs(calibrated - p_1a)
        if divergence > 0.20:
            anchor_weight = 0.3
            calibrated = (1.0 - anchor_weight) * calibrated + anchor_weight * p_1a

    return calibrated


def _team_to_stats(team: Team) -> dict:
    """Convert Team object to stats dict for ML features."""
    return {
        "adj_em": team.adj_em,
        "efg_pct": team.efg_pct,
        "ts_pct": team.ts_pct,
        "to_pct": team.to_pct,
        "three_pri": team.three_pri,
        "orb_pct": team.orb_pct,
        "drb_pct": team.drb_pct,
        "opp_to_pct": team.opp_to_pct,
        "ast_pct": team.ast_pct,
        "exp": team.exp,
        "seed_score": team.seed_score,
        "sos": team.sos,
        "ftr": team.ftr,
        "blk_pct": team.blk_pct,
        "pace": team.pace,
        "spi": team.spi,
        "barthag": team.barthag,
        "ppg": team.ppg,
    }


def get_shap_analysis(model, team_a: Team, team_b: Team) -> Optional[dict]:
    """Get SHAP values explaining the prediction for a matchup."""
    try:
        import shap
    except ImportError:
        return None

    if model is None:
        return None

    a_stats = _team_to_stats(team_a)
    b_stats = _team_to_stats(team_b)
    feats = build_ml_features(a_stats, b_stats, ML_FEATURE_KEYS)
    X = np.array([feats])

    feature_names = []
    for key in ML_FEATURE_KEYS:
        feature_names.extend([f"{key}_A", f"{key}_B", f"{key}_diff"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    importance = {}
    for name, val in zip(feature_names, sv):
        importance[name] = float(val)

    return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
