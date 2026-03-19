"""Normalization, helpers, and team name mapping."""

import numpy as np
import pandas as pd
from typing import Dict, List
from src.weights import INVERTED_PARAMS, Z_SCORE_PARAMS


def normalize_min_max(values: np.ndarray, invert: bool = False) -> np.ndarray:
    v_min, v_max = np.nanmin(values), np.nanmax(values)
    if v_max == v_min:
        return np.full_like(values, 0.5, dtype=float)
    normed = (values - v_min) / (v_max - v_min)
    if invert:
        normed = 1.0 - normed
    return normed


def normalize_z_score(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Z-score normalize then squash to [0, 1] via sigmoid-like mapping.
    Better for narrow-range params where min-max loses discrimination.
    """
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std < 1e-9:
        return np.full_like(values, 0.5, dtype=float)
    z = (values - mean) / std
    # Squash to [0, 1]: 3 stdev range maps to ~[0.05, 0.95]
    normed = 1.0 / (1.0 + np.exp(-z))
    if invert:
        normed = 1.0 - normed
    return normed


def normalize_teams(teams: list, param_keys: list) -> None:
    """Normalize all parameters across tournament field to [0, 1].
    Uses z-score for params in Z_SCORE_PARAMS, min-max for the rest.
    """
    for key in param_keys:
        raw_values = np.array([getattr(t, key, 0.0) for t in teams], dtype=float)
        invert = key in INVERTED_PARAMS
        if key in Z_SCORE_PARAMS:
            normed = normalize_z_score(raw_values, invert=invert)
        else:
            normed = normalize_min_max(raw_values, invert=invert)
        for i, team in enumerate(teams):
            team.normalized_params[key] = float(normed[i])


TEAM_NAME_MAP = {
    "Connecticut": "Connecticut",
    "UConn": "Connecticut",
    "Miami FL": "Miami FL",
    "Miami (FL)": "Miami FL",
    "Miami (Fla.)": "Miami FL",
    "Miami OH": "Miami OH",
    "Miami (OH)": "Miami OH",
    "St. John's": "St. John's",
    "Saint John's": "St. John's",
    "St John's": "St. John's",
    "Cal Baptist": "Cal Baptist",
    "California Baptist": "Cal Baptist",
    "LIU Brooklyn": "LIU Brooklyn",
    "Long Island": "LIU Brooklyn",
    "LIU": "LIU Brooklyn",
    "Michigan St.": "Michigan State",
    "Michigan State": "Michigan State",
    "North Dakota St.": "North Dakota State",
    "North Dakota State": "North Dakota State",
    "Ohio St.": "Ohio State",
    "Ohio State": "Ohio State",
    "Iowa St.": "Iowa State",
    "Iowa State": "Iowa State",
    "Utah St.": "Utah State",
    "Utah State": "Utah State",
    "McNeese St.": "McNeese State",
    "McNeese State": "McNeese State",
    "Kennesaw St.": "Kennesaw State",
    "Kennesaw State": "Kennesaw State",
    "Wright St.": "Wright State",
    "Wright State": "Wright State",
    "Tennessee St.": "Tennessee State",
    "Tennessee State": "Tennessee State",
    "North Carolina St.": "NC State",
    "NC State": "NC State",
    "North Carolina State": "NC State",
    "Saint Mary's": "Saint Mary's",
    "St. Mary's": "Saint Mary's",
    "Northern Iowa": "Northern Iowa",
    "Hawaii": "Hawaii",
    "Hawai'i": "Hawaii",
    "Prairie View A&M": "Prairie View A&M",
    "Prairie View": "Prairie View A&M",
    "South Florida": "South Florida",
    "USF": "South Florida",
    "UConn": "Connecticut",
    "Uconn": "Connecticut",
    "UCONN": "Connecticut",
    "Michigan St": "Michigan State",
    "Michigan St.": "Michigan State",
    "Iowa St": "Iowa State",
    "Iowa St.": "Iowa State",
    "Ohio St": "Ohio State",
    "Ohio St.": "Ohio State",
    "Texas A&M": "Texas A&M",
    "Mich St": "Michigan State",
    "Mich State": "Michigan State",
    "Penn": "Penn",
    "Upenn": "Penn",
    "UPenn": "Penn",
}


def canonical_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


SEED_EXPECTED_ROUND = {
    1: 3.4, 2: 2.6, 3: 2.2, 4: 1.9,
    5: 1.5, 6: 1.4, 7: 1.3, 8: 1.1,
    9: 0.9, 10: 0.7, 11: 0.8, 12: 0.7,
    13: 0.3, 14: 0.2, 15: 0.1, 16: 0.05,
}
