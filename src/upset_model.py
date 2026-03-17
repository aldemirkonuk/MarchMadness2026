"""Upset detection model for flagging high-risk tournament matchups.

Uses a specialized logistic regression trained on historical 4-9 seed-diff
games to identify when the main model may be overconfident in the favorite.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression

from src.models import Team, Matchup


UPSET_FEATURE_KEYS = [
    "adj_em", "barthag", "sos", "consistency", "ctf", "legacy_factor",
    "scoring_margin_std", "shooting_eff", "exp", "to_pct", "seed_score",
    "momentum", "fragility_score",
]


def _extract_team_params(team: Team) -> dict:
    """Extract params used by the upset model from a Team object."""
    return {
        "adj_em": team.adj_em,
        "barthag": team.barthag,
        "sos": team.sos,
        "consistency": team.consistency,
        "ctf": team.ctf,
        "legacy_factor": team.legacy_factor,
        "scoring_margin_std": team.scoring_margin_std,
        "shooting_eff": team.shooting_eff,
        "exp": team.exp,
        "to_pct": team.to_pct,
        "seed_score": team.seed_score,
        "momentum": team.momentum,
        "fragility_score": team.fragility_score,
    }


def train_upset_model() -> Optional[LogisticRegression]:
    """Train the upset detection model on historical data."""
    try:
        from src.weight_optimizer import _build_eval_games
        kb = pd.read_csv("archive-3/KenPom Barttorvik.csv")
        tm = pd.read_csv("archive-3/Tournament Matchups.csv")
        games = _build_eval_games(kb, tm)
    except Exception:
        return None

    X, y = [], []
    for pa, pb, won in games:
        sa = int(round(1.0 / pa.get("seed_score", 0.125))) if pa.get("seed_score", 0) > 0 else 8
        sb = int(round(1.0 / pb.get("seed_score", 0.125))) if pb.get("seed_score", 0) > 0 else 8
        sd = abs(sa - sb)
        if sd < 4 or sd > 9:
            continue

        if sa < sb:
            fav, dog, fav_won = pa, pb, won
        else:
            fav, dog, fav_won = pb, pa, 1 - won

        feats = [fav.get(k, 0) - dog.get(k, 0) for k in UPSET_FEATURE_KEYS]
        feats.append(sd)
        X.append(feats)
        y.append(fav_won)

    if len(X) < 50:
        return None

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(np.array(X), np.array(y))
    return model


def flag_upset_risks(matchups: List[Matchup],
                     model: Optional[LogisticRegression] = None
                     ) -> List[Tuple[Matchup, float, str]]:
    """Flag matchups where the upset model sees elevated risk.

    Returns list of (matchup, upset_prob, reason) tuples.
    """
    if model is None:
        model = train_upset_model()
    if model is None:
        return []

    flags = []
    for m in matchups:
        sd = abs(m.team_a.seed - m.team_b.seed)
        if sd < 3:
            continue

        if m.team_a.seed < m.team_b.seed:
            fav, dog = m.team_a, m.team_b
        else:
            fav, dog = m.team_b, m.team_a

        fav_p = _extract_team_params(fav)
        dog_p = _extract_team_params(dog)

        feats = [fav_p.get(k, 0) - dog_p.get(k, 0) for k in UPSET_FEATURE_KEYS]
        feats.append(sd)
        X = np.array([feats])

        fav_win_prob = model.predict_proba(X)[0][1]
        upset_prob = 1.0 - fav_win_prob

        reasons = []
        if fav.scoring_margin_std > dog.scoring_margin_std:
            reasons.append(f"Fav volatile (std={fav.scoring_margin_std:.1f})")
        if dog.ctf > fav.ctf:
            reasons.append(f"Underdog coach edge (CTF {dog.ctf:.3f}>{fav.ctf:.3f})")
        if dog.legacy_factor > fav.legacy_factor:
            reasons.append(f"Underdog legacy ({dog.name} PASE={dog.legacy_factor:.1f})")
        if dog.consistency > fav.consistency:
            reasons.append(f"Underdog more consistent")

        if upset_prob > 0.25:
            flags.append((m, upset_prob, "; ".join(reasons) if reasons else "Model signal"))

    return sorted(flags, key=lambda x: -x[1])


def print_upset_flags(matchups: List[Matchup]) -> str:
    """Generate a formatted report of upset-risk matchups."""
    flags = flag_upset_risks(matchups)

    lines = [
        "=" * 70,
        "  UPSET DETECTION MODEL FLAGS",
        "=" * 70,
        "",
    ]

    if not flags:
        lines.append("  No elevated upset risks detected.")
    else:
        for m, prob, reason in flags:
            fav = m.team_a if m.team_a.seed < m.team_b.seed else m.team_b
            dog = m.team_b if m.team_a.seed < m.team_b.seed else m.team_a
            risk = "HIGH" if prob > 0.40 else "MEDIUM" if prob > 0.30 else "ELEVATED"
            lines.append(f"  [{risk}] ({dog.seed}) {dog.name} over ({fav.seed}) {fav.name}")
            lines.append(f"         Upset prob: {prob:.1%}  |  {reason}")
            lines.append("")

    return "\n".join(lines)
