"""Narrative Verification Layer — 2026-only qualitative overlay.

Reads manually-curated team-level adjustments from data/narratives.csv,
applies capped probability shifts as a wrapper around the ensemble
prob_func, and produces a conflict report comparing model predictions
vs. public/expert consensus.

NOT backtested.  The core model accuracy (73.7%) is unaffected.
This layer only nudges final probabilities within a configurable cap.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, field

from src.models import Team, Matchup
from src.utils import canonical_name
from src.weights import NARRATIVE_CAP, NARRATIVE_INJURY_OVERLAP_DISCOUNT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

VALID_FACTOR_TYPES = {
    "personnel_loss", "tactical_counter", "momentum",
    "coach_history", "public_bias", "other",
}
VALID_DIRECTIONS = {"penalty", "bonus"}
CONFIDENCE_SCALE = {"high": 1.0, "medium": 0.7, "low": 0.4}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeFactor:
    factor_type: str
    direction: str
    magnitude: float
    confidence: str
    notes: str
    effective_magnitude: float = 0.0  # magnitude * confidence_scale

    def __post_init__(self):
        scale = CONFIDENCE_SCALE.get(self.confidence, 0.7)
        sign = -1.0 if self.direction == "penalty" else 1.0
        self.effective_magnitude = sign * self.magnitude * scale


@dataclass
class TeamNarrative:
    team_name: str
    factors: List[NarrativeFactor] = field(default_factory=list)
    total_adjustment: float = 0.0  # signed, capped at [-cap, +cap]

    def compute_total(self, cap: float, injury_discount: float,
                      has_injury_profile: bool):
        """Sum effective magnitudes, discount personnel_loss if injury model active."""
        total = 0.0
        for f in self.factors:
            mag = f.effective_magnitude
            if (f.factor_type == "personnel_loss"
                    and has_injury_profile
                    and mag < 0):
                mag *= (1.0 - injury_discount)
            total += mag
        self.total_adjustment = np.clip(total, -cap, cap)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_narratives() -> Dict[str, TeamNarrative]:
    """Read data/narratives.csv and return per-team narrative adjustments."""
    path = os.path.join(DATA_DIR, "narratives.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [NARRATIVE] Failed to read narratives.csv: {e}")
        return {}

    required = {"team", "factor_type", "direction", "magnitude", "confidence"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        print(f"  [NARRATIVE] Missing columns in narratives.csv: {missing}")
        return {}

    narratives: Dict[str, TeamNarrative] = {}

    for _, row in df.iterrows():
        team = canonical_name(str(row["team"]).strip())
        ftype = str(row["factor_type"]).strip().lower()
        direction = str(row["direction"]).strip().lower()
        magnitude = float(row.get("magnitude", 0.0))
        confidence = str(row.get("confidence", "medium")).strip().lower()
        notes = str(row.get("notes", "")).strip()

        if ftype not in VALID_FACTOR_TYPES:
            print(f"  [NARRATIVE] Unknown factor_type '{ftype}' for {team} — skipping")
            continue
        if direction not in VALID_DIRECTIONS:
            print(f"  [NARRATIVE] Unknown direction '{direction}' for {team} — skipping")
            continue

        factor = NarrativeFactor(
            factor_type=ftype,
            direction=direction,
            magnitude=min(magnitude, NARRATIVE_CAP),
            confidence=confidence,
            notes=notes,
        )

        if team not in narratives:
            narratives[team] = TeamNarrative(team_name=team)
        narratives[team].factors.append(factor)

    return narratives


def finalize_narratives(narratives: Dict[str, TeamNarrative],
                        injury_profiles: Optional[Dict] = None):
    """Compute total adjustments per team, applying injury overlap discounts."""
    injury_profiles = injury_profiles or {}
    for team_name, tn in narratives.items():
        has_injury = team_name in injury_profiles
        tn.compute_total(NARRATIVE_CAP, NARRATIVE_INJURY_OVERLAP_DISCOUNT,
                         has_injury)


# ─────────────────────────────────────────────────────────────────────────────
# Probability Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def build_narrative_prob_func(
    base_func: Callable,
    narratives: Dict[str, TeamNarrative],
    injury_profiles: Optional[Dict] = None,
) -> Callable:
    """Wrap the ensemble prob_func with narrative probability shifts.

    For each matchup:
      1. Get base probability from base_func(team_a, team_b)
      2. Look up narrative adjustments for both teams
      3. Net shift = adj_b - adj_a  (team with penalty helps opponent)
      4. p_adjusted = clip(p_base + shift, 0.02, 0.98)
    """
    finalize_narratives(narratives, injury_profiles)
    _cache: Dict[Tuple[str, str], float] = {}

    def narrative_func(team_a: Team, team_b: Team) -> float:
        key = (team_a.name, team_b.name)
        if key in _cache:
            return _cache[key]

        p_base = base_func(team_a, team_b)

        adj_a = narratives[team_a.name].total_adjustment if team_a.name in narratives else 0.0
        adj_b = narratives[team_b.name].total_adjustment if team_b.name in narratives else 0.0

        # A penalty on team_a (negative adj_a) reduces team_a's probability.
        # A bonus on team_b (positive adj_b) also reduces team_a's probability.
        # Net shift from A's perspective: adj_a - adj_b
        shift = adj_a - adj_b

        p_adj = float(np.clip(p_base + shift, 0.02, 0.98))

        _cache[key] = p_adj
        _cache[(team_b.name, team_a.name)] = 1.0 - p_adj
        return p_adj

    return narrative_func


# ─────────────────────────────────────────────────────────────────────────────
# Audit Report
# ─────────────────────────────────────────────────────────────────────────────

def narrative_audit_report(
    narratives: Dict[str, TeamNarrative],
    matchups: List[Matchup],
    base_prob_func: Optional[Callable] = None,
) -> str:
    """Formatted report of narrative adjustments and conflict deltas."""
    lines = [
        "=" * 70,
        "  NARRATIVE VERIFICATION REPORT",
        "  Layer: Manual qualitative overlay (2026 only, not backtested)",
        f"  Cap: {NARRATIVE_CAP*100:.0f}% max probability shift per team",
        "=" * 70,
        "",
    ]

    # Team adjustments
    lines.append("  TEAM ADJUSTMENTS:")
    sorted_teams = sorted(narratives.values(),
                          key=lambda tn: abs(tn.total_adjustment), reverse=True)
    for tn in sorted_teams:
        pct = tn.total_adjustment * 100
        sign = "+" if pct >= 0 else ""
        lines.append(f"  {tn.team_name:<24s} {sign}{pct:.1f}%")
        for f in tn.factors:
            eff_pct = f.effective_magnitude * 100
            disc = ""
            if (f.factor_type == "personnel_loss"
                    and abs(f.effective_magnitude) < abs(f.magnitude)):
                disc = " (after injury discount)"
            lines.append(
                f"    {f.factor_type}: {f.notes[:60]}"
                f" [{eff_pct:+.1f}%]{disc}"
            )
    lines.append("")

    # Conflict deltas (matchups shifted by >3%)
    conflicts = []
    flips = []
    for m in matchups:
        adj_a = narratives[m.team_a.name].total_adjustment if m.team_a.name in narratives else 0.0
        adj_b = narratives[m.team_b.name].total_adjustment if m.team_b.name in narratives else 0.0
        shift = adj_a - adj_b

        if abs(shift) < 0.001:
            continue

        p_base = m.win_prob_a_ensemble
        p_adj = float(np.clip(p_base + shift, 0.02, 0.98))
        delta = p_adj - p_base

        if abs(delta) > 0.03:
            conflicts.append((m, p_base, p_adj, delta))

        base_winner_a = p_base > 0.5
        adj_winner_a = p_adj > 0.5
        if base_winner_a != adj_winner_a:
            flips.append((m, p_base, p_adj, delta))

    lines.append("  CONFLICT DELTAS (narrative shifts >3%):")
    if conflicts:
        for m, p_base, p_adj, delta in sorted(conflicts, key=lambda x: -abs(x[3])):
            fav_base = m.team_a.name if p_base > 0.5 else m.team_b.name
            prob_base = max(p_base, 1 - p_base)
            fav_adj = m.team_a.name if p_adj > 0.5 else m.team_b.name
            prob_adj = max(p_adj, 1 - p_adj)
            lines.append(
                f"  ({m.team_a.seed}) {m.team_a.name} vs "
                f"({m.team_b.seed}) {m.team_b.name}"
            )
            lines.append(
                f"    Model: {fav_base} {prob_base:.1%}  |  "
                f"Adjusted: {fav_adj} {prob_adj:.1%}  |  "
                f"Delta: {delta:+.1%}"
            )
    else:
        lines.append("  None")
    lines.append("")

    lines.append("  WINNER FLIPS (narrative changes predicted winner):")
    if flips:
        for m, p_base, p_adj, delta in flips:
            old_winner = m.team_a.name if p_base > 0.5 else m.team_b.name
            new_winner = m.team_a.name if p_adj > 0.5 else m.team_b.name
            lines.append(
                f"  ({m.team_a.seed}) {m.team_a.name} vs "
                f"({m.team_b.seed}) {m.team_b.name}"
            )
            lines.append(f"    {old_winner} -> {new_winner}  (delta: {delta:+.1%})")
    else:
        lines.append("  None")

    lines.append("")
    lines.append("  NOTE: These adjustments are qualitative and not validated against")
    lines.append("  historical data. The core model accuracy (73.7%) is unaffected.")
    lines.append("=" * 70)

    return "\n".join(lines)
