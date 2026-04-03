"""Round Survival Filter — Post-filter layer that applies round-specific
stat profiles as probability modifiers.

Architecture:
  Sits AFTER ROOT + Branch Engine in the pipeline.
  Does NOT touch ROOT or branch calibration.

  Pipeline:
    ROOT → Branch Engine → Round Survival Filter → FINAL probability

  The filter asks: "Given this team's statistical DNA, how well does it
  match the profile of teams that historically WIN in this specific round?"

Three layers of intelligence:
  LAYER 1 — Seed Matchup Tiers (structural)
    Safe zone (1v16, 2v15, 3v14): 8.4% upset → talent dominates
    Danger zone (4v13): 22.1% → occasional upsets
    Coin flip zone (5v12, 6v11, 7v10, 8v9): 44.6% → anything goes

  LAYER 2 — Round-Specific Stat Profiles (backtested 2008-2025)
    R64: Quality gap + defensive efficiency
    R32: OFFENSIVE EFFICIENCY surges (EFG% +10%, PPPO +9%)
    S16: 3PT% and OREB% emerge, talent predictiveness drops
    E8:  3PT% king + tempo INVERTS + ball movement

  LAYER 3 — Upset-Zone Dynamics
    In coin-flip matchups: shooting stats ANTI-predictive
    Slower underdogs upset more (tempo control)
    3PT variance (not average) is the Cinderella factor

Scoring: Hybrid (threshold flags → severity → tanh curve)
  Same architecture as branch engine sub-conditions.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurvivalFlag:
    """One threshold check in the round survival profile."""
    name: str
    met: bool = False
    weight: float = 1.0
    value: float = 0.0
    description: str = ""


@dataclass
class RoundSurvivalResult:
    """Result of applying the round survival filter to one team in a matchup."""
    team_name: str
    round_name: str
    survival_score: float = 0.0     # 0 to 1, how well team matches the round profile
    shift: float = 0.0              # probability shift to apply
    flags_triggered: List[SurvivalFlag] = field(default_factory=list)
    flags_total: int = 0
    explanation: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Severity curve (same tanh architecture as branch engine)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_severity(flags: List[SurvivalFlag]) -> float:
    """Map triggered flags to a 0→1 severity score using the same hybrid
    compounding as branch engine: additive base, multiplicative cascade
    at 3+ flags, mapped through tanh."""
    if not flags:
        return 0.0

    triggered = [f for f in flags if f.met]
    if not triggered:
        return 0.0

    total_weight = sum(f.weight for f in flags)
    if total_weight == 0:
        return 0.0

    triggered_weight = sum(f.weight for f in triggered)
    raw = triggered_weight / total_weight

    n_triggered = len(triggered)

    # Multiplicative cascade at 3+ triggers
    if n_triggered >= 3:
        cascade = 1.0
        for f in triggered:
            cascade *= (1.0 + f.weight * 0.3)
        raw = min(raw * cascade * 0.6, 1.0)

    # Tanh mapping: smooth, bounded 0→1
    severity = math.tanh(raw * 2.0)
    return severity


# ─────────────────────────────────────────────────────────────────────────────
# Seed matchup tier detection
# ─────────────────────────────────────────────────────────────────────────────

SAFE_ZONE = {(1, 16), (2, 15), (3, 14), (16, 1), (15, 2), (14, 3)}
DANGER_ZONE = {(4, 13), (13, 4)}
# Everything else in R64 with seed gap <= 4 is coin flip zone


def _get_seed_tier(seed_a: int, seed_b: int, round_name: str) -> str:
    """Classify the matchup into a seed tier."""
    pair = (min(seed_a, seed_b), max(seed_a, seed_b))
    canonical = (pair[0], pair[1])

    if round_name == "R64":
        if canonical in {(1, 16), (2, 15), (3, 14)}:
            return "SAFE"
        elif canonical in {(4, 13)}:
            return "DANGER"
        else:
            return "COIN_FLIP"
    else:
        # In later rounds, seed tiers matter less — seed gaps are smaller
        gap = abs(seed_a - seed_b)
        if gap >= 6:
            return "SAFE"
        elif gap >= 3:
            return "DANGER"
        else:
            return "COIN_FLIP"


# ─────────────────────────────────────────────────────────────────────────────
# Round profile definitions — backtested thresholds (2008-2025)
# ─────────────────────────────────────────────────────────────────────────────

# Max shift per round (confidence-scaled by sample size)
ROUND_MAX_SHIFT = {
    "R64": 0.06,   # 318 games, moderate confidence
    "R32": 0.08,   # 158 games, strong signals
    "S16": 0.07,   # 78 games, moderate
    "E8":  0.06,   # 38 games, lower confidence but strong signals
    "F4":  0.04,   # 20 games, low sample
    "CHAMP": 0.03, # 10 games, minimal
}


def _evaluate_r64_profile(team, opponent, seed_tier: str) -> List[SurvivalFlag]:
    """R64: Quality gap dominates. In upset zone, different rules apply."""
    flags = []

    if seed_tier in ("SAFE", "DANGER"):
        # Safe/danger zone: talent and quality dominate (>72% predictive)
        flags.append(SurvivalFlag(
            name="quality_floor",
            met=team.adj_em > opponent.adj_em,
            weight=0.30,
            value=team.adj_em - opponent.adj_em,
            description=f"Adj EM edge: {team.adj_em:.1f} vs {opponent.adj_em:.1f} (73% predictive in safe zone)"
        ))
        flags.append(SurvivalFlag(
            name="defensive_efficiency",
            met=team.adj_d < opponent.adj_d,  # lower is better for defense
            weight=0.25,
            value=opponent.adj_d - team.adj_d,
            description=f"Defensive edge: {team.adj_d:.1f} vs {opponent.adj_d:.1f} (73% predictive)"
        ))
        flags.append(SurvivalFlag(
            name="strength_of_schedule",
            met=team.sos > opponent.sos,
            weight=0.20,
            value=team.sos - opponent.sos,
            description=f"SOS edge: {team.sos:.2f} vs {opponent.sos:.2f} (74% predictive)"
        ))
        flags.append(SurvivalFlag(
            name="turnover_discipline",
            met=team.to_pct < opponent.to_pct,  # lower is better
            weight=0.15,
            value=opponent.to_pct - team.to_pct,
            description=f"TOV% edge: {team.to_pct:.1f} vs {opponent.to_pct:.1f} (61% predictive)"
        ))
        flags.append(SurvivalFlag(
            name="offensive_boards",
            met=team.orb_pct > opponent.orb_pct,
            weight=0.10,
            value=team.orb_pct - opponent.orb_pct,
            description=f"OREB% edge: {team.orb_pct:.1f} vs {opponent.orb_pct:.1f} (59% predictive)"
        ))
    else:
        # COIN FLIP ZONE — different rules entirely
        # Season shooting stats are ANTI-PREDICTIVE here
        # What matters: overall quality (weakened), turnovers, tempo control

        flags.append(SurvivalFlag(
            name="quality_edge_weak",
            met=team.adj_em > opponent.adj_em,
            weight=0.25,
            value=team.adj_em - opponent.adj_em,
            description=f"Adj EM edge (weak in coin flip): {team.adj_em:.1f} vs {opponent.adj_em:.1f} (58%)"
        ))
        flags.append(SurvivalFlag(
            name="turnover_discipline",
            met=team.to_pct < opponent.to_pct,
            weight=0.20,
            value=opponent.to_pct - team.to_pct,
            description=f"TOV% edge (matters in upset zone): {team.to_pct:.1f} vs {opponent.to_pct:.1f} (55%)"
        ))
        # TEMPO CONTROL: in upset zone, SLOWER underdogs win more
        # If this team is the underdog AND slower → survival boost
        is_underdog = team.seed > opponent.seed
        is_slower = team.pace < opponent.pace
        flags.append(SurvivalFlag(
            name="tempo_control_underdog",
            met=is_underdog and is_slower,
            weight=0.25,
            value=opponent.pace - team.pace if is_underdog else 0.0,
            description=f"Slower underdog tempo control: pace {team.pace:.1f} vs {opponent.pace:.1f} (+10% upset lift)"
        ))
        # ANTI-FLAG: if underdog has better season 3PT%, that's actually NEUTRAL to negative
        # So we flag the OPPOSITE: underdog with WORSE 3PT% but strong defense
        flags.append(SurvivalFlag(
            name="grind_profile",
            met=is_underdog and team.adj_d < opponent.adj_d and team.three_p_pct <= opponent.three_p_pct,
            weight=0.20,
            value=opponent.adj_d - team.adj_d,
            description=f"Grinder underdog: better D ({team.adj_d:.1f}), not reliant on 3PT ({team.three_p_pct:.1%})"
        ))
        flags.append(SurvivalFlag(
            name="offensive_boards_upset",
            met=team.orb_pct > opponent.orb_pct,
            weight=0.10,
            value=team.orb_pct - opponent.orb_pct,
            description=f"OREB% edge in upset zone: {team.orb_pct:.1f} vs {opponent.orb_pct:.1f}"
        ))

    return flags


def _evaluate_r32_profile(team, opponent) -> List[SurvivalFlag]:
    """R32: Offensive efficiency surges. EFG% jumps +10% in predictive power.
    Faster tempo helps. Quality still matters but shooting efficiency is the new signal."""
    flags = []

    # OFFENSIVE EFFICIENCY (biggest R32 signal — EFG% 63.9%, PPPO 69.6%)
    flags.append(SurvivalFlag(
        name="offensive_efficiency",
        met=team.adj_o > opponent.adj_o,
        weight=0.25,
        value=team.adj_o - opponent.adj_o,
        description=f"Adj O edge (peaks R32 at 75%): {team.adj_o:.1f} vs {opponent.adj_o:.1f}"
    ))

    # EFG% — the stat that jumps most from R64→R32 (+10.4%)
    flags.append(SurvivalFlag(
        name="shooting_efficiency",
        met=team.efg_pct > opponent.efg_pct,
        weight=0.20,
        value=team.efg_pct - opponent.efg_pct,
        description=f"EFG% edge (surges to 64% in R32): {team.efg_pct:.1%} vs {opponent.efg_pct:.1%}"
    ))

    # Overall quality (still strong at 76.6% but captured largely by ROOT)
    flags.append(SurvivalFlag(
        name="quality_margin",
        met=team.adj_em > opponent.adj_em,
        weight=0.15,
        value=team.adj_em - opponent.adj_em,
        description=f"Adj EM edge (76.6% in R32): {team.adj_em:.1f} vs {opponent.adj_em:.1f}"
    ))

    # TEMPO — faster teams win in R32 (58.2%)
    flags.append(SurvivalFlag(
        name="tempo_advantage",
        met=team.pace > opponent.pace,
        weight=0.15,
        value=team.pace - opponent.pace,
        description=f"Tempo edge (58% R32): pace {team.pace:.1f} vs {opponent.pace:.1f}"
    ))

    # 2PT% emerges in R32 (61.4%)
    flags.append(SurvivalFlag(
        name="inside_scoring",
        met=team.two_pt_pct > opponent.two_pt_pct,
        weight=0.15,
        value=team.two_pt_pct - opponent.two_pt_pct,
        description=f"2PT% edge (61% R32): {team.two_pt_pct:.1%} vs {opponent.two_pt_pct:.1%}"
    ))

    # Defensive efficiency (still 65.2% in R32)
    flags.append(SurvivalFlag(
        name="defensive_efficiency",
        met=team.adj_d < opponent.adj_d,
        weight=0.10,
        value=opponent.adj_d - team.adj_d,
        description=f"Adj D edge (65% R32): {team.adj_d:.1f} vs {opponent.adj_d:.1f}"
    ))

    return flags


def _evaluate_s16_profile(team, opponent) -> List[SurvivalFlag]:
    """S16: Talent drops (55.1%). 3PT% and OREB% emerge. WAB still strong."""
    flags = []

    # WAB — best single predictor at S16 (65.4%)
    # We approximate WAB with q1_record + sos
    flags.append(SurvivalFlag(
        name="wins_above_bubble",
        met=team.q1_record > opponent.q1_record,
        weight=0.25,
        value=team.q1_record - opponent.q1_record,
        description=f"Q1 record edge (WAB proxy, 65% S16): {team.q1_record:.2f} vs {opponent.q1_record:.2f}"
    ))

    # 3PT% emerges at S16 (59.0%)
    flags.append(SurvivalFlag(
        name="three_pt_shooting",
        met=team.three_p_pct > opponent.three_p_pct,
        weight=0.20,
        value=team.three_p_pct - opponent.three_p_pct,
        description=f"3PT% edge (emerges S16 at 59%): {team.three_p_pct:.1%} vs {opponent.three_p_pct:.1%}"
    ))

    # OREB% holds steady (60.3% at S16)
    flags.append(SurvivalFlag(
        name="offensive_rebounding",
        met=team.orb_pct > opponent.orb_pct,
        weight=0.20,
        value=team.orb_pct - opponent.orb_pct,
        description=f"OREB% edge (60% S16): {team.orb_pct:.1f} vs {opponent.orb_pct:.1f}"
    ))

    # Offensive efficiency (BADJ O at 64.1%)
    flags.append(SurvivalFlag(
        name="offensive_efficiency",
        met=team.adj_o > opponent.adj_o,
        weight=0.20,
        value=team.adj_o - opponent.adj_o,
        description=f"Adj O edge (64% S16): {team.adj_o:.1f} vs {opponent.adj_o:.1f}"
    ))

    # Defensive efficiency (60.3%)
    flags.append(SurvivalFlag(
        name="defensive_efficiency",
        met=team.adj_d < opponent.adj_d,
        weight=0.15,
        value=opponent.adj_d - team.adj_d,
        description=f"Adj D edge (60% S16): {team.adj_d:.1f} vs {opponent.adj_d:.1f}"
    ))

    return flags


def _evaluate_e8_profile(team, opponent) -> List[SurvivalFlag]:
    """E8: 3PT% is KING (65.8%). Tempo INVERTS — slower wins (63.2%).
    Ball movement matters (AST% 60.5%). FTR crashes (29% = anti-predictive)."""
    flags = []

    # 3PT% — KING of Elite 8 (65.8% predictive)
    flags.append(SurvivalFlag(
        name="three_pt_king",
        met=team.three_p_pct > opponent.three_p_pct,
        weight=0.30,
        value=team.three_p_pct - opponent.three_p_pct,
        description=f"3PT% edge (KING at 66% E8): {team.three_p_pct:.1%} vs {opponent.three_p_pct:.1%}"
    ))

    # TEMPO INVERTS — SLOWER teams win in E8 (63.2%)
    flags.append(SurvivalFlag(
        name="tempo_control_e8",
        met=team.pace < opponent.pace,  # SLOWER is better in E8!
        weight=0.25,
        value=opponent.pace - team.pace,
        description=f"Slower tempo (63% E8!): pace {team.pace:.1f} vs {opponent.pace:.1f}"
    ))

    # Ball movement — AST% rebounds to 60.5%
    flags.append(SurvivalFlag(
        name="ball_movement",
        met=team.ast_pct > opponent.ast_pct,
        weight=0.20,
        value=team.ast_pct - opponent.ast_pct,
        description=f"AST% edge (61% E8): {team.ast_pct:.1f} vs {opponent.ast_pct:.1f}"
    ))

    # Offensive efficiency (PPPO at 65.8%)
    flags.append(SurvivalFlag(
        name="points_per_possession",
        met=team.adj_o > opponent.adj_o,
        weight=0.15,
        value=team.adj_o - opponent.adj_o,
        description=f"Adj O edge (PPPO 66% E8): {team.adj_o:.1f} vs {opponent.adj_o:.1f}"
    ))

    # FTR anti-signal — teams that foul a lot (high FTRD) lose in E8
    # We flag the INVERSE: low opponent free throw rate = good
    flags.append(SurvivalFlag(
        name="foul_discipline",
        met=team.ftr < opponent.ftr,  # lower FTR = less reliant on fouls
        weight=0.10,
        value=opponent.ftr - team.ftr,
        description=f"Foul discipline (FTR crashes to 29% E8): FTR {team.ftr:.2f} vs {opponent.ftr:.2f}"
    ))

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_round_survival(
    team_a,
    team_b,
    round_name: str,
    p_post_branch: float,
) -> Tuple[float, RoundSurvivalResult, RoundSurvivalResult]:
    """Apply the round survival filter to a matchup.

    Args:
        team_a: Team object for team A
        team_b: Team object for team B
        round_name: "R64", "R32", "S16", "E8", "F4", "CHAMP"
        p_post_branch: probability for team_a AFTER branch engine

    Returns:
        (p_final, result_a, result_b)
        p_final = adjusted probability for team_a after round survival filter
    """
    max_shift = ROUND_MAX_SHIFT.get(round_name, 0.05)

    # Get seed tier (only meaningful for R64)
    seed_tier = _get_seed_tier(team_a.seed, team_b.seed, round_name)

    # Evaluate round-specific profiles for both teams
    if round_name == "R64":
        flags_a = _evaluate_r64_profile(team_a, team_b, seed_tier)
        flags_b = _evaluate_r64_profile(team_b, team_a, seed_tier)
    elif round_name == "R32":
        flags_a = _evaluate_r32_profile(team_a, team_b)
        flags_b = _evaluate_r32_profile(team_b, team_a)
    elif round_name == "S16":
        flags_a = _evaluate_s16_profile(team_a, team_b)
        flags_b = _evaluate_s16_profile(team_b, team_a)
    elif round_name in ("E8", "F4", "CHAMP"):
        flags_a = _evaluate_e8_profile(team_a, team_b)
        flags_b = _evaluate_e8_profile(team_b, team_a)
    else:
        # Unknown round — no adjustment
        result_a = RoundSurvivalResult(team_name=team_a.name, round_name=round_name)
        result_b = RoundSurvivalResult(team_name=team_b.name, round_name=round_name)
        return p_post_branch, result_a, result_b

    # Compute severity for each team
    sev_a = _compute_severity(flags_a)
    sev_b = _compute_severity(flags_b)

    # Net survival advantage: positive = team_a has better round profile
    # The shift is proportional to the DIFFERENCE in survival scores
    net_survival = sev_a - sev_b

    # Map to shift via tanh (bounded, smooth)
    shift = math.tanh(net_survival * 1.5) * max_shift

    # Apply shift
    p_final = max(0.01, min(0.99, p_post_branch + shift))

    # Build results
    result_a = RoundSurvivalResult(
        team_name=team_a.name,
        round_name=round_name,
        survival_score=sev_a,
        shift=shift,
        flags_triggered=[f for f in flags_a if f.met],
        flags_total=len(flags_a),
        explanation=f"Round profile severity={sev_a:.2f}, net shift={shift:+.1%}"
    )
    result_b = RoundSurvivalResult(
        team_name=team_b.name,
        round_name=round_name,
        survival_score=sev_b,
        shift=-shift,
        flags_triggered=[f for f in flags_b if f.met],
        flags_total=len(flags_b),
        explanation=f"Round profile severity={sev_b:.2f}, net shift={-shift:+.1%}"
    )

    return p_final, result_a, result_b


def round_survival_report(result: RoundSurvivalResult) -> str:
    """Generate a human-readable report for one team's survival result."""
    lines = []
    lines.append(f"  {result.team_name} [{result.round_name} SURVIVAL]:")
    lines.append(f"    Score: {result.survival_score:.2f} | Shift: {result.shift:+.1%}")
    lines.append(f"    Flags: {len(result.flags_triggered)}/{result.flags_total} triggered")
    for f in result.flags_triggered:
        lines.append(f"      ✓ {f.name}: {f.description}")
    # Also show non-triggered
    return "\n".join(lines)
