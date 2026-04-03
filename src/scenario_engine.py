"""Scenario Engine: Evidence-based reasoning layer for matchup analysis.

Replaces the flat 15-branch engine with a 4-category scenario model:
  1. ROSTER STATE: injuries, depth, star isolation, fatigue, rotation
  2. MATCHUP STYLE: rebounding, physicality, pace, 3PT profile, paint scoring
  3. FORM & TRAJECTORY: season collapse, tournament surge, per-stat trajectory
  4. INTANGIBLES: coach DNA, experience, pressure, revenge, lockdown defender

Architecture:
  Layer 1: Gather evidence per category (sub-signals compound within)
  Layer 2: Within-category compounding (related signals amplify each other)
  Layer 3: Cross-category interaction (reinforcing evidence amplifies)
  Layer 4: Coherence scoring (all-aligned = high confidence = bigger shift)

Design principles:
  - Additive shift, not multiplicative survival*boost (transparent, auditable)
  - Category ceilings prevent runaway (ROSTER max 0.20, STYLE 0.15, FORM 0.12, INTANGIBLES 0.10)
  - Coherence bonus when all evidence aligns (up to +15%)
  - Base-probability dampening: shifts compressed near p=0 or p=1 (4*p*(1-p), floor 0.35)
  - Trajectory decay: first-weekend data fades in later rounds (S16 0.70, F4 0.30)
  - Asymmetric data guard: one-sided trajectory capped at 0.02 (requires both teams' data for full signal)
  - Tournament-adaptive: consumes live box-score data and detects trajectories
  - Every signal is traceable: the report shows exactly WHY
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    """One piece of evidence within a category."""
    name: str
    met: bool = False
    weight: float = 0.0
    raw_value: float = 0.0
    description: str = ""


@dataclass
class CategoryResult:
    """Output of one evidence category evaluation."""
    category: str
    signals: List[Signal] = field(default_factory=list)
    raw_shift: float = 0.0
    compound_multiplier: float = 1.0
    final_shift: float = 0.0
    direction: str = ""       # "team_a", "team_b", or "neutral"
    confidence: float = 0.0   # 0.0-1.0 how strong the evidence is
    explanation: str = ""
    max_shift: float = 0.0


@dataclass
class ScenarioResult:
    """Complete scenario analysis for one matchup."""
    team_a_name: str = ""
    team_b_name: str = ""
    p_base: float = 0.5
    p_final: float = 0.5
    total_shift: float = 0.0
    categories: List[CategoryResult] = field(default_factory=list)
    coherence: float = 0.0
    coherence_bonus: float = 0.0
    uncertainty: float = 0.0
    confidence_post: float = 0.0
    narrative: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Category ceilings and configuration
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_CONFIG = {
    "ROSTER_STATE": {"max_shift": 0.20, "compound_threshold": 2},
    "MATCHUP_STYLE": {"max_shift": 0.15, "compound_threshold": 2},
    "FORM_TRAJECTORY": {"max_shift": 0.12, "compound_threshold": 2},
    "INTANGIBLES": {"max_shift": 0.10, "compound_threshold": 2},
}

MAX_SCENARIO_SHIFT = 0.35
COHERENCE_AMPLIFY_THRESHOLD = 0.5

PHYSICAL_CONFERENCES = {
    "Big 12": 4, "SEC": 3, "Big East": 3, "Big Ten": 2,
    "ACC": 2, "Mountain West": 2, "AAC": 1, "A-10": 1,
}
FINESSE_CONFERENCES = {
    "WCC": True, "Big Sky": True, "Horizon": True, "MAAC": True,
    "Patriot": True, "Ivy": True, "MEAC": True, "SWAC": True,
    "Southland": True, "NEC": True, "Summit": True,
}

ROUND_FATIGUE = {"R64": 1.0, "R32": 1.1, "S16": 1.2, "E8": 1.3, "F4": 1.4, "Championship": 1.5}

TRAJECTORY_DECAY = {
    "R64": 1.0, "R32": 1.0, "S16": 0.70,
    "E8": 0.45, "F4": 0.30, "Championship": 0.20,
}

RECENCY_TOURNEY_WEIGHT = {
    "R64": 0.05, "R32": 0.15, "S16": 0.25,
    "E8": 0.40, "F4": 0.50, "Championship": 0.55,
}


# ─────────────────────────────────────────────────────────────────────────────
# Scenario Context -- everything the engine needs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioContext:
    """All data for scenario evaluation. Callers populate what they have."""
    team_a_name: str = ""
    team_b_name: str = ""
    p_base: float = 0.5
    round_name: str = "R64"
    seed_a: int = 1
    seed_b: int = 16

    # --- ROSTER STATE ---
    injury_penalty_a: float = 0.0
    injury_penalty_b: float = 0.0
    star_lost_a: bool = False
    star_lost_b: bool = False
    star_bpr_share_a: float = 0.0
    star_bpr_share_b: float = 0.0
    sii_a: float = 0.0
    sii_b: float = 0.0
    crippled_roster_a: bool = False
    crippled_roster_b: bool = False
    crippled_weeks_out_a: float = 0.0
    crippled_weeks_out_b: float = 0.0
    bench_depth_a: float = 0.5
    bench_depth_b: float = 0.5
    rotation_size_a: int = 9
    rotation_size_b: int = 9
    foul_trouble_rate_a: float = 0.0
    foul_trouble_rate_b: float = 0.0
    top_player_minutes_a: float = 0.0
    top_player_minutes_b: float = 0.0

    # --- MATCHUP STYLE ---
    orb_pct_a: float = 0.0
    orb_pct_b: float = 0.0
    drb_pct_a: float = 0.0
    drb_pct_b: float = 0.0
    rbm_a: float = 0.0
    rbm_b: float = 0.0
    conference_a: str = ""
    conference_b: str = ""
    three_pt_share_a: float = 0.0
    three_pt_share_b: float = 0.0
    three_pt_pct_a: float = 0.0
    three_pt_pct_b: float = 0.0
    three_pt_std_a: float = 0.0
    three_pt_std_b: float = 0.0
    pace_a: float = 68.0
    pace_b: float = 68.0
    # Tournament box-score derived
    tourney_orb_pct_a: float = 0.0
    tourney_orb_pct_b: float = 0.0
    tourney_paint_pct_a: float = 0.0
    tourney_paint_pct_b: float = 0.0
    tourney_ast_rate_a: float = 0.0
    tourney_ast_rate_b: float = 0.0
    tourney_games_a: int = 0
    tourney_games_b: int = 0
    tourney_data_confidence_a: float = 0.0
    tourney_data_confidence_b: float = 0.0

    # --- FORM & TRAJECTORY ---
    form_trend_a: float = 0.5
    form_trend_b: float = 0.5
    momentum_a: float = 0.5
    momentum_b: float = 0.5
    losses_last_6_a: int = 0
    losses_last_6_b: int = 0
    offensive_burst_a: float = 0.0
    offensive_burst_b: float = 0.0
    q3_adj_a: float = 0.0
    q3_adj_b: float = 0.0
    # Tournament momentum (from tournament_momentum.py)
    tourney_momentum_a: float = 0.0
    tourney_momentum_b: float = 0.0

    # Trajectory slopes from tournament games (per-game change)
    trajectory_fg_pct_a: float = 0.0
    trajectory_fg_pct_b: float = 0.0
    trajectory_ast_a: float = 0.0
    trajectory_ast_b: float = 0.0
    trajectory_tov_a: float = 0.0
    trajectory_tov_b: float = 0.0
    trajectory_paint_a: float = 0.0
    trajectory_paint_b: float = 0.0
    trajectory_bench_a: float = 0.0
    trajectory_bench_b: float = 0.0
    tourney_efg_a: float = 0.0
    tourney_efg_b: float = 0.0
    tourney_tov_pct_a: float = 0.0
    tourney_tov_pct_b: float = 0.0
    conf_tourney_early_exit_a: bool = False
    conf_tourney_early_exit_b: bool = False

    # --- INTANGIBLES ---
    team_exp_a: float = 2.0
    team_exp_b: float = 2.0
    coach_tourney_wins_a: int = 0
    coach_tourney_wins_b: int = 0
    coach_e8_plus_a: int = 0
    coach_e8_plus_b: int = 0
    lockdown_dbpr_a: float = 0.0
    lockdown_dbpr_b: float = 0.0
    star_ppg_a: float = 0.0
    star_ppg_b: float = 0.0
    won_prev_round_a: bool = False
    won_prev_round_b: bool = False
    prev_was_upset_a: bool = False
    prev_was_upset_b: bool = False
    first_tourney_a: bool = False
    first_tourney_b: bool = False
    recent_final_fours_a: int = 0
    recent_final_fours_b: int = 0
    comeback_rate_a: float = 0.0
    comeback_rate_b: float = 0.0
    comeback_games_a: int = 0
    comeback_games_b: int = 0
    comeback_confidence_a: float = 0.0
    comeback_confidence_b: float = 0.0

    # SOS / quality
    sos_rank_a: int = 50
    sos_rank_b: int = 50
    q1_wins_a: int = 5
    q1_wins_b: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 3.0) -> float:
    return 1.0 / (1.0 + math.exp(-x * scale))


def _compound_signals(signals: List[Signal], threshold: int = 2) -> Tuple[float, float]:
    """Compound active signals. Returns (raw_shift, compound_multiplier)."""
    active = [s for s in signals if s.met]
    if not active:
        return 0.0, 1.0
    active.sort(key=lambda s: abs(s.weight), reverse=True)
    raw = sum(s.weight for s in active)
    n_active = len(active)
    multiplier = 1.0
    if n_active >= threshold + 1:
        multiplier = 1.0 + 0.1 * (n_active - threshold)
    return raw, min(multiplier, 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1: ROSTER STATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_roster_state(ctx: ScenarioContext) -> Tuple[CategoryResult, CategoryResult]:
    """Evaluate roster health for both teams. Returns (result_for_a, result_for_b)."""
    config = CATEGORY_CONFIG["ROSTER_STATE"]

    def _eval_team(name, suffix):
        inj_pen = getattr(ctx, f"injury_penalty_{suffix}")
        star_lost = getattr(ctx, f"star_lost_{suffix}")
        sii = getattr(ctx, f"sii_{suffix}")
        crippled = getattr(ctx, f"crippled_roster_{suffix}")
        weeks = getattr(ctx, f"crippled_weeks_out_{suffix}")
        bds = getattr(ctx, f"bench_depth_{suffix}")
        rot = getattr(ctx, f"rotation_size_{suffix}")
        top_mins = getattr(ctx, f"top_player_minutes_{suffix}")
        foul = getattr(ctx, f"foul_trouble_rate_{suffix}")
        opp_suffix = "b" if suffix == "a" else "a"
        opp_bds = getattr(ctx, f"bench_depth_{opp_suffix}")
        opp_rot = getattr(ctx, f"rotation_size_{opp_suffix}")
        bpr_share = getattr(ctx, f"star_bpr_share_{suffix}")

        signals = []

        signals.append(Signal(
            name="significant_injury", met=inj_pen > 0.05,
            weight=min(0.10, inj_pen * 0.5),
            raw_value=inj_pen,
            description=f"Injury AdjEM penalty: {inj_pen:.2f}",
        ))
        signals.append(Signal(
            name="star_lost", met=star_lost and bpr_share >= 0.20,
            weight=0.08,
            description=f"Star carrier out (BPR share {bpr_share:.2f})",
        ))
        signals.append(Signal(
            name="star_isolation", met=sii >= 0.08,
            weight=min(0.08, (sii / 0.15) * 0.08),
            raw_value=sii,
            description=f"SII={sii:.3f} (one-man offense risk)",
        ))
        signals.append(Signal(
            name="crippled_roster", met=crippled,
            weight=min(0.12, 0.04 + weeks * 0.01),
            raw_value=weeks,
            description=f"Top scorer out {weeks:.0f} weeks, offense identity collapsed",
        ))
        signals.append(Signal(
            name="shallow_bench", met=bds < 0.35,
            weight=0.04 * ROUND_FATIGUE.get(ctx.round_name, 1.0),
            raw_value=bds,
            description=f"Bench depth {bds:.2f} (shallow < 0.35), round={ctx.round_name}",
        ))
        signals.append(Signal(
            name="depth_disadvantage",
            met=bds < 0.40 and opp_bds - bds > 0.20,
            weight=0.04,
            description=f"Bench gap: opponent {opp_bds:.2f} vs {bds:.2f}",
        ))
        signals.append(Signal(
            name="fatigue_risk",
            met=top_mins >= 37 and rot <= 7,
            weight=0.05 * ROUND_FATIGUE.get(ctx.round_name, 1.0),
            raw_value=top_mins,
            description=f"Star played {top_mins:.0f} min with {rot}-man rotation",
        ))

        raw, mult = _compound_signals(signals, config["compound_threshold"])

        hurt_shift = min(raw * mult, config["max_shift"])

        return CategoryResult(
            category="ROSTER_STATE",
            signals=signals,
            raw_shift=raw,
            compound_multiplier=mult,
            final_shift=hurt_shift,
            direction=name,
            confidence=min(1.0, hurt_shift / config["max_shift"]),
            max_shift=config["max_shift"],
            explanation=f"{name}: {sum(1 for s in signals if s.met)} roster signals, shift={hurt_shift:.3f}",
        )

    return _eval_team(ctx.team_a_name, "a"), _eval_team(ctx.team_b_name, "b")


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2: MATCHUP STYLE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_matchup_style(ctx: ScenarioContext) -> CategoryResult:
    """Evaluate style matchup — who has the stylistic edge?

    Positive shift = favors team_a, negative = favors team_b.
    """
    config = CATEGORY_CONFIG["MATCHUP_STYLE"]
    signals = []

    orb_diff = ctx.orb_pct_a - ctx.orb_pct_b
    drb_diff = ctx.drb_pct_a - ctx.drb_pct_b
    rbm_diff = ctx.rbm_a - ctx.rbm_b

    reb_edge = orb_diff * 0.6 + drb_diff * 0.4
    has_reb_edge = abs(reb_edge) > 0.02 or abs(rbm_diff) > 0.06
    if ctx.tourney_games_a >= 2 and ctx.tourney_games_b >= 2 and ctx.tourney_orb_pct_a > 0 and ctx.tourney_orb_pct_b > 0:
        reb_edge_tourney = ctx.tourney_orb_pct_a - ctx.tourney_orb_pct_b
        has_reb_edge = has_reb_edge or abs(reb_edge_tourney) > 0.03

    reb_weight = min(0.06, abs(reb_edge) * 1.5 + abs(rbm_diff) * 0.5)
    reb_direction = 1.0 if reb_edge > 0 else -1.0

    signals.append(Signal(
        name="rebounding_edge",
        met=has_reb_edge,
        weight=reb_weight * reb_direction,
        raw_value=reb_edge,
        description=f"Reb edge: ORB diff={orb_diff:+.3f}, DRB diff={drb_diff:+.3f}, RBM diff={rbm_diff:+.3f}",
    ))

    phys_a = PHYSICAL_CONFERENCES.get(ctx.conference_a, 0)
    phys_b = PHYSICAL_CONFERENCES.get(ctx.conference_b, 0)
    opp_finesse_a = ctx.conference_b in FINESSE_CONFERENCES
    opp_finesse_b = ctx.conference_a in FINESSE_CONFERENCES
    phys_gap = phys_a - phys_b
    phys_met = abs(phys_gap) >= 2 or (phys_a >= 3 and opp_finesse_a) or (phys_b >= 3 and opp_finesse_b)
    phys_weight = min(0.05, abs(phys_gap) * 0.015)
    if opp_finesse_a and phys_a >= 3:
        phys_weight += 0.02
    if opp_finesse_b and phys_b >= 3:
        phys_weight += 0.02
    phys_direction = 1.0 if phys_gap > 0 else (-1.0 if phys_gap < 0 else 0.0)

    signals.append(Signal(
        name="conference_physicality",
        met=phys_met,
        weight=phys_weight * phys_direction,
        raw_value=float(phys_gap),
        description=f"Physicality: {ctx.conference_a}({phys_a}) vs {ctx.conference_b}({phys_b})",
    ))

    paint_diff = ctx.tourney_paint_pct_a - ctx.tourney_paint_pct_b
    has_paint_edge = abs(paint_diff) > 0.05 and ctx.tourney_games_a >= 2 and ctx.tourney_games_b >= 2
    paint_weight = min(0.04, abs(paint_diff) * 0.3)
    paint_dir = 1.0 if paint_diff > 0 else -1.0

    signals.append(Signal(
        name="paint_dominance",
        met=has_paint_edge,
        weight=paint_weight * paint_dir,
        raw_value=paint_diff,
        description=f"Tournament paint scoring: A={ctx.tourney_paint_pct_a:.1%} B={ctx.tourney_paint_pct_b:.1%}",
    ))

    tpt_a_heavy = ctx.three_pt_share_a > 0.38
    tpt_b_heavy = ctx.three_pt_share_b > 0.38
    if tpt_a_heavy or tpt_b_heavy:
        tpt_weight = 0.03
        if tpt_a_heavy and not tpt_b_heavy:
            tpt_dir = -1.0  # A is 3PT dependent = variance risk for A
        elif tpt_b_heavy and not tpt_a_heavy:
            tpt_dir = 1.0
        else:
            tpt_dir = 0.0
            tpt_weight = 0.01
        signals.append(Signal(
            name="three_pt_dependency",
            met=True,
            weight=tpt_weight * tpt_dir,
            raw_value=ctx.three_pt_share_a - ctx.three_pt_share_b,
            description=f"3PT share: A={ctx.three_pt_share_a:.1%} B={ctx.three_pt_share_b:.1%}",
        ))
    else:
        signals.append(Signal(name="three_pt_dependency", met=False, description="Neither team 3PT-heavy"))

    high_var_a = ctx.three_pt_std_a > 0.08 and ctx.three_pt_pct_a > 0.30
    high_var_b = ctx.three_pt_std_b > 0.08 and ctx.three_pt_pct_b > 0.30
    ceiling_a = ctx.three_pt_pct_a + 1.3 * ctx.three_pt_std_a
    ceiling_b = ctx.three_pt_pct_b + 1.3 * ctx.three_pt_std_b
    is_close = 0.35 < ctx.p_base < 0.65
    if (high_var_a or high_var_b) and is_close:
        var_diff = (ceiling_a - ceiling_b) if (high_var_a and high_var_b) else (0.015 if high_var_a else -0.015)
        signals.append(Signal(
            name="three_pt_variance_upside",
            met=True,
            weight=min(0.02, abs(var_diff) * 0.5) * (1.0 if var_diff > 0 else -1.0),
            raw_value=var_diff,
            description=f"3PT ceiling: A={ceiling_a:.1%} B={ceiling_b:.1%}",
        ))
    else:
        signals.append(Signal(name="three_pt_variance_upside", met=False, description="No 3PT variance edge or not close"))

    raw, mult = _compound_signals(signals, config["compound_threshold"])
    final = max(-config["max_shift"], min(config["max_shift"], raw * mult))

    direction = "team_a" if final > 0.001 else ("team_b" if final < -0.001 else "neutral")

    return CategoryResult(
        category="MATCHUP_STYLE",
        signals=signals,
        raw_shift=raw,
        compound_multiplier=mult,
        final_shift=final,
        direction=direction,
        confidence=min(1.0, abs(final) / config["max_shift"]),
        max_shift=config["max_shift"],
        explanation=f"Style shift: {final:+.3f} ({direction})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3: FORM & TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

def _count_accelerating(ctx: ScenarioContext, suffix: str) -> int:
    """Count how many tournament stat trajectories are accelerating."""
    count = 0
    fg_slope = getattr(ctx, f"trajectory_fg_pct_{suffix}")
    ast_slope = getattr(ctx, f"trajectory_ast_{suffix}")
    tov_slope = getattr(ctx, f"trajectory_tov_{suffix}")
    paint_slope = getattr(ctx, f"trajectory_paint_{suffix}")
    bench_slope = getattr(ctx, f"trajectory_bench_{suffix}")

    if fg_slope > 2.0:
        count += 1
    if ast_slope > 1.5:
        count += 1
    if tov_slope < -1.0:
        count += 1
    if paint_slope > 2.0:
        count += 1
    if bench_slope > 1.0:
        count += 1
    return count


def evaluate_form_trajectory(ctx: ScenarioContext) -> CategoryResult:
    """Evaluate form and trajectory for both teams. Positive = favors A."""
    config = CATEGORY_CONFIG["FORM_TRAJECTORY"]
    signals = []

    # Recency: scale season vs tournament signals by round
    tourney_w = RECENCY_TOURNEY_WEIGHT.get(ctx.round_name, 0.05)
    season_w = 1.0 - tourney_w

    # Season form (dampened in later rounds)
    form_diff = ctx.form_trend_a - ctx.form_trend_b
    mom_diff = ctx.momentum_a - ctx.momentum_b
    season_signal = form_diff * 0.5 + mom_diff * 0.5

    signals.append(Signal(
        name="season_form_differential",
        met=abs(season_signal) > 0.08,
        weight=min(0.04, abs(season_signal) * 0.2) * (1.0 if season_signal > 0 else -1.0) * season_w,
        raw_value=season_signal,
        description=f"Form trend A={ctx.form_trend_a:.2f} B={ctx.form_trend_b:.2f}, Momentum A={ctx.momentum_a:.2f} B={ctx.momentum_b:.2f} [season_w={season_w:.0%}]",
    ))

    collapse_a = ctx.losses_last_6_a >= 3 and ctx.form_trend_a < 0.30
    collapse_b = ctx.losses_last_6_b >= 3 and ctx.form_trend_b < 0.30

    if collapse_a and not collapse_b:
        signals.append(Signal(
            name="form_collapse",
            met=True, weight=-0.04,
            description=f"Team A collapsing: {ctx.losses_last_6_a} losses in last 6, form_trend={ctx.form_trend_a:.2f}",
        ))
    elif collapse_b and not collapse_a:
        signals.append(Signal(
            name="form_collapse",
            met=True, weight=0.04,
            description=f"Team B collapsing: {ctx.losses_last_6_b} losses in last 6, form_trend={ctx.form_trend_b:.2f}",
        ))
    else:
        signals.append(Signal(name="form_collapse", met=False, description="No clear collapse"))

    # H1/H2 early-lead mirage
    mirage_a = ctx.offensive_burst_a > 3.0 and ctx.q3_adj_a < -1.5
    mirage_b = ctx.offensive_burst_b > 3.0 and ctx.q3_adj_b < -1.5
    if mirage_a and not mirage_b:
        signals.append(Signal(
            name="early_lead_mirage", met=True, weight=-0.03,
            description=f"Team A: burst={ctx.offensive_burst_a:.1f} but Q3={ctx.q3_adj_a:+.1f} (fades after half)",
        ))
    elif mirage_b and not mirage_a:
        signals.append(Signal(
            name="early_lead_mirage", met=True, weight=0.03,
            description=f"Team B: burst={ctx.offensive_burst_b:.1f} but Q3={ctx.q3_adj_b:+.1f} (fades after half)",
        ))
    else:
        signals.append(Signal(name="early_lead_mirage", met=False, description="No mirage pattern"))

    # Tournament momentum (amplified in later rounds by recency weight)
    tmom_diff = ctx.tourney_momentum_a - ctx.tourney_momentum_b
    tmom_scale = 1.0 + tourney_w
    if abs(tmom_diff) > 0.005:
        tmom_weight = max(-0.04, min(0.04, tmom_diff * tmom_scale))
        signals.append(Signal(
            name="tournament_momentum",
            met=abs(tmom_diff) > 0.01,
            weight=tmom_weight,
            raw_value=tmom_diff,
            description=f"Tourney momentum A={ctx.tourney_momentum_a:+.3f} B={ctx.tourney_momentum_b:+.3f} [tourney_w={tourney_w:.0%}]",
        ))
    else:
        signals.append(Signal(name="tournament_momentum", met=False, description="No momentum differential"))

    # Tournament trajectory
    has_traj_a = ctx.tourney_games_a >= 2
    has_traj_b = ctx.tourney_games_b >= 2
    accel_a = _count_accelerating(ctx, "a") if has_traj_a else 0
    accel_b = _count_accelerating(ctx, "b") if has_traj_b else 0

    traj_decay = TRAJECTORY_DECAY.get(ctx.round_name, 1.0)
    trajectory_multiplier = 1.0
    traj_signal = 0.0

    both_have_data = has_traj_a and has_traj_b
    one_sided = (has_traj_a or has_traj_b) and not both_have_data
    ONE_SIDED_CAP = 0.02

    if both_have_data and accel_a >= 3 and accel_b < 2:
        traj_signal = 0.03 * traj_decay
        trajectory_multiplier = 1.15
        signals.append(Signal(
            name="tournament_trajectory", met=True, weight=traj_signal,
            description=f"Team A ACCELERATING on {accel_a}/5 stats vs B {accel_b}/5. decay={traj_decay:.2f}",
        ))
    elif both_have_data and accel_b >= 3 and accel_a < 2:
        traj_signal = -0.03 * traj_decay
        trajectory_multiplier = 1.15
        signals.append(Signal(
            name="tournament_trajectory", met=True, weight=traj_signal,
            description=f"Team B ACCELERATING on {accel_b}/5 stats vs A {accel_a}/5. decay={traj_decay:.2f}",
        ))
    elif both_have_data and (accel_a >= 2 or accel_b >= 2):
        net = accel_a - accel_b
        traj_signal = net * 0.01 * traj_decay
        signals.append(Signal(
            name="tournament_trajectory",
            met=abs(net) >= 1,
            weight=traj_signal,
            description=f"Trajectory: A {accel_a}/5, B {accel_b}/5. decay={traj_decay:.2f}",
        ))
    elif one_sided:
        who = "A" if has_traj_a else "B"
        accel = accel_a if has_traj_a else accel_b
        raw = min(ONE_SIDED_CAP, accel * 0.005) * traj_decay
        if has_traj_b:
            raw = -raw
        traj_signal = raw
        signals.append(Signal(
            name="tournament_trajectory",
            met=accel >= 2,
            weight=traj_signal,
            description=f"One-sided trajectory: {who} has {accel}/5 accel (opponent no data). CAPPED at {ONE_SIDED_CAP}. decay={traj_decay:.2f}",
        ))
    else:
        signals.append(Signal(name="tournament_trajectory", met=False, description="No tournament trajectory data"))

    raw, mult = _compound_signals(signals, config["compound_threshold"])
    mult = max(mult, trajectory_multiplier) if abs(traj_signal) > 0 else mult
    final = max(-config["max_shift"], min(config["max_shift"], raw * mult))

    direction = "team_a" if final > 0.001 else ("team_b" if final < -0.001 else "neutral")

    return CategoryResult(
        category="FORM_TRAJECTORY",
        signals=signals,
        raw_shift=raw,
        compound_multiplier=mult,
        final_shift=final,
        direction=direction,
        confidence=min(1.0, abs(final) / config["max_shift"]),
        max_shift=config["max_shift"],
        explanation=f"Form/trajectory shift: {final:+.3f} ({direction}), trajectory_mult={trajectory_multiplier:.1f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 4: INTANGIBLES
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_intangibles(ctx: ScenarioContext) -> CategoryResult:
    """Evaluate human/contextual factors. Positive = favors A."""
    config = CATEGORY_CONFIG["INTANGIBLES"]
    signals = []

    exp_diff = ctx.team_exp_a - ctx.team_exp_b
    exp_met = abs(exp_diff) > 0.3
    signals.append(Signal(
        name="experience_edge",
        met=exp_met,
        weight=min(0.04, abs(exp_diff) * 0.04) * (1.0 if exp_diff > 0 else -1.0),
        raw_value=exp_diff,
        description=f"Experience: A={ctx.team_exp_a:.2f} B={ctx.team_exp_b:.2f} (diff={exp_diff:+.2f})",
    ))

    coach_diff = (ctx.coach_tourney_wins_a + ctx.coach_e8_plus_a * 3) - \
                 (ctx.coach_tourney_wins_b + ctx.coach_e8_plus_b * 3)
    coach_met = abs(coach_diff) >= 3
    coach_weight = min(0.04, abs(coach_diff) * 0.005)
    signals.append(Signal(
        name="coach_tournament_dna",
        met=coach_met,
        weight=coach_weight * (1.0 if coach_diff > 0 else -1.0),
        raw_value=float(coach_diff),
        description=f"Coach DNA: A={ctx.coach_tourney_wins_a}W/{ctx.coach_e8_plus_a}E8+, B={ctx.coach_tourney_wins_b}W/{ctx.coach_e8_plus_b}E8+",
    ))

    youth_a = ctx.team_exp_a < 1.5 and ctx.first_tourney_a
    youth_b = ctx.team_exp_b < 1.5 and ctx.first_tourney_b
    if youth_a and not youth_b:
        signals.append(Signal(
            name="youth_pressure", met=True, weight=-0.03,
            description=f"Team A: young ({ctx.team_exp_a:.2f}) + first tournament",
        ))
    elif youth_b and not youth_a:
        signals.append(Signal(
            name="youth_pressure", met=True, weight=0.03,
            description=f"Team B: young ({ctx.team_exp_b:.2f}) + first tournament",
        ))
    else:
        signals.append(Signal(name="youth_pressure", met=False, description="No youth pressure"))

    ld_a = ctx.lockdown_dbpr_a >= 3.0 and ctx.star_ppg_b >= 16.0
    ld_b = ctx.lockdown_dbpr_b >= 3.0 and ctx.star_ppg_a >= 16.0
    if ld_a and not ld_b:
        signals.append(Signal(
            name="lockdown_defender", met=True, weight=0.03,
            description=f"Team A lockdown (DBPR={ctx.lockdown_dbpr_a:.1f}) vs B star ({ctx.star_ppg_b:.0f} PPG)",
        ))
    elif ld_b and not ld_a:
        signals.append(Signal(
            name="lockdown_defender", met=True, weight=-0.03,
            description=f"Team B lockdown (DBPR={ctx.lockdown_dbpr_b:.1f}) vs A star ({ctx.star_ppg_a:.0f} PPG)",
        ))
    else:
        signals.append(Signal(name="lockdown_defender", met=False, description="No lockdown matchup"))

    upset_momentum_a = ctx.won_prev_round_a and ctx.prev_was_upset_a
    upset_momentum_b = ctx.won_prev_round_b and ctx.prev_was_upset_b
    if upset_momentum_a and not upset_momentum_b:
        signals.append(Signal(
            name="upset_momentum", met=True, weight=0.02,
            description=f"Team A carries upset momentum from prior round",
        ))
    elif upset_momentum_b and not upset_momentum_a:
        signals.append(Signal(
            name="upset_momentum", met=True, weight=-0.02,
            description=f"Team B carries upset momentum from prior round",
        ))
    else:
        signals.append(Signal(name="upset_momentum", met=False, description="No upset momentum"))

    cb_has_data = ctx.comeback_games_a >= 2 or ctx.comeback_games_b >= 2
    cb_diff = ctx.comeback_rate_a - ctx.comeback_rate_b
    if cb_has_data and abs(cb_diff) > 0.10:
        signals.append(Signal(
            name="comeback_resilience",
            met=True,
            weight=min(0.03, abs(cb_diff) * 0.08) * (1.0 if cb_diff > 0 else -1.0),
            raw_value=cb_diff,
            description=(
                f"Comeback rate: A={ctx.comeback_rate_a:.0%} ({ctx.comeback_games_a}g) "
                f"B={ctx.comeback_rate_b:.0%} ({ctx.comeback_games_b}g)"
            ),
        ))
    else:
        signals.append(Signal(name="comeback_resilience", met=False, description="No comeback differential"))

    # Tournament pedigree: recent Final Four appearances (2-year window)
    ped_a = ctx.recent_final_fours_a
    ped_b = ctx.recent_final_fours_b
    ped_diff = ped_a - ped_b
    if abs(ped_diff) >= 1:
        ped_weight = min(0.02, abs(ped_diff) * 0.015) * (1.0 if ped_diff > 0 else -1.0)
        signals.append(Signal(
            name="tournament_pedigree", met=True, weight=ped_weight,
            raw_value=float(ped_diff),
            description=f"Recent F4s: A={ped_a} B={ped_b} (program DNA advantage)",
        ))
    else:
        signals.append(Signal(name="tournament_pedigree", met=False, description="No pedigree edge"))

    raw, mult = _compound_signals(signals, config["compound_threshold"])
    final = max(-config["max_shift"], min(config["max_shift"], raw * mult))

    direction = "team_a" if final > 0.001 else ("team_b" if final < -0.001 else "neutral")

    return CategoryResult(
        category="INTANGIBLES",
        signals=signals,
        raw_shift=raw,
        compound_multiplier=mult,
        final_shift=final,
        direction=direction,
        confidence=min(1.0, abs(final) / config["max_shift"]),
        max_shift=config["max_shift"],
        explanation=f"Intangibles shift: {final:+.3f} ({direction})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-CATEGORY INTERACTION + COHERENCE
# ─────────────────────────────────────────────────────────────────────────────

def compute_coherence(categories: List[CategoryResult]) -> Tuple[float, float]:
    """Compute how aligned the evidence is across categories.

    Returns (coherence_score, coherence_bonus).
    coherence_score: 0.0-1.0 (1.0 = all categories agree)
    coherence_bonus: multiplier on total shift (0.0 to 0.15)
    """
    active = [c for c in categories if abs(c.final_shift) > 0.005]
    if len(active) <= 1:
        return 0.0, 0.0

    favors_a = sum(1 for c in active if c.final_shift > 0.005)
    favors_b = sum(1 for c in active if c.final_shift < -0.005)
    n_aligned = max(favors_a, favors_b)
    n_total = len(active)

    coherence = n_aligned / n_total

    if coherence > COHERENCE_AMPLIFY_THRESHOLD:
        bonus = 0.3 * (coherence - COHERENCE_AMPLIFY_THRESHOLD)
    else:
        bonus = 0.0

    return coherence, bonus


def compute_net_shift(roster_a: CategoryResult, roster_b: CategoryResult,
                      style: CategoryResult, form: CategoryResult,
                      intangibles: CategoryResult) -> float:
    """Compute net shift toward team_a.

    Roster results are penalties (hurt the team they apply to).
    Style/form/intangibles are directional (positive = favors A).
    """
    roster_shift = -roster_a.final_shift + roster_b.final_shift
    return roster_shift + style.final_shift + form.final_shift + intangibles.final_shift


def compute_uncertainty(ctx: ScenarioContext, coherence: float) -> float:
    """Estimate uncertainty used to compress overconfident late-round edges."""
    avg_live_conf = (ctx.tourney_data_confidence_a + ctx.tourney_data_confidence_b) / 2.0
    avg_comeback_conf = (ctx.comeback_confidence_a + ctx.comeback_confidence_b) / 2.0
    closeness = max(0.0, 1.0 - abs(ctx.p_base - 0.5) / 0.25)
    three_pt_vol = min(1.0, (ctx.three_pt_std_a + ctx.three_pt_std_b) / 0.24)
    uncertainty = (
        0.06 * closeness +
        0.06 * (1.0 - avg_live_conf) +
        0.03 * (1.0 - avg_comeback_conf) +
        0.03 * (1.0 - coherence) +
        0.02 * three_pt_vol
    )
    return min(0.18, max(0.02, uncertainty))


# ─────────────────────────────────────────────────────────────────────────────
# MASTER: evaluate_scenario
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scenario(ctx: ScenarioContext) -> ScenarioResult:
    """Run the full 4-category scenario analysis.

    Returns ScenarioResult with p_final and full category breakdown.
    """
    roster_a, roster_b = evaluate_roster_state(ctx)
    style = evaluate_matchup_style(ctx)
    form = evaluate_form_trajectory(ctx)
    intangibles = evaluate_intangibles(ctx)

    all_directional = [style, form, intangibles]
    # Roster contributes to alignment check too
    if roster_a.final_shift > 0.01:
        # A has roster damage = effectively favors B
        all_directional.append(CategoryResult(
            category="ROSTER_A_DAMAGE", final_shift=-roster_a.final_shift, direction="team_b",
            max_shift=0.20,
        ))
    if roster_b.final_shift > 0.01:
        all_directional.append(CategoryResult(
            category="ROSTER_B_DAMAGE", final_shift=roster_b.final_shift, direction="team_a",
            max_shift=0.20,
        ))

    coherence, coherence_bonus = compute_coherence(all_directional)

    net_shift = compute_net_shift(roster_a, roster_b, style, form, intangibles)

    if ((roster_a.final_shift > 0.05 and style.final_shift < -0.01) or
            (roster_b.final_shift > 0.05 and style.final_shift > 0.01)):
        net_shift += style.final_shift * 0.30

    amplified_shift = net_shift * (1.0 + coherence_bonus)

    # Base-probability dampening: shifts near p=0 or p=1 are compressed
    # At p=0.50 dampening=1.0, at p=0.12 dampening=0.42, floor at 0.35
    dampening = max(0.35, 4.0 * ctx.p_base * (1.0 - ctx.p_base))
    amplified_shift *= dampening

    amplified_shift = max(-MAX_SCENARIO_SHIFT, min(MAX_SCENARIO_SHIFT, amplified_shift))

    p_shifted = max(0.02, min(0.98, ctx.p_base + amplified_shift))
    uncertainty = compute_uncertainty(ctx, coherence)
    p_final = max(0.02, min(0.98, 0.5 + (p_shifted - 0.5) * (1.0 - uncertainty)))

    categories = [roster_a, roster_b, style, form, intangibles]

    narrative_parts = []
    if abs(style.final_shift) > 0.01:
        dominant = "team_a" if style.final_shift > 0 else "team_b"
        narrative_parts.append(f"{'A' if dominant == 'team_a' else 'B'} has stylistic edge")
    if abs(form.final_shift) > 0.01:
        dominant = "team_a" if form.final_shift > 0 else "team_b"
        narrative_parts.append(f"{'A' if dominant == 'team_a' else 'B'} has form/trajectory advantage")
    if roster_a.final_shift > 0.02:
        narrative_parts.append(f"A has roster concerns")
    if roster_b.final_shift > 0.02:
        narrative_parts.append(f"B has roster concerns")
    if coherence >= 0.75:
        narrative_parts.append(f"Evidence is highly coherent ({coherence:.0%})")

    return ScenarioResult(
        team_a_name=ctx.team_a_name,
        team_b_name=ctx.team_b_name,
        p_base=ctx.p_base,
        p_final=p_final,
        total_shift=amplified_shift,
        categories=categories,
        coherence=coherence,
        coherence_bonus=coherence_bonus,
        uncertainty=uncertainty,
        confidence_post=abs(p_final - 0.5) * 2.0,
        narrative="; ".join(narrative_parts) if narrative_parts else "No significant scenario signals",
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def scenario_report(result: ScenarioResult) -> str:
    """Generate human-readable fired-only scenario report."""
    lines = []
    a = result.team_a_name
    b = result.team_b_name

    lines.append(f"\n{'─' * 80}")
    lines.append(f"  {a} vs {b} — Scenario Analysis")
    lines.append(f"  p_base (ensemble): {result.p_base:.1%}")
    lines.append(f"{'─' * 80}")

    for cat in result.categories:
        fired_signals = [s for s in cat.signals if s.met]
        if not fired_signals and abs(cat.final_shift) < 0.001:
            lines.append(f"\n  {cat.category}: Neutral (no signals fired)")
            continue

        direction_label = ""
        if cat.direction == "team_a" or (cat.category == "ROSTER_STATE" and cat.final_shift > 0.01):
            team_label = cat.explanation.split(":")[0] if ":" in cat.explanation else a
            if "ROSTER" in cat.category:
                direction_label = f"Hurts {team_label}"
            else:
                direction_label = f"Favors {a}" if cat.final_shift > 0 else f"Favors {b}"
        elif cat.direction == "team_b":
            direction_label = f"Favors {b}"
        elif cat.direction:
            direction_label = f"Favors {cat.direction}"

        lines.append(f"\n  {cat.category}: {direction_label} (shift: {cat.final_shift:+.3f}, confidence: {cat.confidence:.0%})")

        for s in fired_signals:
            lines.append(f"    [{s.name}] weight={s.weight:+.3f}: {s.description}")

        if cat.compound_multiplier > 1.0:
            lines.append(f"    Compounding: {len(fired_signals)} signals -> {cat.compound_multiplier:.1f}x multiplier")

    lines.append(f"\n  CROSS-CATEGORY:")
    lines.append(f"    Coherence: {result.coherence:.0%} -> bonus: {result.coherence_bonus:+.0%}")
    lines.append(f"    Net shift: {result.total_shift:+.3f} (before: {result.p_base:.1%}, after: {result.p_final:.1%})")
    lines.append(f"    Uncertainty: {result.uncertainty:.1%} -> post confidence: {result.confidence_post:.1%}")

    lines.append(f"\n  NARRATIVE: {result.narrative}")
    lines.append(f"\n  p_final: {result.p_final:.1%} ({a}) vs {1-result.p_final:.1%} ({b})")
    lines.append(f"{'─' * 80}")

    return "\n".join(lines)
