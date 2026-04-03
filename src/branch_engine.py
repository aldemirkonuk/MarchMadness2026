"""Branch Engine: Conditional probability modifiers on top of the ROOT model.

Architecture:
  ROOT (Model A, 73.7%) produces P_base for each matchup.
  BranchEngine scans for scenario-specific conditions and produces
  a probability MODIFIER that shifts P_base.

Key design principles:
  1. ROOT is never touched. It's the bulletproof 73.7% foundation.
  2. Each branch has a SCALING CURVE from min (barely matters) to max (devastating).
  3. Inside each branch, sub-conditions COMPOUND:
     - Additive base: each sub-condition adds to the severity score
     - At 3+ sub-conditions: multiplicative tipping point kicks in
  4. Cross-branch: branches compound against SURVIVAL PROBABILITY.
     If branch A says "team loses 8% chance" and branch B says "team loses 5%",
     the team's survival goes: 1.0 → 0.92 → 0.92 * 0.95 = 0.874.
     Net shift = -12.6%, not -13%. Slightly sub-additive but allows BIG swings
     when many branches fire.  Critically, this means 5 mild branches produce
     a LARGER shift than their sum — problems compound like they do in reality.
  5. No cap on total shift. Full flips allowed. If the evidence says flip, flip.

Each branch has:
  - trigger(): bool — does this scenario apply?
  - severity(): float 0→1 — how bad is it? (scaling curve)
  - max_shift: float — the branch's ceiling (varies per branch type)
  - direction: "underdog" | "favorite" | "variance" — which way to push
  - sub_conditions: list of (condition, weight) pairs that feed severity()
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SubCondition:
    """One factor inside a branch that contributes to severity."""
    name: str
    met: bool = False          # did this sub-condition trigger?
    weight: float = 1.0        # how much it contributes (1.0 = normal)
    value: float = 0.0         # raw measurement (for scaling)
    description: str = ""      # human-readable explanation


@dataclass
class BranchResult:
    """Output of a single branch evaluation."""
    branch_name: str
    triggered: bool = False
    severity: float = 0.0      # 0.0 → 1.0 (scaling curve output)
    shift: float = 0.0         # actual probability shift (severity × max_shift)
    max_shift: float = 0.0     # this branch's ceiling
    direction: str = "underdog"  # "underdog", "favorite", or "variance"
    sub_conditions: List[SubCondition] = field(default_factory=list)
    explanation: str = ""      # why this branch fired


@dataclass
class BranchEngineResult:
    """Aggregated output of all branches for one team in a matchup."""
    team_name: str
    p_base: float = 0.0       # ROOT model probability
    p_final: float = 0.0      # after all branch modifications
    total_shift: float = 0.0  # net probability change
    variance_widen: float = 0.0  # distribution widening (from UNICORN etc.)
    branches_fired: List[BranchResult] = field(default_factory=list)
    survival_multiplier: float = 1.0  # product of (1 - branch_shift) for each branch


# ─────────────────────────────────────────────────────────────────────────────
# Severity scaling curve
# ─────────────────────────────────────────────────────────────────────────────

def severity_curve(sub_conditions: List[SubCondition],
                   tipping_point: int = 3) -> float:
    """Compute severity from sub-conditions using hybrid additive/multiplicative.

    Phase 1 (additive): Sum weighted sub-conditions up to tipping_point.
    Phase 2 (multiplicative): Beyond tipping_point, each additional sub-condition
        MULTIPLIES severity by (1 + weight), modeling cascade/collapse.

    Returns: float in [0, 1] range (can exceed 1.0 in extreme cases,
             which maps to max_shift via capping).

    Examples:
        1 sub (w=0.3):              severity = 0.30  → mild
        2 subs (w=0.3, w=0.25):     severity = 0.55  → moderate
        3 subs (w=0.3, w=0.25, w=0.2): severity = 0.75 → tipping point
        4 subs (above + w=0.15):    severity = 0.75 * 1.15 = 0.86 → cascade
        5 subs (above + w=0.10):    severity = 0.86 * 1.10 = 0.95 → devastating
    """
    active = [sc for sc in sub_conditions if sc.met]
    if not active:
        return 0.0

    # Sort by weight descending so biggest factors hit first
    active.sort(key=lambda sc: sc.weight, reverse=True)

    # Phase 1: additive accumulation
    additive_count = min(len(active), tipping_point)
    severity = sum(sc.weight for sc in active[:additive_count])

    # Phase 2: multiplicative cascade for subs beyond tipping point
    for sc in active[additive_count:]:
        severity *= (1.0 + sc.weight)

    return min(severity, 1.5)  # soft cap at 1.5 (allows oversaturation)


def compute_shift(severity: float, max_shift: float,
                  min_shift: float = 0.008) -> float:
    """Convert severity (0→1+) into a probability shift.

    Uses a sigmoid-like curve so that:
      - Low severity → barely above min_shift (0.8%)
      - Mid severity → proportional growth
      - High severity → approaches max_shift asymptotically
      - Severity > 1.0 → CAN exceed max_shift (catastrophic override)

    The curve: shift = min_shift + (max_shift - min_shift) * tanh(severity * 1.5)
    tanh(0) = 0, tanh(1.5) ≈ 0.91, tanh(2.25) ≈ 0.98
    """
    if severity <= 0:
        return 0.0

    # tanh gives us the smooth 0→1 curve with diminishing returns
    curve = math.tanh(severity * 1.5)

    shift = min_shift + (max_shift - min_shift) * curve

    # If severity > 1.0 (tipping point cascade), allow overshoot up to 1.3x max
    if severity > 1.0:
        overshoot = (severity - 1.0) * 0.3 * max_shift
        shift += overshoot

    return shift


# ─────────────────────────────────────────────────────────────────────────────
# Branch definitions
# ─────────────────────────────────────────────────────────────────────────────

# Branch ceiling configurations: (max_shift, branch_type)
# Player branches can devastate (up to 15%)
# Team branches are significant (up to 10%)
# Context branches are lighter (up to 6%)
BRANCH_CONFIG = {
    # ── TIER 1: Player-level (ceiling 10-15%) ────────────────────────────
    "STAR_ISOLATION": {
        "max_shift": 0.15,
        "direction": "underdog",
        "description": "Team's offense runs through ONE player. Predictable + fatigued.",
    },
    "ALPHA_VACUUM": {
        "max_shift": 0.15,
        "direction": "underdog",
        "description": "Star left → chemistry collapsed → remaining players selfish.",
    },
    "UNICORN_PLAYER": {
        "max_shift": 0.08,
        "direction": "variance",  # widens distribution, doesn't shift
        "description": "Specialist who can single-handedly swing the game.",
    },
    "LOCKDOWN_DEFENDER": {
        "max_shift": 0.08,
        "direction": "favorite",
        "description": "Elite defender neutralizes opponent's primary scorer.",
    },

    # ── TIER 2: Team-level (ceiling 6-10%) ───────────────────────────────
    "FATIGUE_TRAP": {
        "max_shift": 0.10,
        "direction": "underdog",
        "description": "Shallow rotation + heavy minutes in prior round.",
    },
    "REBOUNDING_MISMATCH": {
        "max_shift": 0.08,
        "direction": "rebounder",  # toward the better rebounder
        "description": "Dominant rebounding team in a competitive matchup.",
    },
    "EARLY_LEAD_MIRAGE": {
        "max_shift": 0.07,
        "direction": "underdog",
        "description": "Team starts fast but collapses after halftime adjustments.",
    },
    "CONFERENCE_PHYSICALITY": {
        "max_shift": 0.08,  # increased from 0.04 — WCC/Gonzaga pattern
        "direction": "physical_team",
        "description": "Physical conference team vs finesse/weak conference team. Includes SOS inflation detection.",
    },

    # ── TIER 3: Context-level (ceiling 3-6%) ─────────────────────────────
    "YOUTH_UNDER_PRESSURE": {
        "max_shift": 0.06,
        "direction": "underdog",
        "description": "Young/inexperienced team in tournament pressure.",
    },
    "FORM_COLLAPSE": {
        "max_shift": 0.06,
        "direction": "underdog",
        "description": "Team on a losing streak entering the tournament.",
    },
    "COACH_TOURNAMENT_DNA": {
        "max_shift": 0.05,
        "direction": "experienced_coach",
        "description": "Coach's tournament track record in tight games.",
    },
    "THREE_PT_VARIANCE_BOMB": {
        "max_shift": 0.06,
        "direction": "variance",
        "description": "3PT-dependent team → wider outcome distribution.",
    },
    "DEPTH_ADVANTAGE": {
        "max_shift": 0.06,
        "direction": "deeper_team",
        "description": "Bench depth advantage compounds in later rounds.",
    },
    "REVENGE_MOMENTUM": {
        "max_shift": 0.04,
        "direction": "momentum_team",
        "description": "Psychological boost from dramatic prior-round win.",
    },

    # ── BRANCH 15: Long-term star absence (Gonzaga/Huff pattern) ──────
    "CRIPPLED_ROSTER": {
        "max_shift": 0.15,
        "direction": "underdog",
        "description": "Top scorer out 4+ weeks → offense identity collapsed, stats inflated by pre-injury play.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-branch compounding: survival probability model
# ─────────────────────────────────────────────────────────────────────────────

def compound_branches(p_base: float,
                      branch_results: List[BranchResult]) -> Tuple[float, float, float]:
    """Compound all triggered branches into a final probability.

    Uses SURVIVAL PROBABILITY model:
      For each branch that fires against team A:
        survival *= (1 - shift)
      p_final = p_base * survival

    This is mathematically correct: if the root says you win 80% of the time,
    and one branch says "you lose 10% of your remaining edge," your new
    probability is 80% * 0.90 = 72%.  If another says "lose 8% more,"
    it's 72% * 0.92 = 66.2%.

    Cross-branch stacking is naturally handled:
      - 5 mild branches (3% each): 0.97^5 = 0.859 → 14.1% total loss
      - 1 devastating branch (15%): 0.85 → 15% total loss
      - 3 moderate + 2 mild: real compound effect

    No artificial cap. If the evidence says flip, flip.

    Returns: (p_final, total_shift, survival_multiplier)
    """
    if not branch_results:
        return p_base, 0.0, 1.0

    # Separate shifts that hurt team_a vs help team_a vs widen variance
    hurt_a = []    # branches that reduce team_a's chances
    help_a = []    # branches that boost team_a's chances
    variance = 0.0 # distribution widening (additive)

    for br in branch_results:
        if not br.triggered or br.shift == 0:
            continue

        if br.direction == "variance":
            variance += br.shift
        elif br.shift > 0:
            hurt_a.append(br.shift)
        else:
            help_a.append(abs(br.shift))

    # Apply hurting branches to team_a's win probability (survival model)
    survival = 1.0
    for shift in hurt_a:
        survival *= (1.0 - shift)

    # Apply helping branches (boost)
    boost = 1.0
    for shift in help_a:
        boost *= (1.0 + shift)

    p_final = p_base * survival * boost

    # Clamp to [0.01, 0.99] — no certainties
    p_final = max(0.01, min(0.99, p_final))

    total_shift = p_final - p_base
    final_survival = survival * boost

    return p_final, total_shift, final_survival


# ─────────────────────────────────────────────────────────────────────────────
# Example: STAR_ISOLATION branch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_star_isolation(team_name: str,
                            sii: float,
                            top_player_minutes: float,
                            top_player_usage_pct: float,
                            second_option_injured: bool,
                            team_record_post_injury: Optional[Tuple[int, int]] = None,
                            recent_form_losses: int = 0) -> BranchResult:
    """Evaluate STAR_ISOLATION for a team.

    Sub-conditions that build severity:
      1. SII score above threshold (0.08-0.15)       → weight 0.30
      2. Top player plays 36+ minutes                  → weight 0.20
      3. Top player usage > 35%                        → weight 0.15
      4. Second-best player is injured/out              → weight 0.20
      5. Team went ≤.500 since second option went down  → weight 0.15
      6. Team lost 2+ of last 4 games                  → weight 0.10

    Example — BYU (Day 1):
      SII = ~0.15 (Dybantsa dominated)               ✓ 0.30
      Dybantsa played 40/40 minutes                   ✓ 0.20
      Dybantsa usage ~40%                             ✓ 0.15
      Saunders torn ACL (season-ending)               ✓ 0.20
      BYU went 2-4 after Saunders injury              ✓ 0.15
      Lost 4 of last 6                                ✓ 0.10
      ──────────────────────────────────────────────────
      6 sub-conditions met → additive to 3, then multiply
      Phase 1: 0.30 + 0.20 + 0.20 = 0.70
      Phase 2: 0.70 * 1.15 * 1.15 * 1.10 = 1.02
      severity = 1.02 → shift = tanh(1.53) * 0.15 ≈ 0.14 (14%)
      BYU's 75% drops to: 75% * (1 - 0.14) = 64.5%
      With other branches (FORM_COLLAPSE etc.) could go lower.
    """
    config = BRANCH_CONFIG["STAR_ISOLATION"]
    subs = []

    # Sub 1: SII score
    sii_met = sii >= 0.08
    sii_weight = min(0.40, (sii / 0.15) * 0.30) if sii >= 0.08 else 0
    subs.append(SubCondition(
        name="high_sii", met=sii_met, weight=sii_weight,
        value=sii, description=f"SII={sii:.3f} (threshold 0.08)"
    ))

    # Sub 2: Top player heavy minutes
    heavy_mins = top_player_minutes >= 36
    subs.append(SubCondition(
        name="heavy_minutes", met=heavy_mins, weight=0.20,
        value=top_player_minutes,
        description=f"Top player {top_player_minutes:.0f} min (threshold 36)"
    ))

    # Sub 3: Top player high usage
    high_usage = top_player_usage_pct >= 0.35
    subs.append(SubCondition(
        name="high_usage", met=high_usage, weight=0.15,
        value=top_player_usage_pct,
        description=f"Usage {top_player_usage_pct*100:.0f}% (threshold 35%)"
    ))

    # Sub 4: Second option injured
    subs.append(SubCondition(
        name="second_option_out", met=second_option_injured, weight=0.20,
        description="Second-best player injured/out"
    ))

    # Sub 5: Bad team record post-injury
    bad_record = False
    if team_record_post_injury:
        wins, losses = team_record_post_injury
        if wins + losses >= 4 and wins / (wins + losses) <= 0.50:
            bad_record = True
    subs.append(SubCondition(
        name="bad_post_injury_record", met=bad_record, weight=0.15,
        description=f"Record post-injury: {team_record_post_injury}"
    ))

    # Sub 6: Recent losses
    losing_form = recent_form_losses >= 2
    subs.append(SubCondition(
        name="losing_recent_form", met=losing_form, weight=0.10,
        value=recent_form_losses,
        description=f"Lost {recent_form_losses} of last 4-6 games"
    ))

    # Check trigger: at least 2 sub-conditions must be met
    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(
            branch_name="STAR_ISOLATION", triggered=False,
            sub_conditions=subs
        )

    # Compute severity via hybrid curve
    sev = severity_curve(subs, tipping_point=3)

    # Compute shift from severity
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="STAR_ISOLATION",
        triggered=True,
        severity=sev,
        shift=shift,
        max_shift=config["max_shift"],
        direction=config["direction"],
        sub_conditions=subs,
        explanation=f"{n_met} sub-conditions met → severity {sev:.2f} → shift {shift*100:.1f}%"
    )


def evaluate_alpha_vacuum(team_name: str,
                          star_lost: bool,
                          star_bpr_share: float,
                          ast_rate_change: Optional[float] = None,
                          record_after_loss: Optional[Tuple[int, int]] = None,
                          team_exp: float = 2.0,
                          star_ppg: float = 0.0) -> BranchResult:
    """Evaluate ALPHA_VACUUM — chemistry collapse after losing the alpha.

    Different from STAR_ISOLATION: this is about CHEMISTRY destruction,
    not about one player carrying. The remaining players can't coexist.

    Sub-conditions:
      1. Star lost (BPR share > 0.20)                → weight 0.25
      2. Assist rate dropped >10% post-injury          → weight 0.25
      3. Team went ≤.500 after star left               → weight 0.20
      4. Young team (exp < 1.8) — no leadership left   → weight 0.15
      5. Star was high-PPG scorer (>16 PPG)            → weight 0.10
      6. Star was the primary facilitator (>4 APG)     → weight 0.10

    Example — UNC (Day 1):
      Wilson out (top-5 pick, ~25% BPR share)        ✓ 0.25
      Assists likely dropped (Trimble selfish play)  ✓ 0.25
      UNC went 5-3 after Wilson injury               ✓ 0.20
      Young team                                     ✓ 0.15
      Wilson high-PPG                                ✓ 0.10
      ─────────────────────────────────────────────────
      5 sub-conditions → additive to 3, multiply rest
      Phase 1: 0.25 + 0.25 + 0.20 = 0.70
      Phase 2: 0.70 * 1.15 * 1.10 = 0.885
      severity = 0.89 → shift ≈ tanh(1.33)*0.15 ≈ 0.13 (13%)
      UNC was 78.5% → 78.5% * (1-0.13) = 68.3%
      PLUS injury model had 0.187 penalty already baked in.
      Combined: closer to 60% → VCU at 40% is a live upset.
    """
    config = BRANCH_CONFIG["ALPHA_VACUUM"]
    subs = []

    # Sub 1: Star lost with significant share
    star_significant = star_lost and star_bpr_share >= 0.20
    star_weight = min(0.35, star_bpr_share * 1.0) if star_significant else 0
    subs.append(SubCondition(
        name="star_lost_significant", met=star_significant,
        weight=star_weight, value=star_bpr_share,
        description=f"Star lost with {star_bpr_share*100:.0f}% BPR share"
    ))

    # Sub 2: Assist rate dropped
    ast_dropped = ast_rate_change is not None and ast_rate_change < -0.10
    subs.append(SubCondition(
        name="assist_rate_dropped", met=ast_dropped, weight=0.25,
        value=ast_rate_change or 0,
        description=f"Assist rate change: {(ast_rate_change or 0)*100:.0f}%"
    ))

    # Sub 3: Bad record after star left
    bad_record = False
    if record_after_loss:
        w, l = record_after_loss
        if w + l >= 3 and w / (w + l) <= 0.55:
            bad_record = True
    subs.append(SubCondition(
        name="bad_record_post_loss", met=bad_record, weight=0.20,
        description=f"Record after star loss: {record_after_loss}"
    ))

    # Sub 4: Young team
    young = team_exp < 1.8
    subs.append(SubCondition(
        name="young_team", met=young, weight=0.15,
        value=team_exp, description=f"Team EXP={team_exp:.2f}"
    ))

    # Sub 5: High PPG scorer lost
    high_ppg = star_ppg >= 16.0
    subs.append(SubCondition(
        name="high_ppg_star", met=high_ppg, weight=0.10,
        value=star_ppg, description=f"Star PPG={star_ppg:.1f}"
    ))

    # Trigger: star must be lost + at least 1 other sub
    n_met = sum(1 for sc in subs if sc.met)
    triggered = star_significant and n_met >= 2

    if not triggered:
        return BranchResult(
            branch_name="ALPHA_VACUUM", triggered=False,
            sub_conditions=subs
        )

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="ALPHA_VACUUM",
        triggered=True,
        severity=sev,
        shift=shift,
        max_shift=config["max_shift"],
        direction=config["direction"],
        sub_conditions=subs,
        explanation=f"{n_met} sub-conditions → severity {sev:.2f} → shift {shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 3: UNICORN_PLAYER — outcome distribution widener
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_unicorn_player(team_name: str,
                            has_conference_stat_leader: bool,
                            has_national_top10_stat: bool,
                            three_pt_pct: float = 0.0,
                            career_threes: int = 0,
                            ppg: float = 0.0,
                            is_underdog: bool = True) -> BranchResult:
    """Evaluate UNICORN_PLAYER — specialist who widens outcome distribution.

    This is NOT a probability shift. It's a VARIANCE WIDENER. The upset
    doesn't become likely, it becomes POSSIBLE.

    Sub-conditions:
      1. Conference stat leader in a category            → weight 0.25
      2. National top-10 in a specific stat              → weight 0.30
      3. 3PT% > 40% (hot shooting specialist)            → weight 0.25
      4. High career 3PT volume (300+ career threes)     → weight 0.15
      5. 18+ PPG scorer on underdog team                 → weight 0.15

    Example — Johnston (High Point, Day 1):
      Big South 3PT leader                              ✓ 0.25
      45.2% from 3 (national elite)                     ✓ 0.30
      3PT% > 40%                                        ✓ 0.25
      415 career threes                                 ✓ 0.15
      → severity ~0.95 → variance widen ≈ 7.6%
    """
    config = BRANCH_CONFIG["UNICORN_PLAYER"]
    subs = []

    subs.append(SubCondition(
        name="conf_stat_leader", met=has_conference_stat_leader, weight=0.25,
        description="Leads conference in a key stat category"
    ))

    subs.append(SubCondition(
        name="national_top10", met=has_national_top10_stat, weight=0.30,
        description="Top-10 nationally in a specific stat"
    ))

    elite_shooter = three_pt_pct >= 0.40
    subs.append(SubCondition(
        name="elite_3pt_shooter", met=elite_shooter, weight=0.25,
        value=three_pt_pct,
        description=f"3PT% = {three_pt_pct*100:.1f}% (threshold 40%)"
    ))

    high_volume = career_threes >= 300
    subs.append(SubCondition(
        name="high_volume_3pt", met=high_volume, weight=0.15,
        value=career_threes,
        description=f"{career_threes} career 3s (threshold 300)"
    ))

    scorer_on_underdog = ppg >= 18.0 and is_underdog
    subs.append(SubCondition(
        name="underdog_scorer", met=scorer_on_underdog, weight=0.15,
        value=ppg,
        description=f"{ppg:.1f} PPG on underdog team"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(branch_name="UNICORN_PLAYER", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="UNICORN_PLAYER", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="variance", sub_conditions=subs,
        explanation=f"UNICORN: {n_met} traits → variance widen {shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 4: LOCKDOWN_DEFENDER — opponent's star neutralizer
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_lockdown_defender(team_name: str,
                               has_switchable_defender: bool,
                               defender_dbpr: float = 0.0,
                               opp_top_scorer_ppg: float = 0.0,
                               team_opp_fg_pct: float = 0.50,
                               opp_star_isolation: float = 0.0) -> BranchResult:
    """Evaluate LOCKDOWN_DEFENDER — elite defender neutralizes opponent's star.

    Sub-conditions:
      1. Switchable defender (guards 2+ positions)        → weight 0.25
      2. Defender has elite DBPR (> 3.0)                  → weight 0.30
      3. Opponent relies heavily on one scorer (SII>0.08) → weight 0.20
      4. Team holds opponents to <42% FG                  → weight 0.15
      5. Opponent star is high-PPG (>18)                  → weight 0.10
    """
    config = BRANCH_CONFIG["LOCKDOWN_DEFENDER"]
    subs = []

    subs.append(SubCondition(
        name="switchable_defender", met=has_switchable_defender, weight=0.25,
        description="Defender guards multiple positions"
    ))

    elite_dbpr = defender_dbpr >= 3.0
    subs.append(SubCondition(
        name="elite_dbpr", met=elite_dbpr, weight=0.30,
        value=defender_dbpr,
        description=f"Defender DBPR={defender_dbpr:.1f} (threshold 3.0)"
    ))

    opp_star_dependent = opp_star_isolation >= 0.08
    subs.append(SubCondition(
        name="opp_star_dependent", met=opp_star_dependent, weight=0.20,
        value=opp_star_isolation,
        description=f"Opponent SII={opp_star_isolation:.3f} (star-dependent)"
    ))

    stingy_defense = team_opp_fg_pct < 0.42
    subs.append(SubCondition(
        name="stingy_team_defense", met=stingy_defense, weight=0.15,
        value=team_opp_fg_pct,
        description=f"Team holds opp to {team_opp_fg_pct*100:.1f}% FG"
    ))

    high_ppg_opp = opp_top_scorer_ppg >= 18.0
    subs.append(SubCondition(
        name="opp_high_ppg_star", met=high_ppg_opp, weight=0.10,
        value=opp_top_scorer_ppg,
        description=f"Opponent star scores {opp_top_scorer_ppg:.1f} PPG"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(branch_name="LOCKDOWN_DEFENDER", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="LOCKDOWN_DEFENDER", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="favorite", sub_conditions=subs,
        explanation=f"LOCKDOWN: {n_met} factors → {shift*100:.1f}% boost for defensive team"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 5: FATIGUE_TRAP — shallow rotation + cumulative exhaustion
# ─────────────────────────────────────────────────────────────────────────────

ROUND_FATIGUE_MULTIPLIER = {
    "R64": 1.0, "R32": 1.2, "S16": 1.4, "E8": 1.6, "F4": 1.8, "NCG": 2.0
}

def evaluate_fatigue_trap(team_name: str,
                          rotation_size: int,
                          opp_rotation_size: int,
                          top_player_prev_minutes: float = 0.0,
                          bench_depth_score: float = 0.5,
                          round_name: str = "R64") -> BranchResult:
    """Evaluate FATIGUE_TRAP — shallow rotation compounds through rounds.

    Sub-conditions:
      1. Rotation ≤ 7 players                            → weight 0.25
      2. Opponent has 3+ more rotation players            → weight 0.20
      3. Top player logged 37+ min in prior round         → weight 0.25
      4. Bench depth score < 0.35 (shallow bench)         → weight 0.15
      5. Round multiplier (later rounds = worse)          → weight 0.15
    """
    config = BRANCH_CONFIG["FATIGUE_TRAP"]
    subs = []

    shallow_rotation = rotation_size <= 7
    subs.append(SubCondition(
        name="shallow_rotation", met=shallow_rotation, weight=0.25,
        value=rotation_size,
        description=f"{rotation_size}-man rotation (threshold ≤7)"
    ))

    depth_gap = opp_rotation_size - rotation_size >= 3
    subs.append(SubCondition(
        name="depth_gap", met=depth_gap, weight=0.20,
        value=opp_rotation_size - rotation_size,
        description=f"Opponent has {opp_rotation_size - rotation_size:+d} more rotation players"
    ))

    heavy_prev_mins = top_player_prev_minutes >= 37.0
    subs.append(SubCondition(
        name="heavy_prev_minutes", met=heavy_prev_mins, weight=0.25,
        value=top_player_prev_minutes,
        description=f"Top player played {top_player_prev_minutes:.0f} min in prior round"
    ))

    weak_bench = bench_depth_score < 0.35
    subs.append(SubCondition(
        name="weak_bench", met=weak_bench, weight=0.15,
        value=bench_depth_score,
        description=f"Bench depth score {bench_depth_score:.2f} (threshold 0.35)"
    ))

    fatigue_mult = ROUND_FATIGUE_MULTIPLIER.get(round_name, 1.0)
    later_round = fatigue_mult > 1.0
    subs.append(SubCondition(
        name="later_round_fatigue", met=later_round,
        weight=0.15 * fatigue_mult,  # weight scales with round
        value=fatigue_mult,
        description=f"Round {round_name}: fatigue multiplier {fatigue_mult:.1f}x"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(branch_name="FATIGUE_TRAP", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="FATIGUE_TRAP", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="underdog", sub_conditions=subs,
        explanation=f"FATIGUE: {n_met} factors in {round_name} → {shift*100:.1f}% shift"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 6: EARLY_LEAD_MIRAGE — fast start, halftime collapse
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_early_lead_mirage(team_name: str,
                                offensive_burst: float,
                                q3_adj_strength: float,
                                team_exp: float = 2.0,
                                is_lower_seed: bool = False,
                                historical_blown_leads: int = 0,
                                opp_q3_adj: float = 0.0) -> BranchResult:
    """Evaluate EARLY_LEAD_MIRAGE — teams that start fast then fade.

    Enhanced: now also checks if opponent is a strong halftime adjuster (q3_adj > 1.5).
    The Gonzaga/Texas pattern: Gonzaga starts fast (burst=9.4, q3_adj=0.3) but
    Texas adjusts well (q3_adj=2.3) → Gonzaga's early lead evaporates.

    Sub-conditions:
      1. High offensive_burst (>3.0 1H advantage)        → weight 0.20
      2. Negative q3_adj_strength (<-1.5)                 → weight 0.25 (lowered threshold)
      3. Opponent is strong adjuster (opp q3_adj > 1.5)   → weight 0.15 (NEW)
      4. Young/inexperienced team (exp < 1.8)             → weight 0.15
      5. Lower seed (surprise early leads are suspicious) → weight 0.10
      6. History of blown leads (2+ this season)          → weight 0.15
    """
    config = BRANCH_CONFIG["EARLY_LEAD_MIRAGE"]
    subs = []

    fast_start = offensive_burst > 3.0
    subs.append(SubCondition(
        name="fast_start", met=fast_start, weight=0.20,
        value=offensive_burst,
        description=f"Offensive burst={offensive_burst:.1f} (threshold 3.0)"
    ))

    bad_adjustment = q3_adj_strength < -1.5  # lowered from -2.0
    subs.append(SubCondition(
        name="bad_halftime_adj", met=bad_adjustment, weight=0.25,
        value=q3_adj_strength,
        description=f"Q3 adj strength={q3_adj_strength:.1f} (threshold -1.5)"
    ))

    # NEW: Opponent is a strong halftime adjuster
    opp_strong_adjuster = opp_q3_adj > 1.5
    subs.append(SubCondition(
        name="opp_strong_adjuster", met=opp_strong_adjuster, weight=0.15,
        value=opp_q3_adj,
        description=f"Opponent Q3 adj={opp_q3_adj:.1f} (threshold >1.5 = gets better at halftime)"
    ))

    young = team_exp < 1.8
    subs.append(SubCondition(
        name="young_team", met=young, weight=0.15,
        value=team_exp,
        description=f"EXP={team_exp:.2f} (threshold 1.8)"
    ))

    subs.append(SubCondition(
        name="lower_seed", met=is_lower_seed, weight=0.10,
        description="Lower seed (early leads are suspicious)"
    ))

    blown_leads = historical_blown_leads >= 2
    subs.append(SubCondition(
        name="blown_lead_history", met=blown_leads, weight=0.15,
        value=historical_blown_leads,
        description=f"{historical_blown_leads} blown leads this season"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2 and (fast_start or bad_adjustment or opp_strong_adjuster)

    if not triggered:
        return BranchResult(branch_name="EARLY_LEAD_MIRAGE", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="EARLY_LEAD_MIRAGE", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="underdog", sub_conditions=subs,
        explanation=f"MIRAGE: starts fast, fades late → {shift*100:.1f}% shift"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 7: REBOUNDING_MISMATCH
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_rebounding_mismatch(team_name: str,
                                  orb_pct_diff: float,
                                  drb_pct_diff: float,
                                  total_rbm_diff: float,
                                  game_is_competitive: bool = True,
                                  team_has_physical_identity: bool = False) -> BranchResult:
    """Evaluate REBOUNDING_MISMATCH — dominant board work in tight games.

    Sub-conditions:
      1. ORB% diff > 0.05 (5%+ more offensive boards)    → weight 0.30
      2. DRB% diff > 0.04                                → weight 0.15
      3. Total RBM diff in top quartile (>0.08)           → weight 0.25
      4. Game is competitive (within 35-65% prob range)   → weight 0.15
      5. Team identity built on physicality               → weight 0.15
    """
    config = BRANCH_CONFIG["REBOUNDING_MISMATCH"]
    subs = []

    orb_edge = orb_pct_diff > 0.05
    subs.append(SubCondition(
        name="orb_edge", met=orb_edge, weight=0.30,
        value=orb_pct_diff,
        description=f"ORB% edge={orb_pct_diff*100:.1f}% (threshold 5%)"
    ))

    drb_edge = drb_pct_diff > 0.04
    subs.append(SubCondition(
        name="drb_edge", met=drb_edge, weight=0.15,
        value=drb_pct_diff,
        description=f"DRB% edge={drb_pct_diff*100:.1f}%"
    ))

    total_edge = total_rbm_diff > 0.08
    subs.append(SubCondition(
        name="total_rbm_edge", met=total_edge, weight=0.25,
        value=total_rbm_diff,
        description=f"Total RBM edge={total_rbm_diff:.3f} (top quartile)"
    ))

    subs.append(SubCondition(
        name="competitive_game", met=game_is_competitive, weight=0.15,
        description="Game is competitive (35-65% probability)"
    ))

    subs.append(SubCondition(
        name="physical_identity", met=team_has_physical_identity, weight=0.15,
        description="Team identity built on physicality/rebounding"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2 and (orb_edge or total_edge)  # need actual rebound edge

    if not triggered:
        return BranchResult(branch_name="REBOUNDING_MISMATCH", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="REBOUNDING_MISMATCH", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="rebounder", sub_conditions=subs,
        explanation=f"BOARDS: {n_met} rebounding edges → {shift*100:.1f}% toward rebounder"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 8: CONFERENCE_PHYSICALITY
# ─────────────────────────────────────────────────────────────────────────────

# Conferences ranked by physical play style (subjective but data-informed)
PHYSICAL_CONFERENCES = {"Big 12": 5, "SEC": 4, "Big East": 3, "ACC": 2}
FINESSE_CONFERENCES = {"Big Ten": 1, "WCC": 1, "Mountain West": 1}

def evaluate_conference_physicality(team_name: str,
                                     team_conference: str,
                                     opp_conference: str,
                                     conf_tourney_day1_record: Optional[Tuple[int, int]] = None,
                                     opp_sos_rank: int = 50,
                                     opp_q1_wins: int = 5) -> BranchResult:
    """Evaluate CONFERENCE_PHYSICALITY — physical style + stat inflation.

    Enhanced to also detect weak-conference stat inflation (Gonzaga/WCC pattern).
    When a team from a weak conference (high SOS rank, few Q1 wins) faces a
    physical/power conference team, their regular-season stats overstate ability.

    Sub-conditions:
      1. Team from top physical conference                → weight 0.20
      2. Opponent from finesse/weak conference            → weight 0.20
      3. Team's conference has strong Day 1 record        → weight 0.15
      4. Physicality gap ≥ 3 tiers                        → weight 0.15
      5. Opponent has weak SOS (rank > 80)                → weight 0.15 (NEW)
      6. Opponent has few Q1 wins (< 4)                   → weight 0.15 (NEW)
    """
    config = BRANCH_CONFIG["CONFERENCE_PHYSICALITY"]
    subs = []

    team_phys = PHYSICAL_CONFERENCES.get(team_conference, 0)
    opp_phys = PHYSICAL_CONFERENCES.get(opp_conference, 0)
    opp_finesse = FINESSE_CONFERENCES.get(opp_conference, 0)

    is_physical = team_phys >= 3
    subs.append(SubCondition(
        name="physical_conference", met=is_physical, weight=0.20,
        value=team_phys,
        description=f"{team_conference} physicality score={team_phys}"
    ))

    opp_is_finesse = opp_finesse >= 1
    subs.append(SubCondition(
        name="opp_finesse_conference", met=opp_is_finesse, weight=0.20,
        value=opp_finesse,
        description=f"{opp_conference} is finesse/weak-style"
    ))

    strong_conf_day1 = False
    if conf_tourney_day1_record:
        w, l = conf_tourney_day1_record
        if w + l > 0 and w / (w + l) >= 0.70:
            strong_conf_day1 = True
    subs.append(SubCondition(
        name="strong_conf_day1", met=strong_conf_day1, weight=0.15,
        description=f"Conference Day 1 record: {conf_tourney_day1_record}"
    ))

    big_gap = team_phys - max(opp_phys, opp_finesse) >= 3
    subs.append(SubCondition(
        name="physicality_gap", met=big_gap, weight=0.15,
        description=f"Physicality gap: {team_phys} vs {opp_phys}"
    ))

    # NEW: Opponent has weak SOS (inflated stats from weak schedule)
    weak_opp_sos = opp_sos_rank > 80
    subs.append(SubCondition(
        name="weak_opp_sos", met=weak_opp_sos, weight=0.15,
        value=opp_sos_rank,
        description=f"Opponent SOS rank={opp_sos_rank} (threshold >80 = inflated stats)"
    ))

    # NEW: Opponent has few quality wins
    few_q1 = opp_q1_wins < 4
    subs.append(SubCondition(
        name="opp_few_q1_wins", met=few_q1, weight=0.15,
        value=opp_q1_wins,
        description=f"Opponent Q1 wins={opp_q1_wins} (threshold <4)"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(branch_name="CONFERENCE_PHYSICALITY", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="CONFERENCE_PHYSICALITY", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="physical_team", sub_conditions=subs,
        explanation=f"PHYSICALITY: {team_conference} vs {opp_conference} → {shift*100:.1f}% (includes SOS inflation check)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 9: YOUTH_UNDER_PRESSURE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_youth_under_pressure(team_name: str,
                                   team_exp: float,
                                   opp_exp: float,
                                   first_tourney_game: bool = True,
                                   freshmen_starters: int = 0,
                                   round_name: str = "R64") -> BranchResult:
    """Evaluate YOUTH_UNDER_PRESSURE — inexperience in March.

    Sub-conditions:
      1. Team EXP < 1.5                                 → weight 0.30
      2. Opponent EXP gap > 0.5 years                    → weight 0.20
      3. First tournament game (or first for 3+ starters)→ weight 0.25
      4. 3+ freshmen starters                            → weight 0.15
      5. Round is R64 (nerves strongest in first game)   → weight 0.10
    """
    config = BRANCH_CONFIG["YOUTH_UNDER_PRESSURE"]
    subs = []

    very_young = team_exp < 1.5
    subs.append(SubCondition(
        name="very_young", met=very_young, weight=0.30,
        value=team_exp,
        description=f"EXP={team_exp:.2f} (threshold 1.5)"
    ))

    exp_gap = opp_exp - team_exp > 0.5
    subs.append(SubCondition(
        name="experience_gap", met=exp_gap, weight=0.20,
        value=opp_exp - team_exp,
        description=f"Opponent {opp_exp - team_exp:.2f} years more experience"
    ))

    subs.append(SubCondition(
        name="first_tourney", met=first_tourney_game, weight=0.25,
        description="First tournament appearance for key players"
    ))

    many_freshmen = freshmen_starters >= 3
    subs.append(SubCondition(
        name="freshmen_starters", met=many_freshmen, weight=0.15,
        value=freshmen_starters,
        description=f"{freshmen_starters} freshmen starters"
    ))

    # Nerves fade after first game — strongest in R64
    r64_nerves = round_name == "R64"
    subs.append(SubCondition(
        name="r64_first_game_nerves", met=r64_nerves, weight=0.10,
        description=f"Round {round_name} (R64 = max nerves)"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2 and very_young

    if not triggered:
        return BranchResult(branch_name="YOUTH_UNDER_PRESSURE", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="YOUTH_UNDER_PRESSURE", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="underdog", sub_conditions=subs,
        explanation=f"YOUTH: inexperience under pressure → {shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 10: FORM_COLLAPSE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_form_collapse(team_name: str,
                            losses_last_6: int,
                            conf_tourney_early_exit: bool = False,
                            form_trend: float = 0.5,
                            momentum_score: float = 0.5) -> BranchResult:
    """Evaluate FORM_COLLAPSE — team on a skid entering the tournament.

    Sub-conditions:
      1. Lost 3+ of last 6 games                         → weight 0.30
      2. Conference tourney early exit                    → weight 0.25
      3. Form trend in bottom quartile (<0.25)            → weight 0.25
      4. Momentum score below 0.35                        → weight 0.20
    """
    config = BRANCH_CONFIG["FORM_COLLAPSE"]
    subs = []

    bad_streak = losses_last_6 >= 3
    subs.append(SubCondition(
        name="losing_streak", met=bad_streak, weight=0.30,
        value=losses_last_6,
        description=f"Lost {losses_last_6} of last 6 games"
    ))

    subs.append(SubCondition(
        name="conf_tourney_exit", met=conf_tourney_early_exit, weight=0.25,
        description="Early conference tournament exit"
    ))

    cold_form = form_trend < 0.25
    subs.append(SubCondition(
        name="cold_form_trend", met=cold_form, weight=0.25,
        value=form_trend,
        description=f"Form trend={form_trend:.2f} (threshold 0.25)"
    ))

    low_momentum = momentum_score < 0.35
    subs.append(SubCondition(
        name="low_momentum", met=low_momentum, weight=0.20,
        value=momentum_score,
        description=f"Momentum={momentum_score:.2f} (threshold 0.35)"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(branch_name="FORM_COLLAPSE", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="FORM_COLLAPSE", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="underdog", sub_conditions=subs,
        explanation=f"FORM: {losses_last_6} recent losses → {shift*100:.1f}% shift"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 11: COACH_TOURNAMENT_DNA
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_coach_tournament_dna(team_name: str,
                                   coach_tourney_wins: int,
                                   coach_elite8_plus: int,
                                   coach_first_tourney: bool = False,
                                   opp_coach_tourney_wins: int = 0,
                                   game_is_close: bool = False) -> BranchResult:
    """Evaluate COACH_TOURNAMENT_DNA — coaching adjustments in tight games.

    Only fires in tight games where coaching adjustments (halftime, timeout
    plays) are decisive.

    Sub-conditions:
      1. Coach has 5+ tourney wins                       → weight 0.25
      2. Coach has Elite 8+ appearances                  → weight 0.25
      3. Opponent coach is first-time tourney coach      → weight 0.20
      4. Game is projected close (35-65% range)          → weight 0.15
      5. Big experience gap (10+ more wins than opp)     → weight 0.15
    """
    config = BRANCH_CONFIG["COACH_TOURNAMENT_DNA"]
    subs = []

    experienced = coach_tourney_wins >= 5
    subs.append(SubCondition(
        name="experienced_coach", met=experienced, weight=0.25,
        value=coach_tourney_wins,
        description=f"Coach has {coach_tourney_wins} tourney wins (threshold 5)"
    ))

    deep_runs = coach_elite8_plus >= 1
    subs.append(SubCondition(
        name="deep_run_coach", met=deep_runs, weight=0.25,
        value=coach_elite8_plus,
        description=f"Coach has {coach_elite8_plus} Elite 8+ appearances"
    ))

    opp_rookie = coach_first_tourney
    subs.append(SubCondition(
        name="opp_first_tourney_coach", met=opp_rookie, weight=0.20,
        description="Opponent's coach is a first-time tourney coach"
    ))

    subs.append(SubCondition(
        name="close_game", met=game_is_close, weight=0.15,
        description="Game projected to be close"
    ))

    big_gap = coach_tourney_wins - opp_coach_tourney_wins >= 10
    subs.append(SubCondition(
        name="coaching_gap", met=big_gap, weight=0.15,
        value=coach_tourney_wins - opp_coach_tourney_wins,
        description=f"Coaching tourney win gap: {coach_tourney_wins - opp_coach_tourney_wins}"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2 and (experienced or deep_runs)

    if not triggered:
        return BranchResult(branch_name="COACH_TOURNAMENT_DNA", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="COACH_TOURNAMENT_DNA", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="experienced_coach", sub_conditions=subs,
        explanation=f"COACH DNA: {coach_tourney_wins} wins, {coach_elite8_plus} deep runs → {shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 12: REVENGE_MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_revenge_momentum(team_name: str,
                               won_prev_round: bool,
                               prev_win_was_comeback: bool = False,
                               prev_win_margin: float = 0.0,
                               beat_higher_seed: bool = False) -> BranchResult:
    """Evaluate REVENGE_MOMENTUM — psychological boost from dramatic wins.

    Sub-conditions:
      1. Won previous round                              → weight 0.20
      2. Previous win was a comeback (trailed by 10+)    → weight 0.30
      3. Win was close (margin ≤ 5)                      → weight 0.15
      4. Beat a higher seed                              → weight 0.20
      5. Comeback was dramatic (15+ point deficit)        → weight 0.15
    """
    config = BRANCH_CONFIG["REVENGE_MOMENTUM"]
    subs = []

    subs.append(SubCondition(
        name="won_prev_round", met=won_prev_round, weight=0.20,
        description="Won in previous round"
    ))

    subs.append(SubCondition(
        name="comeback_win", met=prev_win_was_comeback, weight=0.30,
        description="Previous win was a comeback from 10+ down"
    ))

    close_win = 0 < prev_win_margin <= 5
    subs.append(SubCondition(
        name="close_win", met=close_win, weight=0.15,
        value=prev_win_margin,
        description=f"Won by {prev_win_margin:.0f} pts"
    ))

    subs.append(SubCondition(
        name="beat_higher_seed", met=beat_higher_seed, weight=0.20,
        description="Upset a higher seed"
    ))

    dramatic = prev_win_was_comeback and prev_win_margin <= 8
    subs.append(SubCondition(
        name="dramatic_comeback", met=dramatic, weight=0.15,
        description="Dramatic comeback (tight margin after big deficit)"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = won_prev_round and n_met >= 3  # need the win + extras

    if not triggered:
        return BranchResult(branch_name="REVENGE_MOMENTUM", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="REVENGE_MOMENTUM", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="momentum_team", sub_conditions=subs,
        explanation=f"MOMENTUM: dramatic win → {shift*100:.1f}% psychological boost"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 13: THREE_PT_VARIANCE_BOMB
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_three_pt_variance_bomb(team_name: str,
                                     three_pt_share_of_offense: float,
                                     three_pt_pct: float,
                                     three_pt_std: float = 0.0,
                                     opp_perimeter_defense_elite: bool = False,
                                     is_underdog: bool = True) -> BranchResult:
    """Evaluate THREE_PT_VARIANCE_BOMB — 3PT-dependent team widens distribution.

    Sub-conditions:
      1. 3PT attempts > 40% of FGA                       → weight 0.25
      2. 3PT% > 37% (good enough to get hot)             → weight 0.25
      3. High 3PT variance (std > 0.06)                  → weight 0.20
      4. Opponent does NOT have elite perimeter D         → weight 0.15
      5. Team is an underdog (variance helps underdogs)   → weight 0.15
    """
    config = BRANCH_CONFIG["THREE_PT_VARIANCE_BOMB"]
    subs = []

    heavy_3pt = three_pt_share_of_offense > 0.40
    subs.append(SubCondition(
        name="heavy_3pt_offense", met=heavy_3pt, weight=0.25,
        value=three_pt_share_of_offense,
        description=f"3PA/FGA={three_pt_share_of_offense*100:.0f}% (threshold 40%)"
    ))

    good_shooting = three_pt_pct > 0.37
    subs.append(SubCondition(
        name="good_3pt_shooting", met=good_shooting, weight=0.25,
        value=three_pt_pct,
        description=f"3PT%={three_pt_pct*100:.1f}% (threshold 37%)"
    ))

    high_variance = three_pt_std > 0.06
    subs.append(SubCondition(
        name="high_3pt_variance", met=high_variance, weight=0.20,
        value=three_pt_std,
        description=f"3PT std={three_pt_std:.3f} (threshold 0.06)"
    ))

    # NOT facing elite D → the hot night is more likely to happen
    not_elite_d = not opp_perimeter_defense_elite
    subs.append(SubCondition(
        name="beatable_perimeter_d", met=not_elite_d, weight=0.15,
        description="Opponent does NOT have elite perimeter defense"
    ))

    subs.append(SubCondition(
        name="underdog_status", met=is_underdog, weight=0.15,
        description="Variance helps underdogs (asymmetric payoff)"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2 and (heavy_3pt or good_shooting)

    if not triggered:
        return BranchResult(branch_name="THREE_PT_VARIANCE_BOMB", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="THREE_PT_VARIANCE_BOMB", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="variance", sub_conditions=subs,
        explanation=f"3PT BOMB: hot-shooting variance → widen ±{shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 14: DEPTH_ADVANTAGE — later rounds compound bench depth
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_depth_advantage(team_name: str,
                              team_bds: float,
                              opp_bds: float,
                              round_name: str = "R64",
                              team_rotation: int = 9,
                              opp_foul_trouble_rate: float = 0.0) -> BranchResult:
    """Evaluate DEPTH_ADVANTAGE — bench depth compounds in later rounds.

    Sub-conditions:
      1. Team BDS > 0.60 (deep bench)                   → weight 0.25
      2. Opponent BDS < 0.40 (shallow bench)             → weight 0.25
      3. BDS gap > 0.25                                  → weight 0.20
      4. Round is R32 or later (depth matters more)      → weight 0.15
      5. Opponent has foul trouble risk (>0.5)            → weight 0.15
    """
    config = BRANCH_CONFIG["DEPTH_ADVANTAGE"]
    subs = []

    deep = team_bds > 0.60
    subs.append(SubCondition(
        name="deep_bench", met=deep, weight=0.25,
        value=team_bds,
        description=f"BDS={team_bds:.2f} (threshold 0.60)"
    ))

    opp_shallow = opp_bds < 0.40
    subs.append(SubCondition(
        name="opp_shallow_bench", met=opp_shallow, weight=0.25,
        value=opp_bds,
        description=f"Opponent BDS={opp_bds:.2f} (threshold 0.40)"
    ))

    big_gap = team_bds - opp_bds > 0.25
    subs.append(SubCondition(
        name="bds_gap", met=big_gap, weight=0.20,
        value=team_bds - opp_bds,
        description=f"BDS gap={team_bds - opp_bds:.2f} (threshold 0.25)"
    ))

    later_round = round_name in ("R32", "S16", "E8", "F4", "NCG")
    round_mult = ROUND_FATIGUE_MULTIPLIER.get(round_name, 1.0)
    subs.append(SubCondition(
        name="later_round", met=later_round,
        weight=0.15 * round_mult,
        description=f"Round {round_name} (depth multiplier {round_mult:.1f}x)"
    ))

    foul_risk = opp_foul_trouble_rate > 0.5
    subs.append(SubCondition(
        name="opp_foul_trouble", met=foul_risk, weight=0.15,
        value=opp_foul_trouble_rate,
        description=f"Opponent foul trouble rate={opp_foul_trouble_rate:.2f}"
    ))

    n_met = sum(1 for sc in subs if sc.met)
    triggered = n_met >= 2

    if not triggered:
        return BranchResult(branch_name="DEPTH_ADVANTAGE", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="DEPTH_ADVANTAGE", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="deeper_team", sub_conditions=subs,
        explanation=f"DEPTH: bench edge compounds in {round_name} → {shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Branch 15: CRIPPLED_ROSTER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_crippled_roster(team_name: str,
                              has_crippled_roster: bool = False,
                              crippled_weeks_out: float = 0.0,
                              crippled_ppg_lost: float = 0.0,
                              team_offensive_burst: float = 0.0,
                              team_q3_adj: float = 0.0,
                              star_bpr_share: float = 0.0,
                              is_higher_seed: bool = False) -> BranchResult:
    """Branch 15: Long-term absence of top scorer → offensive identity collapse.

    This branch captures the Gonzaga/Huff pattern:
    - Star scorer out 4+ weeks (not just "game-time decision")
    - Team stats (adj_em, barthag) still reflect pre-injury peak
    - Offense has fundamentally changed: fewer options, more predictable
    - The longer the absence, the worse the "stat inflation" problem

    Different from ALPHA_VACUUM: that's about chemistry collapse after losing
    a star mid-tournament. CRIPPLED_ROSTER is about entering the tournament
    with an already-diminished team whose numbers lie.
    """
    config = BRANCH_CONFIG["CRIPPLED_ROSTER"]
    subs = []

    # Sub 1: Top-2 scorer actually out long-term
    subs.append(SubCondition(
        name="top_scorer_out_longterm",
        met=has_crippled_roster,
        weight=0.35,
        value=1.0 if has_crippled_roster else 0.0,
        description=f"Top scorer out {crippled_weeks_out:.0f}+ weeks" if has_crippled_roster else "No long-term star absence"
    ))

    # Sub 2: Extended absence (>6 weeks = deep stat inflation)
    deep_absence = crippled_weeks_out >= 6
    subs.append(SubCondition(
        name="deep_absence",
        met=deep_absence,
        weight=0.20,
        value=crippled_weeks_out,
        description=f"Out {crippled_weeks_out:.0f} weeks (threshold 6 = deep inflation)"
    ))

    # Sub 3: High PPG lost (the player they lost was their offensive engine)
    high_ppg_lost = crippled_ppg_lost >= 14.0
    subs.append(SubCondition(
        name="high_ppg_lost",
        met=high_ppg_lost,
        weight=0.20,
        value=crippled_ppg_lost,
        description=f"Lost {crippled_ppg_lost:.1f} PPG scorer (threshold 14)"
    ))

    # Sub 4: High BPR share (the lost player was THE system)
    high_share = star_bpr_share >= 0.25
    subs.append(SubCondition(
        name="star_was_system",
        met=high_share,
        weight=0.15,
        value=star_bpr_share,
        description=f"BPR share={star_bpr_share:.0%} (threshold 25%)"
    ))

    # Sub 5: Higher seed (stat inflation more dangerous for favorites)
    subs.append(SubCondition(
        name="higher_seed_inflated",
        met=is_higher_seed,
        weight=0.10,
        value=1.0 if is_higher_seed else 0.0,
        description="Higher seed → inflated stats carry more risk"
    ))

    triggered = has_crippled_roster  # Must have the core condition
    if not triggered:
        return BranchResult(branch_name="CRIPPLED_ROSTER", triggered=False,
                            sub_conditions=subs)

    sev = severity_curve(subs, tipping_point=3)
    shift = compute_shift(sev, config["max_shift"])

    return BranchResult(
        branch_name="CRIPPLED_ROSTER", triggered=True,
        severity=sev, shift=shift, max_shift=config["max_shift"],
        direction="underdog", sub_conditions=subs,
        explanation=f"CRIPPLED: {team_name} lost top scorer {crippled_weeks_out:.0f} weeks ago, stats inflated → {shift*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FUNCTION: Evaluate all branches for a matchup
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchupBranchContext:
    """All the data needed to evaluate branches for a matchup.

    Callers populate what they have. Missing data → branches that need
    it simply won't trigger (graceful degradation).
    """
    # Team identification
    team_a_name: str = ""
    team_b_name: str = ""
    round_name: str = "R64"

    # STAR_ISOLATION inputs (for each team)
    sii_a: float = 0.0
    sii_b: float = 0.0
    top_player_minutes_a: float = 0.0
    top_player_minutes_b: float = 0.0
    top_player_usage_a: float = 0.0
    top_player_usage_b: float = 0.0
    second_option_injured_a: bool = False
    second_option_injured_b: bool = False
    record_post_injury_a: Optional[Tuple[int, int]] = None
    record_post_injury_b: Optional[Tuple[int, int]] = None
    recent_form_losses_a: int = 0
    recent_form_losses_b: int = 0

    # ALPHA_VACUUM inputs
    star_lost_a: bool = False
    star_lost_b: bool = False
    star_bpr_share_a: float = 0.0
    star_bpr_share_b: float = 0.0
    ast_rate_change_a: Optional[float] = None
    ast_rate_change_b: Optional[float] = None
    record_after_loss_a: Optional[Tuple[int, int]] = None
    record_after_loss_b: Optional[Tuple[int, int]] = None
    team_exp_a: float = 2.0
    team_exp_b: float = 2.0
    star_ppg_a: float = 0.0
    star_ppg_b: float = 0.0

    # UNICORN_PLAYER inputs
    unicorn_conf_leader_a: bool = False
    unicorn_conf_leader_b: bool = False
    unicorn_nat_top10_a: bool = False
    unicorn_nat_top10_b: bool = False
    unicorn_3pt_pct_a: float = 0.0
    unicorn_3pt_pct_b: float = 0.0
    unicorn_career_3s_a: int = 0
    unicorn_career_3s_b: int = 0
    unicorn_ppg_a: float = 0.0
    unicorn_ppg_b: float = 0.0

    # LOCKDOWN_DEFENDER inputs
    lockdown_switchable_a: bool = False
    lockdown_switchable_b: bool = False
    lockdown_dbpr_a: float = 0.0
    lockdown_dbpr_b: float = 0.0
    team_opp_fg_pct_a: float = 0.50
    team_opp_fg_pct_b: float = 0.50

    # FATIGUE_TRAP inputs
    rotation_size_a: int = 9
    rotation_size_b: int = 9
    bench_depth_a: float = 0.5
    bench_depth_b: float = 0.5

    # EARLY_LEAD_MIRAGE inputs
    offensive_burst_a: float = 0.0
    offensive_burst_b: float = 0.0
    q3_adj_a: float = 0.0
    q3_adj_b: float = 0.0
    blown_leads_a: int = 0
    blown_leads_b: int = 0

    # REBOUNDING_MISMATCH — computed from team stats (orb_pct, drb_pct, rbm)
    orb_pct_a: float = 0.0
    orb_pct_b: float = 0.0
    drb_pct_a: float = 0.0
    drb_pct_b: float = 0.0
    rbm_a: float = 0.0
    rbm_b: float = 0.0
    physical_identity_a: bool = False
    physical_identity_b: bool = False

    # CONFERENCE_PHYSICALITY
    conference_a: str = ""
    conference_b: str = ""
    conf_day1_record_a: Optional[Tuple[int, int]] = None
    conf_day1_record_b: Optional[Tuple[int, int]] = None

    # YOUTH_UNDER_PRESSURE
    first_tourney_a: bool = False
    first_tourney_b: bool = False
    freshmen_starters_a: int = 0
    freshmen_starters_b: int = 0

    # FORM_COLLAPSE
    losses_last_6_a: int = 0
    losses_last_6_b: int = 0
    conf_tourney_early_exit_a: bool = False
    conf_tourney_early_exit_b: bool = False
    form_trend_a: float = 0.5
    form_trend_b: float = 0.5
    momentum_a: float = 0.5
    momentum_b: float = 0.5

    # COACH_TOURNAMENT_DNA
    coach_wins_a: int = 0
    coach_wins_b: int = 0
    coach_e8_a: int = 0
    coach_e8_b: int = 0
    coach_first_tourney_a: bool = False
    coach_first_tourney_b: bool = False

    # REVENGE_MOMENTUM
    won_prev_round_a: bool = False
    won_prev_round_b: bool = False
    prev_comeback_a: bool = False
    prev_comeback_b: bool = False
    prev_margin_a: float = 0.0
    prev_margin_b: float = 0.0
    beat_higher_seed_a: bool = False
    beat_higher_seed_b: bool = False

    # THREE_PT_VARIANCE_BOMB
    three_pt_share_a: float = 0.0
    three_pt_share_b: float = 0.0
    three_pt_pct_a: float = 0.0
    three_pt_pct_b: float = 0.0
    three_pt_std_a: float = 0.0
    three_pt_std_b: float = 0.0

    # DEPTH_ADVANTAGE
    foul_trouble_rate_a: float = 0.0
    foul_trouble_rate_b: float = 0.0

    # Seed info (for underdog detection)
    seed_a: int = 1
    seed_b: int = 16

    # CRIPPLED_ROSTER inputs
    crippled_roster_a: bool = False
    crippled_roster_b: bool = False
    crippled_weeks_out_a: float = 0.0
    crippled_weeks_out_b: float = 0.0
    crippled_ppg_lost_a: float = 0.0
    crippled_ppg_lost_b: float = 0.0

    # CONFERENCE_PHYSICALITY enhancement: SOS and quality wins
    sos_rank_a: int = 50
    sos_rank_b: int = 50
    q1_wins_a: int = 5
    q1_wins_b: int = 5

    # Base probability from ROOT (for competitive-game detection)
    p_base: float = 0.5


def evaluate_all_branches(ctx: 'MatchupBranchContext') -> Tuple[BranchEngineResult, BranchEngineResult]:
    """Run ALL 14 branches for both teams in a matchup.

    Returns BranchEngineResults for team_a and team_b.
    The caller can then use compound_branches() to get final probability.
    """
    is_competitive = 0.35 <= ctx.p_base <= 0.65
    a_is_underdog = ctx.seed_a > ctx.seed_b
    b_is_underdog = ctx.seed_b > ctx.seed_a

    # ── Evaluate all branches for TEAM A ──────────────────────────────────
    branches_a: List[BranchResult] = []

    # 1. STAR_ISOLATION
    branches_a.append(evaluate_star_isolation(
        ctx.team_a_name, ctx.sii_a, ctx.top_player_minutes_a,
        ctx.top_player_usage_a, ctx.second_option_injured_a,
        ctx.record_post_injury_a, ctx.recent_form_losses_a
    ))

    # 2. ALPHA_VACUUM
    branches_a.append(evaluate_alpha_vacuum(
        ctx.team_a_name, ctx.star_lost_a, ctx.star_bpr_share_a,
        ctx.ast_rate_change_a, ctx.record_after_loss_a,
        ctx.team_exp_a, ctx.star_ppg_a
    ))

    # 3. UNICORN_PLAYER (for team A against team B)
    branches_a.append(evaluate_unicorn_player(
        ctx.team_a_name, ctx.unicorn_conf_leader_a, ctx.unicorn_nat_top10_a,
        ctx.unicorn_3pt_pct_a, ctx.unicorn_career_3s_a,
        ctx.unicorn_ppg_a, is_underdog=a_is_underdog
    ))

    # 4. LOCKDOWN_DEFENDER (team A defending against team B's star)
    branches_a.append(evaluate_lockdown_defender(
        ctx.team_a_name, ctx.lockdown_switchable_a, ctx.lockdown_dbpr_a,
        opp_top_scorer_ppg=ctx.star_ppg_b,
        team_opp_fg_pct=ctx.team_opp_fg_pct_a,
        opp_star_isolation=ctx.sii_b
    ))

    # 5. FATIGUE_TRAP
    branches_a.append(evaluate_fatigue_trap(
        ctx.team_a_name, ctx.rotation_size_a, ctx.rotation_size_b,
        ctx.top_player_minutes_a, ctx.bench_depth_a, ctx.round_name
    ))

    # 6. EARLY_LEAD_MIRAGE
    branches_a.append(evaluate_early_lead_mirage(
        ctx.team_a_name, ctx.offensive_burst_a, ctx.q3_adj_a,
        ctx.team_exp_a, is_lower_seed=a_is_underdog,
        historical_blown_leads=ctx.blown_leads_a,
        opp_q3_adj=ctx.q3_adj_b
    ))

    # 7. REBOUNDING_MISMATCH (A's edge over B)
    orb_diff_a = ctx.orb_pct_a - ctx.orb_pct_b
    drb_diff_a = ctx.drb_pct_a - ctx.drb_pct_b
    rbm_diff_a = ctx.rbm_a - ctx.rbm_b
    branches_a.append(evaluate_rebounding_mismatch(
        ctx.team_a_name, orb_diff_a, drb_diff_a, rbm_diff_a,
        game_is_competitive=is_competitive,
        team_has_physical_identity=ctx.physical_identity_a
    ))

    # 8. CONFERENCE_PHYSICALITY
    branches_a.append(evaluate_conference_physicality(
        ctx.team_a_name, ctx.conference_a, ctx.conference_b,
        ctx.conf_day1_record_a,
        opp_sos_rank=ctx.sos_rank_b, opp_q1_wins=ctx.q1_wins_b
    ))

    # 9. YOUTH_UNDER_PRESSURE
    branches_a.append(evaluate_youth_under_pressure(
        ctx.team_a_name, ctx.team_exp_a, ctx.team_exp_b,
        ctx.first_tourney_a, ctx.freshmen_starters_a, ctx.round_name
    ))

    # 10. FORM_COLLAPSE
    branches_a.append(evaluate_form_collapse(
        ctx.team_a_name, ctx.losses_last_6_a,
        ctx.conf_tourney_early_exit_a, ctx.form_trend_a, ctx.momentum_a
    ))

    # 11. COACH_TOURNAMENT_DNA
    branches_a.append(evaluate_coach_tournament_dna(
        ctx.team_a_name, ctx.coach_wins_a, ctx.coach_e8_a,
        ctx.coach_first_tourney_b,  # opp coach is first-timer?
        ctx.coach_wins_b, game_is_close=is_competitive
    ))

    # 12. REVENGE_MOMENTUM
    branches_a.append(evaluate_revenge_momentum(
        ctx.team_a_name, ctx.won_prev_round_a, ctx.prev_comeback_a,
        ctx.prev_margin_a, ctx.beat_higher_seed_a
    ))

    # 13. THREE_PT_VARIANCE_BOMB
    opp_elite_perim_a = ctx.team_opp_fg_pct_b < 0.31  # team B has elite perimeter D
    branches_a.append(evaluate_three_pt_variance_bomb(
        ctx.team_a_name, ctx.three_pt_share_a, ctx.three_pt_pct_a,
        ctx.three_pt_std_a, opp_perimeter_defense_elite=opp_elite_perim_a,
        is_underdog=a_is_underdog
    ))

    # 14. DEPTH_ADVANTAGE
    branches_a.append(evaluate_depth_advantage(
        ctx.team_a_name, ctx.bench_depth_a, ctx.bench_depth_b,
        ctx.round_name, ctx.rotation_size_a, ctx.foul_trouble_rate_b
    ))

    # 15. CRIPPLED_ROSTER
    branches_a.append(evaluate_crippled_roster(
        ctx.team_a_name,
        has_crippled_roster=ctx.crippled_roster_a,
        crippled_weeks_out=ctx.crippled_weeks_out_a,
        crippled_ppg_lost=ctx.crippled_ppg_lost_a,
        star_bpr_share=ctx.star_bpr_share_a,
        is_higher_seed=ctx.seed_a < ctx.seed_b,
    ))

    # ── Evaluate all branches for TEAM B ──────────────────────────────────
    branches_b: List[BranchResult] = []

    branches_b.append(evaluate_star_isolation(
        ctx.team_b_name, ctx.sii_b, ctx.top_player_minutes_b,
        ctx.top_player_usage_b, ctx.second_option_injured_b,
        ctx.record_post_injury_b, ctx.recent_form_losses_b
    ))

    branches_b.append(evaluate_alpha_vacuum(
        ctx.team_b_name, ctx.star_lost_b, ctx.star_bpr_share_b,
        ctx.ast_rate_change_b, ctx.record_after_loss_b,
        ctx.team_exp_b, ctx.star_ppg_b
    ))

    branches_b.append(evaluate_unicorn_player(
        ctx.team_b_name, ctx.unicorn_conf_leader_b, ctx.unicorn_nat_top10_b,
        ctx.unicorn_3pt_pct_b, ctx.unicorn_career_3s_b,
        ctx.unicorn_ppg_b, is_underdog=b_is_underdog
    ))

    branches_b.append(evaluate_lockdown_defender(
        ctx.team_b_name, ctx.lockdown_switchable_b, ctx.lockdown_dbpr_b,
        opp_top_scorer_ppg=ctx.star_ppg_a,
        team_opp_fg_pct=ctx.team_opp_fg_pct_b,
        opp_star_isolation=ctx.sii_a
    ))

    branches_b.append(evaluate_fatigue_trap(
        ctx.team_b_name, ctx.rotation_size_b, ctx.rotation_size_a,
        ctx.top_player_minutes_b, ctx.bench_depth_b, ctx.round_name
    ))

    branches_b.append(evaluate_early_lead_mirage(
        ctx.team_b_name, ctx.offensive_burst_b, ctx.q3_adj_b,
        ctx.team_exp_b, is_lower_seed=b_is_underdog,
        historical_blown_leads=ctx.blown_leads_b,
        opp_q3_adj=ctx.q3_adj_a
    ))

    orb_diff_b = ctx.orb_pct_b - ctx.orb_pct_a
    drb_diff_b = ctx.drb_pct_b - ctx.drb_pct_a
    rbm_diff_b = ctx.rbm_b - ctx.rbm_a
    branches_b.append(evaluate_rebounding_mismatch(
        ctx.team_b_name, orb_diff_b, drb_diff_b, rbm_diff_b,
        game_is_competitive=is_competitive,
        team_has_physical_identity=ctx.physical_identity_b
    ))

    branches_b.append(evaluate_conference_physicality(
        ctx.team_b_name, ctx.conference_b, ctx.conference_a,
        ctx.conf_day1_record_b,
        opp_sos_rank=ctx.sos_rank_a, opp_q1_wins=ctx.q1_wins_a
    ))

    branches_b.append(evaluate_youth_under_pressure(
        ctx.team_b_name, ctx.team_exp_b, ctx.team_exp_a,
        ctx.first_tourney_b, ctx.freshmen_starters_b, ctx.round_name
    ))

    branches_b.append(evaluate_form_collapse(
        ctx.team_b_name, ctx.losses_last_6_b,
        ctx.conf_tourney_early_exit_b, ctx.form_trend_b, ctx.momentum_b
    ))

    branches_b.append(evaluate_coach_tournament_dna(
        ctx.team_b_name, ctx.coach_wins_b, ctx.coach_e8_b,
        ctx.coach_first_tourney_a,
        ctx.coach_wins_a, game_is_close=is_competitive
    ))

    branches_b.append(evaluate_revenge_momentum(
        ctx.team_b_name, ctx.won_prev_round_b, ctx.prev_comeback_b,
        ctx.prev_margin_b, ctx.beat_higher_seed_b
    ))

    opp_elite_perim_b = ctx.team_opp_fg_pct_a < 0.31
    branches_b.append(evaluate_three_pt_variance_bomb(
        ctx.team_b_name, ctx.three_pt_share_b, ctx.three_pt_pct_b,
        ctx.three_pt_std_b, opp_perimeter_defense_elite=opp_elite_perim_b,
        is_underdog=b_is_underdog
    ))

    branches_b.append(evaluate_depth_advantage(
        ctx.team_b_name, ctx.bench_depth_b, ctx.bench_depth_a,
        ctx.round_name, ctx.rotation_size_b, ctx.foul_trouble_rate_a
    ))

    # 15. CRIPPLED_ROSTER
    branches_b.append(evaluate_crippled_roster(
        ctx.team_b_name,
        has_crippled_roster=ctx.crippled_roster_b,
        crippled_weeks_out=ctx.crippled_weeks_out_b,
        crippled_ppg_lost=ctx.crippled_ppg_lost_b,
        star_bpr_share=ctx.star_bpr_share_b,
        is_higher_seed=ctx.seed_b < ctx.seed_a,
    ))

    # ── Resolve: directional branches modify one team's survival ──────────
    # Branches with direction "underdog" hurt the team they fired on.
    # Branches with direction "favorite" help the team.
    # Branches with "variance" widen the distribution.
    # Branches with special directions (rebounder, physical, etc.) are
    # oriented: if team_a has the rebounding edge, it HELPS team_a.

    # Collect shifts that hurt team_a's win probability
    hurt_a_shifts = []
    help_a_shifts = []
    variance_total = 0.0

    for br in branches_a:
        if not br.triggered:
            continue
        if br.direction == "variance":
            variance_total += br.shift
        elif br.direction == "underdog":
            # This branch hurts team_a (makes upset more likely)
            hurt_a_shifts.append(br.shift)
        elif br.direction in ("favorite", "rebounder", "physical_team",
                               "experienced_coach", "deeper_team",
                               "momentum_team"):
            # These branches HELP team_a
            help_a_shifts.append(br.shift)

    for br in branches_b:
        if not br.triggered:
            continue
        if br.direction == "variance":
            variance_total += br.shift
        elif br.direction == "underdog":
            # This branch hurts team_b → helps team_a
            help_a_shifts.append(br.shift)
        elif br.direction in ("favorite", "rebounder", "physical_team",
                               "experienced_coach", "deeper_team",
                               "momentum_team"):
            # These branches help team_b → hurt team_a
            hurt_a_shifts.append(br.shift)

    # Apply survival model
    survival = 1.0
    for s in hurt_a_shifts:
        survival *= (1.0 - s)
    boost = 1.0
    for s in help_a_shifts:
        boost *= (1.0 + s)

    p_final = ctx.p_base * survival * boost
    p_final = max(0.01, min(0.99, p_final))

    result_a = BranchEngineResult(
        team_name=ctx.team_a_name,
        p_base=ctx.p_base,
        p_final=p_final,
        total_shift=p_final - ctx.p_base,
        variance_widen=variance_total,
        branches_fired=[br for br in branches_a if br.triggered],
        survival_multiplier=survival * boost,
    )

    result_b = BranchEngineResult(
        team_name=ctx.team_b_name,
        p_base=1.0 - ctx.p_base,
        p_final=1.0 - p_final,
        total_shift=(1.0 - p_final) - (1.0 - ctx.p_base),
        variance_widen=variance_total,
        branches_fired=[br for br in branches_b if br.triggered],
        survival_multiplier=1.0 / (survival * boost) if survival * boost != 0 else 1.0,
    )

    return result_a, result_b


# ─────────────────────────────────────────────────────────────────────────────
# Branch Engine: Report generation
# ─────────────────────────────────────────────────────────────────────────────

def branch_engine_report(results_a: List[BranchResult],
                         results_b: List[BranchResult],
                         team_a: str, team_b: str,
                         p_base: float, p_final: float) -> str:
    """Generate human-readable report of branch evaluations."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"BRANCH ENGINE: {team_a} vs {team_b}")
    lines.append(f"ROOT probability: {team_a} {p_base*100:.1f}%")
    lines.append("=" * 70)

    for team_name, results in [(team_a, results_a), (team_b, results_b)]:
        fired = [r for r in results if r.triggered]
        if not fired:
            lines.append(f"\n  {team_name}: No branches triggered")
            continue

        lines.append(f"\n  {team_name}: {len(fired)} branches triggered")
        for br in fired:
            lines.append(f"    ├─ {br.branch_name} (severity {br.severity:.2f})")
            lines.append(f"    │  Shift: {br.shift*100:+.1f}% (max ±{br.max_shift*100:.0f}%)")
            lines.append(f"    │  {br.explanation}")
            for sc in br.sub_conditions:
                icon = "✓" if sc.met else "✗"
                lines.append(f"    │    {icon} {sc.name}: {sc.description} (w={sc.weight:.2f})")

    lines.append(f"\nFINAL: {team_a} {p_final*100:.1f}% | {team_b} {(1-p_final)*100:.1f}%")
    lines.append(f"SHIFT: {(p_final - p_base)*100:+.1f}% from root")
    lines.append("=" * 70)

    return "\n".join(lines)
