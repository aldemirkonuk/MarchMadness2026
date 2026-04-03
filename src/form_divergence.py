"""Form Divergence Detector — The final reality check in the pipeline.

Architecture:
  Sits AFTER ROOT + Branch Engine + Round Survival Filter.

  Pipeline:
    ROOT → Branch Engine → Round Survival → Form Divergence → FINAL probability

  The core insight: Every upstream layer trusts SEASON stats.
  But season stats can LIE about who a team is RIGHT NOW.

  Three divergence patterns detected:

  1. STAT INFLATION (Gonzaga 2026, Virginia 2026):
     Season stats >> Recent performance. Team is DECLINING.
     ROOT overvalues them because season EM is inflated by strong early results.
     Signal: em_divergence << 0, margin_trend negative, efficiency dropping.

  2. STAT DEPRESSION (Texas 2026):
     Season stats << True talent ceiling. Team UNDERPERFORMED early
     but has the talent/coaching to unlock potential.
     Signal: em_divergence < 0 BUT high talent rank, experienced coach,
     low-variance defense = "sleeping giant" profile.
     NOTE: This pattern is hardest to detect from stats alone because the
     transformation often happens IN the tournament. We detect the POTENTIAL
     by cross-referencing declining stats with stable talent indicators.

  3. PAPER TIGER (Virginia 2026):
     Stats look elite but built on weak schedule or youth.
     Cross-reference: high season EM + low SOS + low experience +
     declining trend = fragile under pressure.
     Signal: em_divergence negative + experience below median + SOS below median.

Scoring: Same hybrid architecture as Branch Engine and Round Survival.
  Sub-conditions → severity → tanh curve → bounded shift.

Max shift: ±6% (conservative — this is the LAST layer, we don't want to
overcorrect what 3 prior layers already computed).
"""

import math
import os
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FormFlag:
    """One sub-condition in the form divergence evaluation."""
    name: str
    met: bool = False
    weight: float = 1.0
    value: float = 0.0
    description: str = ""


@dataclass
class FormDivergenceResult:
    """Result of form divergence analysis for one team."""
    team_name: str
    em_divergence: float = 0.0       # last10_EM - season_EM (negative = declining)
    margin_divergence: float = 0.0   # last10_margin - season_margin
    early_late_shift: float = 0.0    # first_half_EM - second_half_EM (positive = declining)
    ppg_divergence: float = 0.0      # last10_ppg - season_ppg
    severity: float = 0.0            # 0 to 1
    shift: float = 0.0               # probability shift
    pattern: str = "STABLE"          # STABLE, INFLATION, DEPRESSION, PAPER_TIGER, SLEEPING_GIANT
    flags: List[FormFlag] = field(default_factory=list)
    explanation: str = ""
    n_games: int = 0                 # games available in log


@dataclass
class GameLogStats:
    """Aggregated stats from a team's game log."""
    team_name: str = ""
    n_games: int = 0

    # Season aggregates
    season_em: float = 0.0
    season_margin: float = 0.0
    season_ppg: float = 0.0
    season_opp_ppg: float = 0.0

    # Last 10 games
    last10_em: float = 0.0
    last10_margin: float = 0.0
    last10_ppg: float = 0.0
    last10_opp_ppg: float = 0.0

    # Last 5 games (late-season form)
    last5_em: float = 0.0
    last5_margin: float = 0.0

    # First half vs second half
    first_half_em: float = 0.0
    second_half_em: float = 0.0

    # Win streaks / recent results
    last10_wins: int = 0
    last5_wins: int = 0

    # Variance metrics
    margin_std: float = 0.0           # scoring margin standard deviation
    rating_trend_slope: float = 0.0   # linear slope of rating over time


# ─────────────────────────────────────────────────────────────────────────────
# Game log loading and aggregation
# ─────────────────────────────────────────────────────────────────────────────

# Map team names to game log filenames
# Handle naming differences between our team objects and CSV filenames
NAME_TO_FILE = {
    "Miami FL": "Miami_FL",
    "St. John's": "St_Johns",
    "Michigan State": "Michigan_State",
    "Texas A&M": "Texas_A_and_M",
    "Iowa State": "Iowa_State",
    "Utah State": "Utah_State",
    "High Point": "High_Point",
    "Texas Tech": "Texas_Tech",
    "Saint Louis": "Saint_Louis",
    "VCU": "VCU",
    "North Carolina": "North_Carolina",
    "NC State": "NC_State",
    "UC San Diego": "UC_San_Diego",
    "Wright State": "Wright_State",
    "San Diego State": "San_Diego_State",
    "California Baptist": "California_Baptist",
}


def _team_to_filename(team_name: str) -> str:
    """Convert team name to game log filename."""
    if team_name in NAME_TO_FILE:
        return NAME_TO_FILE[team_name]
    return team_name.replace(" ", "_").replace("'", "")


def load_game_log(team_name: str, game_logs_dir: str = None) -> Optional[GameLogStats]:
    """Load and aggregate game log stats for a team.

    Game log CSV columns:
      team, opponent, date, venue, result, score_t, score_o,
      rating_t, rating_o, poss_t, poss_o, ...

    The 't_' prefix columns are tournament-adjusted; we use raw columns.
    Actually the columns are: t_score_t, t_score_o, score_t, score_o,
    t_rating_t, t_rating_o, rating_t, rating_o, t_poss_t, t_poss_o, poss_t, poss_o
    We use rating_t, rating_o (adjusted efficiency per game).
    """
    if game_logs_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        game_logs_dir = os.path.join(base, "archive-3", "game-logs")

    filename = _team_to_filename(team_name) + ".csv"
    filepath = os.path.join(game_logs_dir, filename)

    if not os.path.exists(filepath):
        return None

    games = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                game = {
                    'opponent': row.get('opponent', ''),
                    'result': row.get('result', ''),
                    'score_t': float(row.get('score_t', 0)),
                    'score_o': float(row.get('score_o', 0)),
                    'rating_t': float(row.get('rating_t', 0)),
                    'rating_o': float(row.get('rating_o', 0)),
                    'poss_t': float(row.get('poss_t', 0)),
                    'poss_o': float(row.get('poss_o', 0)),
                }
                games.append(game)
            except (ValueError, TypeError):
                continue

    if len(games) < 10:
        return None  # Need at least 10 games for meaningful divergence

    stats = GameLogStats(team_name=team_name, n_games=len(games))

    # Season aggregates
    all_em = [g['rating_t'] - g['rating_o'] for g in games]
    all_margin = [g['score_t'] - g['score_o'] for g in games]
    all_ppg = [g['score_t'] for g in games]
    all_opp_ppg = [g['score_o'] for g in games]

    stats.season_em = sum(all_em) / len(all_em)
    stats.season_margin = sum(all_margin) / len(all_margin)
    stats.season_ppg = sum(all_ppg) / len(all_ppg)
    stats.season_opp_ppg = sum(all_opp_ppg) / len(all_opp_ppg)
    stats.margin_std = _std(all_margin)

    # Last 10 games
    last10 = games[-10:]
    l10_em = [g['rating_t'] - g['rating_o'] for g in last10]
    l10_margin = [g['score_t'] - g['score_o'] for g in last10]
    l10_ppg = [g['score_t'] for g in last10]
    l10_opp_ppg = [g['score_o'] for g in last10]

    stats.last10_em = sum(l10_em) / len(l10_em)
    stats.last10_margin = sum(l10_margin) / len(l10_margin)
    stats.last10_ppg = sum(l10_ppg) / len(l10_ppg)
    stats.last10_opp_ppg = sum(l10_opp_ppg) / len(l10_opp_ppg)
    stats.last10_wins = sum(1 for g in last10 if g['result'] == 'W')

    # Last 5 games
    last5 = games[-5:]
    l5_em = [g['rating_t'] - g['rating_o'] for g in last5]
    l5_margin = [g['score_t'] - g['score_o'] for g in last5]

    stats.last5_em = sum(l5_em) / len(l5_em)
    stats.last5_margin = sum(l5_margin) / len(l5_margin)
    stats.last5_wins = sum(1 for g in last5 if g['result'] == 'W')

    # First half vs second half of season
    mid = len(games) // 2
    first_half = games[:mid]
    second_half = games[mid:]

    fh_em = [g['rating_t'] - g['rating_o'] for g in first_half]
    sh_em = [g['rating_t'] - g['rating_o'] for g in second_half]

    stats.first_half_em = sum(fh_em) / len(fh_em) if fh_em else 0.0
    stats.second_half_em = sum(sh_em) / len(sh_em) if sh_em else 0.0

    # Rating trend: linear regression slope over games
    if len(all_em) >= 5:
        x = list(range(len(all_em)))
        x_mean = sum(x) / len(x)
        y_mean = stats.season_em
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, all_em))
        den = sum((xi - x_mean) ** 2 for xi in x)
        stats.rating_trend_slope = num / den if den > 0 else 0.0

    return stats


def _std(vals: list) -> float:
    """Standard deviation."""
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def load_all_game_logs(team_names: List[str], game_logs_dir: str = None) -> Dict[str, GameLogStats]:
    """Load game logs for all teams."""
    logs = {}
    for name in team_names:
        stats = load_game_log(name, game_logs_dir)
        if stats:
            logs[name] = stats
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Form Divergence Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# Divergence thresholds (calibrated from 2026 R32 data)
# All R32 misses had |em_divergence| > 9.0
# Stable teams (correct picks) clustered around |em_divergence| < 7

MILD_THRESHOLD = 5.0      # |divergence| > 5: start paying attention
SIGNIFICANT_THRESHOLD = 10.0  # |divergence| > 10: meaningful signal
SEVERE_THRESHOLD = 15.0   # |divergence| > 15: red flag

# Max shift per round (conservative — this is the LAST layer)
FORM_MAX_SHIFT = {
    "R64": 0.04,   # Early round, less signal
    "R32": 0.08,   # Strong signal from 2026 validation — same as survival filter
    "S16": 0.07,   # Pressure increases, form matters more
    "E8":  0.06,   # Smaller sample but still relevant
    "F4":  0.04,
    "CHAMP": 0.03,
}


def evaluate_form_divergence(
    team_a,
    team_b,
    log_a: Optional[GameLogStats],
    log_b: Optional[GameLogStats],
    round_name: str,
    p_input: float,
) -> Tuple[float, FormDivergenceResult, FormDivergenceResult]:
    """Apply form divergence as the final pipeline filter.

    Two-component shift:
      1. INDIVIDUAL SEVERITY NET (40% weight): per-team decline evaluation
      2. LATE MOMENTUM DIFFERENTIAL (60% weight): direct last5 EM comparison
         This catches cases where season divergence is small but late-game
         momentum has diverged wildly (e.g. TTU cratering, Alabama surging).

    The momentum component is MORE heavily weighted because it captures
    the most CURRENT signal. Season divergence can be noisy (schedule
    effects, mid-season slumps that recover), but the last 5 games
    right before the tournament are the freshest signal we have.

    Args:
        team_a, team_b: Team objects
        log_a, log_b: GameLogStats for each team (can be None)
        round_name: Current round
        p_input: probability for team_a from upstream (post-survival)

    Returns:
        (p_final, result_a, result_b)
    """
    max_shift = FORM_MAX_SHIFT.get(round_name, 0.04)

    # 1. Individual team evaluations
    result_a = _evaluate_team_form(team_a, log_a, team_b)
    result_b = _evaluate_team_form(team_b, log_b, team_a)

    # 2. COMPONENT 1: Individual severity net (40% weight)
    # Higher severity = more declining = worse for that team
    net_severity = result_b.severity - result_a.severity
    sev_shift = math.tanh(net_severity * 1.8) * max_shift * 0.4

    # 3. COMPONENT 2: Late momentum differential (60% weight)
    # Compare last5 EM directly — this is the FRESHEST signal
    # Positive = team_a has better late momentum → shift toward A
    mom_shift = _compute_late_momentum(log_a, log_b, max_shift) * 0.6

    # 4. Combined shift (capped at max)
    shift = sev_shift + mom_shift
    shift = max(-max_shift, min(max_shift, shift))

    p_final = max(0.01, min(0.99, p_input + shift))

    result_a.shift = shift
    result_b.shift = -shift

    return p_final, result_a, result_b


def _compute_late_momentum(
    log_a: Optional[GameLogStats],
    log_b: Optional[GameLogStats],
    max_shift: float,
) -> float:
    """Compute matchup-level late momentum shift.

    Compares the last5 EM of both teams, RELATIVE to what their
    season stats would predict. This catches divergence that's
    invisible to individual-team evaluation.

    Key insight: season EM gap tells you "who SHOULD be better."
    Last5 EM gap tells you "who IS better RIGHT NOW."
    If these disagree strongly, that's a form divergence signal.

    Example (TTU vs Alabama):
      Season EM gap: 14.1 - 16.5 = -2.4 (close matchup)
      Last5 EM gap: -0.0 - 22.4 = -22.4 (Alabama DOMINATING)
      Gap divergence: -22.4 - (-2.4) = -20.0 → massive Alabama signal

    Returns shift: positive = favors team_a, negative = favors team_b.
    """
    if log_a is None or log_b is None:
        return 0.0

    # Season EM gap (what the model "expects")
    season_gap = log_a.season_em - log_b.season_em

    # Last5 EM gap (what's actually happening NOW)
    last5_gap = log_a.last5_em - log_b.last5_em

    # Gap divergence: how much has the matchup shifted vs expectations
    gap_divergence = last5_gap - season_gap

    # Also check absolute last5 floor — a team near zero EM is in freefall
    # regardless of what their season says
    floor_penalty_a = max(0, 5.0 - log_a.last5_em) * 0.01  # penalty per point below 5
    floor_penalty_b = max(0, 5.0 - log_b.last5_em) * 0.01
    floor_shift = floor_penalty_b - floor_penalty_a  # positive = B is worse

    # Also check last5 acceleration (surging vs cratering)
    accel_a = log_a.last5_em - log_a.last10_em  # positive = improving
    accel_b = log_b.last5_em - log_b.last10_em
    accel_diff = accel_a - accel_b  # positive = A improving more

    # Combine signals
    # Gap divergence: normalize by dividing by ~20 (range is typically -30 to +30)
    raw_momentum = (
        math.tanh(gap_divergence / 20.0) * 0.50 +   # gap divergence signal
        math.tanh(accel_diff / 15.0) * 0.30 +         # acceleration signal
        math.tanh(floor_shift * 20) * 0.20             # absolute floor signal
    )

    return math.tanh(raw_momentum * 2.0) * max_shift


def _evaluate_team_form(team, log: Optional[GameLogStats], opponent) -> FormDivergenceResult:
    """Evaluate form divergence for one team.

    Returns a FormDivergenceResult where higher severity = MORE declining.
    A team with severity 0 is stable/improving.
    """
    result = FormDivergenceResult(team_name=team.name)

    if log is None:
        result.pattern = "NO_DATA"
        result.explanation = "No game log data available"
        return result

    result.n_games = log.n_games

    # Core divergence metrics
    result.em_divergence = log.last10_em - log.season_em
    result.margin_divergence = log.last10_margin - log.season_margin
    result.early_late_shift = log.first_half_em - log.second_half_em  # positive = declining
    result.ppg_divergence = log.last10_ppg - log.season_ppg

    # ─── Sub-condition evaluation ─────────────────────────────────────────
    flags = []

    # 1. EM DIVERGENCE (weight 0.30) — the core signal
    # Negative em_divergence = team is declining relative to season
    em_div_abs = abs(result.em_divergence)
    em_declining = result.em_divergence < -MILD_THRESHOLD

    flags.append(FormFlag(
        name="em_divergence",
        met=em_declining,
        weight=0.30,
        value=result.em_divergence,
        description=f"EM divergence: last10={log.last10_em:.1f} vs season={log.season_em:.1f} "
                    f"(Δ{result.em_divergence:+.1f})"
    ))

    # 2. MARGIN TREND (weight 0.20) — is the scoring margin declining?
    margin_declining = result.margin_divergence < -3.0  # 3+ points worse

    flags.append(FormFlag(
        name="margin_trend",
        met=margin_declining,
        weight=0.20,
        value=result.margin_divergence,
        description=f"Margin trend: last10={log.last10_margin:.1f} vs season={log.season_margin:.1f} "
                    f"(Δ{result.margin_divergence:+.1f})"
    ))

    # 3. EARLY→LATE SHIFT (weight 0.20) — first half of season vs second half
    early_late_declining = result.early_late_shift > 8.0  # 8+ EM point drop

    flags.append(FormFlag(
        name="early_late_shift",
        met=early_late_declining,
        weight=0.20,
        value=result.early_late_shift,
        description=f"Early→Late: first_half_EM={log.first_half_em:.1f} → second_half_EM={log.second_half_em:.1f} "
                    f"(shift {result.early_late_shift:+.1f})"
    ))

    # 4. LAST-5 ACCELERATION (weight 0.15) — is the decline accelerating?
    # Compare last5 to last10 — if last5 is WORSE, the decline is accelerating
    # NOTE: triggers even without full em_divergence threshold — a team can have
    # mild season divergence but CRATERING last 5 games (e.g. TTU: em_div=-3.7
    # but last5=-0.0 vs last10=10.4, that's a -10.4 acceleration)
    last5_vs_last10 = log.last5_em - log.last10_em
    accelerating = last5_vs_last10 < -5.0  # removed em_declining requirement

    flags.append(FormFlag(
        name="decline_acceleration",
        met=accelerating,
        weight=0.15,
        value=last5_vs_last10,
        description=f"Decline acceleration: last5_EM={log.last5_em:.1f} vs last10_EM={log.last10_em:.1f} "
                    f"(Δ{last5_vs_last10:+.1f})"
    ))

    # 5. PAPER TIGER DETECTOR (weight 0.15) — high season stats but weak
    #    schedule + low experience + declining form = fragile
    #    Relaxed experience threshold: even "experienced" teams can be
    #    paper tigers if their early→late drop is extreme (>15 points)
    exp_val = getattr(team, 'exp', 0.5)
    early_late_gap = log.first_half_em - log.second_half_em
    is_paper_tiger = (
        em_declining and
        (
            (exp_val < 0.45 and early_late_gap > 10) or   # inexperienced + moderate drop
            (early_late_gap > 20)                           # ANY team with extreme drop (>20)
        )
    )

    flags.append(FormFlag(
        name="paper_tiger",
        met=is_paper_tiger,
        weight=0.15,
        value=exp_val,
        description=f"Paper tiger: exp={exp_val:.2f}, "
                    f"early_EM={log.first_half_em:.1f}→late_EM={log.second_half_em:.1f} "
                    f"(gap={early_late_gap:+.1f}), declining={em_declining}"
    ))

    # 6. LATE MOMENTUM FLOOR (weight 0.20) — absolute last5 performance
    #    If a team's last5 EM is below 5.0, they're in freefall regardless
    #    of season stats. This catches teams like Texas Tech (last5 = 0.0)
    #    whose season divergence is small but who are cratering RIGHT NOW.
    late_floor_breach = log.last5_em < 5.0

    flags.append(FormFlag(
        name="late_momentum_floor",
        met=late_floor_breach,
        weight=0.20,
        value=log.last5_em,
        description=f"Late momentum floor: last5_EM={log.last5_em:.1f} "
                    f"(threshold: 5.0, {'BREACH' if late_floor_breach else 'OK'})"
    ))

    result.flags = flags

    # ─── Severity computation ─────────────────────────────────────────────
    result.severity = _compute_form_severity(flags, em_div_abs)

    # ─── Pattern classification ───────────────────────────────────────────
    result.pattern = _classify_pattern(result, log, team)

    # ─── Explanation ──────────────────────────────────────────────────────
    n_triggered = sum(1 for f in flags if f.met)
    result.explanation = (
        f"{result.pattern} | severity={result.severity:.2f} | "
        f"em_div={result.em_divergence:+.1f} | "
        f"flags={n_triggered}/{len(flags)}"
    )

    return result


def _compute_form_severity(flags: List[FormFlag], em_div_abs: float) -> float:
    """Compute severity using hybrid compounding + EM-scaled base.

    TWO PATHS to severity:

    PATH A (season divergence): When em_divergence is significant
      - Geometric mean of flag_raw × em_scale
      - Requires BOTH diverging stats AND confirming signals

    PATH B (late momentum floor): When last5 EM is critically low
      - Can trigger even with small season divergence
      - Based on the absolute floor breach severity
      - Ensures teams in late-season freefall aren't invisible

    Final severity = max(path_a, path_b)
    """
    triggered = [f for f in flags if f.met]
    if not triggered:
        return 0.0

    total_weight = sum(f.weight for f in flags)
    if total_weight == 0:
        return 0.0

    triggered_weight = sum(f.weight for f in triggered)
    flag_raw = triggered_weight / total_weight

    n_triggered = len(triggered)

    # Multiplicative cascade at 3+ triggers
    if n_triggered >= 3:
        cascade = 1.0
        for f in triggered:
            cascade *= (1.0 + f.weight * 0.4)
        flag_raw = min(flag_raw * cascade * 0.5, 1.0)

    # PATH A: EM-magnitude scaling (existing approach)
    em_scale = math.tanh(max(0, em_div_abs - MILD_THRESHOLD) / 10.0)
    path_a = math.sqrt(flag_raw * em_scale) if em_scale > 0 else 0.0

    # PATH B: Late momentum floor
    # If the late_momentum_floor flag fired, compute severity based on
    # HOW LOW the last5 EM is. A team at 0 EM gets higher severity than
    # a team at 4 EM, even if their season divergence is small.
    path_b = 0.0
    for f in triggered:
        if f.name == "late_momentum_floor" and f.met:
            # f.value = last5_em. Lower = worse.
            # Map: 5.0→0.0 (threshold), 0.0→0.4, -5.0→0.6, -10.0→0.7
            floor_severity = math.tanh(max(0, 5.0 - f.value) / 8.0) * 0.7
            # Boost if decline_acceleration also fires (confirming signal)
            accel_fired = any(ff.name == "decline_acceleration" and ff.met for ff in flags)
            if accel_fired:
                floor_severity *= 1.3
            path_b = min(floor_severity, 0.8)

    severity = max(path_a, path_b)

    # Final tanh smoothing
    severity = math.tanh(severity * 2.0)

    return severity


def _classify_pattern(result: FormDivergenceResult, log: GameLogStats, team) -> str:
    """Classify the divergence pattern."""
    if result.severity < 0.05:
        return "STABLE"

    em_declining = result.em_divergence < -MILD_THRESHOLD
    em_improving = result.em_divergence > MILD_THRESHOLD

    if not em_declining and not em_improving:
        return "STABLE"

    if em_declining:
        # Check for paper tiger: declining + inexperienced + big early→late drop
        is_paper_tiger = any(f.name == "paper_tiger" and f.met for f in result.flags)
        if is_paper_tiger:
            return "PAPER_TIGER"

        # Check if decline is accelerating
        accelerating = any(f.name == "decline_acceleration" and f.met for f in result.flags)
        if accelerating:
            return "COLLAPSING"

        return "INFLATION"

    if em_improving:
        return "SURGING"

    return "STABLE"


# ─────────────────────────────────────────────────────────────────────────────
# Sleeping Giant Detection (Type 2 — hardest to catch)
# ─────────────────────────────────────────────────────────────────────────────

def detect_sleeping_giant(team, log: Optional[GameLogStats]) -> float:
    """Detect teams whose season stats UNDERSTATE their true quality.

    The "sleeping giant" pattern (Texas 2026):
    - Season stats are mediocre (low EM, bad record)
    - BUT talent indicators are strong (recruiting, roster talent)
    - AND the team has a veteran coach capable of tactical adjustments
    - AND recent defensive trends suggest they're "figuring it out"

    Returns a bonus multiplier (0.0 = not a sleeping giant, 0.0-0.3 = partial)
    that can COUNTERACT a declining form signal.

    This is intentionally conservative because detecting potential
    is fundamentally harder than detecting decline.
    """
    if log is None:
        return 0.0

    # Indicators of a sleeping giant:
    indicators = 0
    total = 0

    # 1. High talent despite mediocre record
    # (talent rank is better than their adj_em rank would suggest)
    talent = getattr(team, 'roster_rank', 100)
    adj_em_rank = getattr(team, 'adj_em_rank', 50)
    if talent > 0 and talent < 30 and adj_em_rank > talent + 15:
        indicators += 1  # Talent is much better than results
    total += 1

    # 2. Defense improving in second half
    # (even if offense is rough, defensive improvement = coach adjusting)
    if log.n_games >= 20:
        mid = log.n_games // 2
        # We don't have per-game defensive efficiency separately,
        # but we can check if opponent scoring is declining
        # Use last10 opp_ppg vs season opp_ppg
        if log.last10_opp_ppg < log.season_opp_ppg - 2:
            indicators += 1  # Defense tightening recently
    total += 1

    # 3. High variance in results (inconsistent but capable of greatness)
    if log.margin_std > 15:
        indicators += 1  # Wide variance = high ceiling exists
    total += 1

    # 4. Better record recently despite worse EM
    # (winning close games = clutch development)
    if log.last10_wins >= 6 and log.last10_em < log.season_em:
        indicators += 1  # Winning despite declining advanced stats
    total += 1

    if total == 0:
        return 0.0

    return min(indicators / total * 0.3, 0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def form_divergence_report(result: FormDivergenceResult) -> str:
    """Human-readable report for one team's form divergence."""
    lines = []
    lines.append(f"  {result.team_name} [FORM DIVERGENCE]:")
    lines.append(f"    Pattern: {result.pattern} | Severity: {result.severity:.2f} | Shift: {result.shift:+.1%}")
    lines.append(f"    EM divergence: {result.em_divergence:+.1f} | Margin div: {result.margin_divergence:+.1f}")
    lines.append(f"    Early→Late shift: {result.early_late_shift:+.1f} | Games: {result.n_games}")

    n_triggered = sum(1 for f in result.flags if f.met)
    lines.append(f"    Flags: {n_triggered}/{len(result.flags)} triggered")
    for f in result.flags:
        marker = "✓" if f.met else "·"
        lines.append(f"      {marker} {f.name} (w={f.weight:.2f}): {f.description}")

    return "\n".join(lines)
