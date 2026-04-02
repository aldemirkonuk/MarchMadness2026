"""Injury-Adjusted Prediction Engine -- Phases 2-5.

Quantifies player importance via BPR analysis, degrades team parameters
when players are out, handles round-by-round availability in Monte Carlo,
and applies post-ensemble star-dependency penalties.

Architecture:
  Layer 1 (Pre-Composite): Adjust team params (adj_em, shooting, ast, etc.)
                           before Phase 1A runs. Injuries flow through the
                           entire pipeline naturally.
  Layer 2 (Post-Ensemble): For star-carrier teams (BPR share > 30%), apply
                           compounding Usage Vacuum penalty after ensemble.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.models import Team, Matchup
from src.utils import canonical_name


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# data/injuries.csv schema (one row per player; UTF-8 CSV):
#   team              — canonical team name (matches KenPom / Team.name)
#   player            — short name, e.g. "J. Jefferson"
#   position          — G | F | C (single letter or short code)
#   status            — OUT | QUESTIONABLE | DOUBTFUL | EXPECTED | PROBABLE
#   injury_type       — free text (Knee, Foot, Undisclosed, etc.)
#   earliest_return_round — R64 | R32 | S16 | E8 | F4 | Championship | NONE
#   play_probability_r64 … play_probability_champ — float 0.0–1.0, availability
#                         if the team reaches that round (0 = cannot play)
#   notes             — context for humans + optional PPG/timeline for penalty floor
ROUND_COLS = {
    "R64": "play_probability_r64",
    "R32": "play_probability_r32",
    "S16": "play_probability_s16",
    "E8":  "play_probability_e8",
    "F4":  "play_probability_f4",
    "Championship": "play_probability_champ",
}

# Star carrier threshold: if a player's BPR share exceeds this, trigger
# the Usage Vacuum (post-ensemble) penalty.
STAR_CARRIER_THRESHOLD = 0.30

# Maximum AdjEM penalty from injuries (prevents complete team collapse
# in case of pathological BPR distributions)
MAX_ADJEM_PENALTY = 8.0

# Per-player penalty cap: no single player's absence should produce a raw
# penalty exceeding this value. With the ~5.5x AdjEM scaling, 0.35 raw ≈
# 1.9 AdjEM — consistent with the worst historical cases (MASH Unit).
MAX_SINGLE_PLAYER_PENALTY = 0.35

# Non-effectant filter: if a player's possessions share is below this
# threshold, the team's season stats already reflect their absence —
# penalizing again would be double-counting.
NON_EFFECTANT_POSS_THRESHOLD = 0.05  # 5% of team possessions

# Months that indicate a player was out before or at the very start of the
# season (2025-26 season starts November).  If the injury notes say "since"
# one of these months, the player is non-effectant.
_EARLY_ABSENCE_MONTHS = {
    "jun", "jul", "aug", "sep", "oct", "nov",
    "june", "july", "august", "september", "october", "november",
}

# Ramp-up effectiveness when a player returns from injury.
# First round back = BASE, each subsequent round += INCREMENT, capped at CAP
RAMPUP_BASE = 0.65
RAMPUP_INCREMENT = 0.10
RAMPUP_CAP = 0.90

# Multi-category leader amplifier: if an injured player leads their team
# in N of the tracked BPR categories (obpr, dbpr, bpr, poss, plus_minus,
# box_obpr, box_dbpr, box_bpr), the penalty is amplified.
#
# Rationale: a player who leads 1 category is replaceable. A player who
# leads 3+ is the HUB — the team's offense, defense, and system all run
# through them. No single replacement can fill multiple roles, so the
# loss compounds across dimensions.
MULTI_CAT_LEADER_CATEGORIES = ["obpr", "dbpr", "bpr", "poss", "plus_minus",
                                "box_obpr", "box_dbpr", "box_bpr"]
MULTI_CAT_AMPLIFIER_2 = 1.15      # leads 2 categories → minor two-dimensional loss
MULTI_CAT_AMPLIFIER_3 = 1.30      # leads 3 categories → system hub
MULTI_CAT_AMPLIFIER_4 = 1.45      # leads 4 categories → team identity loss
MULTI_CAT_AMPLIFIER_5PLUS = 1.60  # leads 5+ categories → complete collapse (capped)

# Collapse detection: when a multi-category leader ALSO has dominant
# share of the team's production, the loss is systemic — the team can't
# just "spread the load" because there's nobody who can fill any of the
# roles adequately.
COLLAPSE_CATS_THRESHOLD = 4       # must lead at least 4 categories (stronger signal)
COLLAPSE_BPR_SHARE_THRESHOLD = 0.20  # AND have >= 20% of team BPR
COLLAPSE_ADDITIONAL_PENALTY = 0.10   # flat additional AdjEM penalty for system breakdown

# ── Star Isolation Index (SII): BYU/Dybantsa problem ────────────────────
# When a team's offense runs through ONE player (due to injury to #2 option
# or simply roster construction), the star gets predictable and fatigued.
# SII = (star_bpr_share) * (star_poss_share).  Higher = more isolated.
#   SII > 0.08  → "star-dependent" — mild additional penalty
#   SII > 0.12  → "star-isolated"  — significant penalty (team = one man)
# Day 1 proof: BYU (Dybantsa 35 pts, 40 min, still lost by 8) had SII ~0.15
STAR_ISOLATION_THRESHOLD_MILD = 0.08
STAR_ISOLATION_THRESHOLD_SEVERE = 0.12
STAR_ISOLATION_PENALTY_MILD = 0.04     # AdjEM penalty for star-dependent
STAR_ISOLATION_PENALTY_SEVERE = 0.10   # AdjEM penalty for star-isolated

# PPG penalty floor: minimum AdjEM penalty based on points-per-game
# extracted from injury notes. Catches high-volume scorers on deep teams
# where BPR share alone is diluted.
PPG_PENALTY_FLOOR = {15.0: 0.08, 18.0: 0.12, 22.0: 0.18}


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InjuredPlayer:
    """Represents one injured player and their impact metrics."""
    team: str
    player: str
    position: str
    status: str
    injury_type: str
    earliest_return_round: str

    # Round-by-round play probabilities (from CSV)
    play_probs: Dict[str, float] = field(default_factory=dict)

    # Computed impact metrics (Phase 2)
    bpr: float = 0.0
    obpr: float = 0.0
    dbpr: float = 0.0
    poss: float = 0.0
    bpr_share: float = 0.0           # player_bpr / team_total_bpr
    minutes_share: float = 0.0       # player_poss / team_total_poss
    replacement_bpr: float = 0.0     # BPR of the replacement player
    replacement_factor: float = 0.0  # replacement_bpr / injured_bpr (capped)
    penalty: float = 0.0             # (BPR × min_share) × (1 - replacement_factor)
    is_star_carrier: bool = False     # BPR share > threshold
    position_scarcity: int = 0       # count of same-position players in top-8
    categories_led: int = 0          # how many stat categories this player leads on team
    multi_cat_amplifier: float = 1.0 # penalty multiplier from multi-category leadership
    collapse_risk: bool = False      # True if multi-cat leader + high BPR share → system breakdown

    notes: str = ""

    # Non-effectant filter: True if the player's absence is already baked
    # into the team's season stats (0 games, or out most of the season).
    non_effectant: bool = False
    non_effectant_reason: str = ""

    # ── Injury severity tier (added for branch engine integration) ────────
    # Tiers: OUT_SEASON, OUT_GAME, DOUBTFUL, QUESTIONABLE, PROBABLE
    # "OUT_SEASON" = out 4+ weeks, fundamentally changes team identity
    # "OUT_GAME"   = recently injured, out for this game specifically
    severity_tier: str = ""              # computed in quantify_player_impacts
    weeks_out: float = 0.0              # approximate weeks since injury
    ppg_before_injury: float = 0.0      # PPG before going down (from notes)


@dataclass
class TeamInjuryProfile:
    """Aggregated injury impact for a team."""
    team_name: str
    injured_players: List[InjuredPlayer] = field(default_factory=list)

    # Aggregated penalties
    total_penalty: float = 0.0       # sum of all player penalties (weighted by play prob)
    adj_em_penalty: float = 0.0      # AdjEM degradation
    shooting_penalty: float = 0.0    # shooting_eff degradation
    ast_penalty: float = 0.0         # ast_pct degradation
    defense_penalty: float = 0.0     # dvi/rpi_rim degradation
    rebound_penalty: float = 0.0     # rbm/orb/drb degradation
    has_star_carrier_out: bool = False
    star_vacuum_penalty: float = 0.0  # post-ensemble multiplier

    # ── Crippled roster detection (for Branch 15) ─────────────────────
    has_crippled_roster: bool = False   # top-2 scorer out 4+ weeks
    crippled_player_name: str = ""
    crippled_weeks_out: float = 0.0
    crippled_ppg_lost: float = 0.0
    top_injured_severity_tier: str = "" # worst severity tier among effectant


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Load Injury Data
# ─────────────────────────────────────────────────────────────────────────────

def load_injuries() -> List[InjuredPlayer]:
    """Load injury data from data/injuries.csv."""
    path = os.path.join(DATA_DIR, "injuries.csv")
    if not os.path.exists(path):
        print("  [INJURY] No injuries.csv found -- skipping injury adjustments")
        return []

    df = pd.read_csv(path)
    injuries = []

    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip().upper()

        # Skip redshirts and non-impact players
        if "redshirt" in str(row.get("notes", "")).lower():
            continue
        if "excluded" in str(row.get("notes", "")).lower():
            continue

        ip = InjuredPlayer(
            team=canonical_name(str(row["team"]).strip()),
            player=str(row["player"]).strip(),
            position=str(row.get("position", "")).strip(),
            status=status,
            injury_type=str(row.get("injury_type", "")).strip(),
            earliest_return_round=str(row.get("earliest_return_round", "NONE")).strip(),
            notes=str(row.get("notes", "")).strip(),
        )

        # Parse round-by-round play probabilities
        for round_name, col_name in ROUND_COLS.items():
            ip.play_probs[round_name] = float(row.get(col_name, 0.0))

        injuries.append(ip)

    return injuries


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Player Impact Quantification
# ─────────────────────────────────────────────────────────────────────────────

def quantify_player_impacts(injuries: List[InjuredPlayer],
                            player_df: pd.DataFrame) -> Dict[str, TeamInjuryProfile]:
    """Compute BPR share, replacement factor, and penalty for each injury.

    Returns a dict of team_name -> TeamInjuryProfile.
    """
    if player_df.empty:
        print("  [INJURY] No player data available -- using role-based fallback")
        return _fallback_role_based(injuries)

    profiles: Dict[str, TeamInjuryProfile] = {}

    # Group injuries by team
    team_injuries: Dict[str, List[InjuredPlayer]] = {}
    for ip in injuries:
        team_injuries.setdefault(ip.team, []).append(ip)

    for team_name, team_inj_list in team_injuries.items():
        profile = TeamInjuryProfile(team_name=team_name)

        # Get all players for this team
        team_players = player_df[player_df["team"] == team_name]
        if team_players.empty:
            # Try fuzzy match
            for alt in player_df["team"].unique():
                if team_name.lower() in alt.lower() or alt.lower() in team_name.lower():
                    team_players = player_df[player_df["team"] == alt]
                    break

        if team_players.empty:
            # Fallback: use role-based estimates with non-effectant filtering
            for ip in team_inj_list:
                ne_by_notes, ne_reason = _is_non_effectant_by_notes(ip.notes)
                if ne_by_notes:
                    ip.non_effectant = True
                    ip.non_effectant_reason = f"[team-fallback] {ne_reason}"
                    ip.penalty = 0.0
                elif (ip.status == "OUT"
                      and ip.play_probs.get("R64", 0) == 0.0
                      and not _notes_suggest_contributor(ip.notes)):
                    ip.non_effectant = True
                    ip.non_effectant_reason = (
                        "[team-fallback] season-ending OUT with no "
                        "contributor indicators — likely never played"
                    )
                    ip.penalty = 0.0
                else:
                    ip.penalty = _role_penalty(ip.status)
                profile.injured_players.append(ip)
            effectant = [ip for ip in profile.injured_players if not ip.non_effectant]
            profile.total_penalty = sum(ip.penalty for ip in effectant)
            profiles[team_name] = profile
            continue

        # Sort by BPR descending (top 8 = rotation)
        team_players = team_players.nlargest(min(12, len(team_players)), "bpr")
        team_total_bpr = max(team_players["bpr"].sum(), 0.1)
        team_total_poss = max(team_players["poss"].sum(), 1.0)

        for ip in team_inj_list:
            # Find this player in the roster
            player_match = _find_player(ip.player, team_players)

            if player_match is not None:
                ip.bpr = float(player_match["bpr"])
                ip.obpr = float(player_match.get("obpr", 0))
                ip.dbpr = float(player_match.get("dbpr", 0))
                ip.poss = float(player_match.get("poss", 0))

                ip.bpr_share = ip.bpr / team_total_bpr
                ip.minutes_share = ip.poss / team_total_poss

                # ── Non-effectant filter ──────────────────────────────
                # If the player barely played (< 5% of team possessions),
                # the team's stats already reflect their absence.
                if ip.minutes_share < NON_EFFECTANT_POSS_THRESHOLD:
                    ip.non_effectant = True
                    ip.non_effectant_reason = (
                        f"poss share {ip.minutes_share:.1%} < {NON_EFFECTANT_POSS_THRESHOLD:.0%} "
                        f"threshold — team stats already reflect absence"
                    )
                    ip.penalty = 0.0
                    profile.injured_players.append(ip)
                    continue

                # Also check notes for early-season injuries
                ne_by_notes, ne_reason = _is_non_effectant_by_notes(ip.notes)
                if ne_by_notes and ip.minutes_share < 0.10:
                    # Notes say early absence AND low minutes: non-effectant
                    ip.non_effectant = True
                    ip.non_effectant_reason = ne_reason
                    ip.penalty = 0.0
                    profile.injured_players.append(ip)
                    continue
                # ──────────────────────────────────────────────────────

                # Find replacement: next-best player at same position (±1 pos)
                # Clean the injury name: remove suffixes like Jr., II, III and periods
                _clean_name = ip.player.replace("Jr.", "").replace("II", "").replace("III", "")
                _name_parts = [p.strip().rstrip(".") for p in _clean_name.split() if p.strip().rstrip(".")]
                _last_name_key = _name_parts[-1].lower() if _name_parts else ""

                if _last_name_key:
                    available = team_players[
                        ~team_players["name"].str.lower().str.contains(
                            _last_name_key, na=False
                        )
                    ]
                else:
                    available = team_players
                if len(available) > 0:
                    # Position-aware: prefer same position, fallback to any
                    pos_key = ip.position.upper()
                    pos_matches = available  # Default: any available player
                    if pos_key in ("G", "F", "C"):
                        # Simple position grouping
                        if pos_key == "G":
                            pos_matches = available[
                                available.get("pos", pd.Series(dtype=str))
                                .str.upper()
                                .str.contains("G", na=True)
                            ] if "pos" in available.columns else available
                        elif pos_key == "F":
                            pos_matches = available[
                                available.get("pos", pd.Series(dtype=str))
                                .str.upper()
                                .str.contains("F", na=True)
                            ] if "pos" in available.columns else available
                        elif pos_key == "C":
                            pos_matches = available[
                                available.get("pos", pd.Series(dtype=str))
                                .str.upper()
                                .str.contains("C|F", na=True)
                            ] if "pos" in available.columns else available

                    if len(pos_matches) == 0:
                        pos_matches = available

                    # The replacement is the best available player not already injured
                    def _extract_last_name(name):
                        clean = name.replace("Jr.", "").replace("II", "").replace("III", "")
                        parts = [p.strip().rstrip(".") for p in clean.split() if p.strip().rstrip(".")]
                        return parts[-1].lower() if parts else ""
                    injured_names = {_extract_last_name(ip2.player)
                                     for ip2 in team_inj_list}
                    injured_names.discard("")  # safety: never match empty string
                    for _, candidate in pos_matches.iterrows():
                        cand_name = str(candidate["name"]).lower()
                        if not any(inj_name in cand_name for inj_name in injured_names):
                            ip.replacement_bpr = float(candidate["bpr"])
                            break
                    else:
                        # All candidates are also injured -- use worst case
                        ip.replacement_bpr = float(pos_matches.iloc[-1]["bpr"])

                ip.replacement_factor = min(
                    ip.replacement_bpr / max(ip.bpr, 0.1), 0.90
                )

                # Core penalty: (BPR × minutes_share) × (1 - replacement_factor)
                ip.penalty = (ip.bpr * ip.minutes_share) * (1.0 - ip.replacement_factor)

                # Position scarcity
                ip.position_scarcity = len(pos_matches) if 'pos_matches' in dir() else len(available)

                # Multi-category leader amplifier
                cats_led = 0
                for cat_col in MULTI_CAT_LEADER_CATEGORIES:
                    if cat_col in team_players.columns:
                        team_leader_val = team_players[cat_col].max()
                        player_val = float(player_match.get(cat_col, 0))
                        if player_val >= team_leader_val and team_leader_val > 0:
                            cats_led += 1
                ip.categories_led = cats_led
                if cats_led >= 5:
                    ip.multi_cat_amplifier = MULTI_CAT_AMPLIFIER_5PLUS
                    ip.penalty *= MULTI_CAT_AMPLIFIER_5PLUS
                elif cats_led == 4:
                    ip.multi_cat_amplifier = MULTI_CAT_AMPLIFIER_4
                    ip.penalty *= MULTI_CAT_AMPLIFIER_4
                elif cats_led == 3:
                    ip.multi_cat_amplifier = MULTI_CAT_AMPLIFIER_3
                    ip.penalty *= MULTI_CAT_AMPLIFIER_3
                elif cats_led == 2:
                    ip.multi_cat_amplifier = MULTI_CAT_AMPLIFIER_2
                    ip.penalty *= MULTI_CAT_AMPLIFIER_2

                # Collapse detection: multi-cat leader + high BPR share
                # = systemic breakdown. The team can't "spread the load"
                # because this player WAS the system.
                if (cats_led >= COLLAPSE_CATS_THRESHOLD
                        and ip.bpr_share >= COLLAPSE_BPR_SHARE_THRESHOLD):
                    ip.collapse_risk = True
                    ip.penalty += COLLAPSE_ADDITIONAL_PENALTY

                # PPG floor: BPR is an efficiency metric and can undervalue
                # high-volume scorers on deep teams. If notes show high PPG,
                # enforce a minimum penalty that reflects actual production.
                ppg = _extract_ppg_from_notes(ip.notes)
                if ppg > 0:
                    for ppg_threshold in sorted(PPG_PENALTY_FLOOR.keys(), reverse=True):
                        if ppg >= ppg_threshold:
                            floor = PPG_PENALTY_FLOOR[ppg_threshold]
                            if ip.penalty < floor:
                                ip.penalty = floor
                            break

                # Per-player cap: prevent any single absence from producing
                # a penalty beyond historical bounds.
                ip.penalty = min(ip.penalty, MAX_SINGLE_PLAYER_PENALTY)

            else:
                # Player not found in roster data at all.
                # This is a strong signal they never played (or played so
                # little that EvanMiya didn't track them).
                ne_by_notes, ne_reason = _is_non_effectant_by_notes(ip.notes)
                if ne_by_notes:
                    # Notes confirm early/pre-season absence
                    ip.non_effectant = True
                    ip.non_effectant_reason = (
                        f"not in roster data + {ne_reason}"
                    )
                    ip.penalty = 0.0
                elif (ip.status == "OUT"
                      and ip.play_probs.get("R64", 0) == 0.0
                      and not _notes_suggest_contributor(ip.notes)):
                    # Season-ending OUT, not in EvanMiya, and notes don't
                    # mention stats/PPG/key player → very likely never played.
                    ip.non_effectant = True
                    ip.non_effectant_reason = (
                        "not found in player data (likely never played or "
                        "minimal contribution) — team stats reflect absence"
                    )
                    ip.penalty = 0.0
                else:
                    # Player not found but may be a real contributor
                    # (name mismatch, or notes suggest they were active).
                    # Use role-based fallback, boosted by PPG floor if available.
                    ip.penalty = _role_penalty(ip.status)
                    ppg = _extract_ppg_from_notes(ip.notes)
                    if ppg > 0:
                        for ppg_threshold in sorted(PPG_PENALTY_FLOOR.keys(), reverse=True):
                            if ppg >= ppg_threshold:
                                floor = PPG_PENALTY_FLOOR[ppg_threshold]
                                if ip.penalty < floor:
                                    ip.penalty = floor
                                break

            # Star carrier detection (only for effectant players)
            if not ip.non_effectant:
                ip.is_star_carrier = ip.bpr_share > STAR_CARRIER_THRESHOLD

            profile.injured_players.append(ip)

        effectant = [ip for ip in profile.injured_players if not ip.non_effectant]
        profile.total_penalty = sum(ip.penalty for ip in effectant)
        profile.has_star_carrier_out = any(ip.is_star_carrier for ip in effectant)

        # ── Compute severity tiers for all effectant players ──────────────
        for ip in effectant:
            ip.severity_tier = compute_severity_tier(ip)
            ip.weeks_out = _estimate_weeks_out(ip.notes)
            ip.ppg_before_injury = _extract_ppg_from_notes_v2(ip.notes)

        # ── Crippled roster detection ─────────────────────────────────────
        # Check if any top-2 BPR player is OUT_SEASON (4+ weeks)
        tier_priority = {"OUT_SEASON": 5, "OUT_GAME": 4, "DOUBTFUL": 3,
                         "QUESTIONABLE": 2, "PROBABLE": 1, "": 0}
        worst_tier = ""
        for ip in effectant:
            if tier_priority.get(ip.severity_tier, 0) > tier_priority.get(worst_tier, 0):
                worst_tier = ip.severity_tier
        profile.top_injured_severity_tier = worst_tier

        # Crippled = top-2 BPR player out for 4+ weeks
        top2_bpr = sorted(effectant, key=lambda x: x.bpr, reverse=True)[:2]
        for ip in top2_bpr:
            if ip.severity_tier == "OUT_SEASON" and ip.weeks_out >= 4:
                profile.has_crippled_roster = True
                profile.crippled_player_name = ip.player
                profile.crippled_weeks_out = ip.weeks_out
                profile.crippled_ppg_lost = ip.ppg_before_injury
                break

        profiles[team_name] = profile

    return profiles


def _is_non_effectant_by_notes(notes: str) -> Tuple[bool, str]:
    """Check injury notes for signals that the player was out all/most of the season.

    Returns (is_non_effectant, reason).
    """
    notes_lower = notes.lower()

    # Pattern: "since <month>" where month is pre-season or early season
    import re
    since_match = re.search(r'since\s+(?:early\s+|mid[- ]?|late\s+)?(\w+)', notes_lower)
    if since_match:
        month_str = since_match.group(1)
        if month_str in _EARLY_ABSENCE_MONTHS:
            return True, f"out since {since_match.group(0)} — team stats already reflect absence"

    # "Season-ending" with no date could be ambiguous, but combined with
    # the poss check below it becomes safe.  Don't auto-filter on notes alone
    # unless we have a clear month signal.

    return False, ""


def _extract_ppg_from_notes(notes: str) -> float:
    """Extract PPG (points per game) from injury notes if present.

    Looks for patterns like "18.2 PPG", "22.5 ppg", etc.
    Returns 0.0 if not found.
    """
    import re
    match = re.search(r'(\d+\.?\d*)\s*ppg', notes.lower())
    if match:
        return float(match.group(1))
    return 0.0


def _extract_apg_from_notes(notes: str) -> float:
    """Extract APG (assists per game) from notes. Returns 0.0 if not found."""
    import re
    match = re.search(r'(\d+\.?\d*)\s*(?:apg|ast)', notes.lower())
    if match:
        return float(match.group(1))
    return 0.0


# PPG-based penalty floor: when a player's notes show high PPG but their
# PPG_PENALTY_FLOOR is defined in the Constants section at top of file.


def compute_severity_tier(ip: "InjuredPlayer") -> str:
    """Compute injury severity tier from status + play probabilities + notes.

    Tiers (from worst to best):
      OUT_SEASON   - out 4+ weeks, team identity has changed (e.g. Huff)
      OUT_GAME     - out for this game but recent injury (< 4 weeks)
      DOUBTFUL     - unlikely to play (< 30% chance)
      QUESTIONABLE - uncertain (30-65% chance)
      PROBABLE     - likely to play (> 65% chance)
    """
    status = ip.status.upper()
    r64_prob = ip.play_probs.get("R64", 0.5)

    if status == "OUT" or r64_prob == 0.0:
        # Distinguish season-long absence from recent injury
        weeks = _estimate_weeks_out(ip.notes)
        if weeks >= 4.0:
            return "OUT_SEASON"
        return "OUT_GAME"
    elif status == "DOUBTFUL" or r64_prob < 0.30:
        return "DOUBTFUL"
    elif status == "QUESTIONABLE" or r64_prob < 0.65:
        return "QUESTIONABLE"
    else:
        return "PROBABLE"


def _estimate_weeks_out(notes: str) -> float:
    """Estimate how many weeks a player has been out from injury notes.

    Looks for month references ("Jan", "Feb", etc.) and calculates
    weeks from that month to mid-March (tournament time).
    """
    import re
    notes_lower = notes.lower()

    # Month mapping to approximate weeks before mid-March tournament
    month_to_weeks = {
        "jan": 9, "january": 9,
        "feb": 5, "february": 5,
        "mar": 2, "march": 2,
        "dec": 12, "december": 12,
        "nov": 16, "november": 16,
        "oct": 20, "october": 20,
    }

    for month, weeks in month_to_weeks.items():
        if month in notes_lower:
            return weeks

    # Look for explicit timeline: "4-8 week", "6 weeks"
    timeline = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*week', notes_lower)
    if timeline:
        return (int(timeline.group(1)) + int(timeline.group(2))) / 2

    timeline_single = re.search(r'(\d+)\s*week', notes_lower)
    if timeline_single:
        return int(timeline_single.group(1))

    # "since" + month
    since_match = re.search(r'since\s+(?:early\s+|mid[- ]?|late\s+)?(\w+)', notes_lower)
    if since_match:
        month_str = since_match.group(1)
        if month_str in month_to_weeks:
            return month_to_weeks[month_str]

    # Default: if OUT status, assume at least 2 weeks
    return 2.0


def _extract_ppg_from_notes_v2(notes: str) -> float:
    """Extract PPG from notes, looking for patterns like '17.8 points per game'."""
    import re
    notes_lower = notes.lower()

    # Pattern: "X.X ppg" or "X.X points per game" or "averaging X.X"
    patterns = [
        r'(\d+\.?\d*)\s*ppg',
        r'(\d+\.?\d*)\s*points?\s*per\s*game',
        r'averaging\s+(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*pts',
    ]
    for pattern in patterns:
        match = re.search(pattern, notes_lower)
        if match:
            return float(match.group(1))
    return 0.0


# ── Severity tier multiplier for branch engine ────────────────────────────
SEVERITY_TIER_MULTIPLIER = {
    "OUT_SEASON":   1.00,  # Full branch trigger
    "OUT_GAME":     1.00,  # Full branch trigger
    "DOUBTFUL":     0.75,
    "QUESTIONABLE": 0.40,
    "PROBABLE":     0.10,
}


def _notes_suggest_contributor(notes: str) -> bool:
    """Check if injury notes contain signals the player was a real contributor.

    Looks for stat lines (PPG, REB, AST), 'key player', 'starter',
    'leading scorer', or recent injury dates (suggesting they played recently).
    """
    notes_lower = notes.lower()
    contributor_signals = [
        "ppg", "rpg", "apg", "reb", "ast", "pts",  # stat references
        "key player", "starter", "leading scorer", "second-leading",
        "best 3pt", "best 3-pt", "top scorer", "top-5 pick",
        "projected", "elite",
        "team went",  # "team went 2-4 after injury" = they mattered
    ]
    return any(signal in notes_lower for signal in contributor_signals)


def _find_player(injury_name: str, team_df: pd.DataFrame) -> Optional[pd.Series]:
    """Fuzzy match an injury name (e.g., 'C. Foster') to player data.

    Handles: 'C. Foster' -> 'Caleb Foster', 'M. Brown Jr.' -> 'Mikel Brown Jr.'
    """
    if team_df.empty:
        return None

    # Extract last name from injury format (e.g., "C. Foster" -> "foster")
    parts = injury_name.replace("Jr.", "").replace("II", "").replace("III", "").strip().split()
    if len(parts) >= 2:
        last_name = parts[-1].lower()
    elif len(parts) == 1:
        last_name = parts[0].lower().rstrip(".")
    else:
        return None

    # First initial
    first_initial = parts[0].rstrip(".").lower() if parts else ""

    for _, row in team_df.iterrows():
        player_name = str(row.get("name", "")).lower()
        player_parts = player_name.replace("jr.", "").replace("ii", "").replace("iii", "").strip().split()

        if len(player_parts) < 2:
            continue

        p_last = player_parts[-1]
        p_first = player_parts[0]

        # Match on last name + first initial
        if p_last == last_name:
            if first_initial and p_first.startswith(first_initial):
                return row
            elif not first_initial:
                return row

    # Fallback: partial last name match
    for _, row in team_df.iterrows():
        player_name = str(row.get("name", "")).lower()
        if last_name in player_name:
            return row

    return None


def _role_penalty(status: str) -> float:
    """Fallback penalty when no BPR data is available."""
    role_penalties = {
        "OUT": 0.15,
        "DOUBTFUL": 0.10,
        "QUESTIONABLE": 0.06,
        "EXPECTED": 0.02,
        "AVAILABLE": 0.0,
    }
    return role_penalties.get(status.upper(), 0.05)


def _fallback_role_based(injuries: List[InjuredPlayer]) -> Dict[str, TeamInjuryProfile]:
    """When no player data exists, use fixed role-based penalties.

    Still applies non-effectant filtering via notes (since we have no
    poss data, notes are the only signal available).
    """
    profiles: Dict[str, TeamInjuryProfile] = {}
    for ip in injuries:
        if ip.team not in profiles:
            profiles[ip.team] = TeamInjuryProfile(team_name=ip.team)

        # ── Non-effectant check (notes-only, no poss data) ────────
        ne_by_notes, ne_reason = _is_non_effectant_by_notes(ip.notes)
        if ne_by_notes:
            ip.non_effectant = True
            ip.non_effectant_reason = f"[fallback] {ne_reason}"
            ip.penalty = 0.0
        elif (ip.status == "OUT"
              and ip.play_probs.get("R64", 0) == 0.0
              and not _notes_suggest_contributor(ip.notes)):
            # Season-ending OUT with no contributor signals in notes
            ip.non_effectant = True
            ip.non_effectant_reason = (
                "[fallback] season-ending OUT with no contributor "
                "indicators in notes — likely never played"
            )
            ip.penalty = 0.0
        else:
            ip.penalty = _role_penalty(ip.status)
            # PPG floor: ensure minimum penalty when notes show high PPG
            ppg = _extract_ppg_from_notes(ip.notes)
            if ppg > 0:
                for threshold in sorted(PPG_PENALTY_FLOOR.keys(), reverse=True):
                    if ppg >= threshold:
                        floor = PPG_PENALTY_FLOOR[threshold]
                        if ip.penalty < floor:
                            ip.penalty = floor
                        break

        profiles[ip.team].injured_players.append(ip)

    # Recompute totals excluding non-effectant
    for profile in profiles.values():
        effectant = [ip for ip in profile.injured_players if not ip.non_effectant]
        profile.total_penalty = sum(ip.penalty for ip in effectant)
        profile.has_star_carrier_out = False  # Can't detect without BPR data

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Team Parameter Degradation (Pre-Composite)
# ─────────────────────────────────────────────────────────────────────────────

def apply_injury_degradation(teams: List[Team],
                             profiles: Dict[str, TeamInjuryProfile],
                             round_name: str = "R64") -> List[Team]:
    """Degrade team parameters based on injury profiles for a given round.

    This modifies Team objects IN PLACE. Should be called BEFORE
    compute_team_strengths() in the pipeline.

    For the deterministic pipeline (matchup predictions), uses expected value:
      penalty × (1 - play_probability) for each injured player.

    For Monte Carlo, use get_mc_team_adjustment() instead (per-simulation coin flip).
    """
    for team in teams:
        profile = profiles.get(team.name)
        if profile is None:
            continue

        _apply_degradation_to_team(team, profile, round_name, stochastic=False)

    return teams


def _apply_degradation_to_team(team: Team, profile: TeamInjuryProfile,
                               round_name: str, stochastic: bool = False,
                               rng: Optional[np.random.Generator] = None) -> float:
    """Apply injury degradation to a single team.

    If stochastic=True (Monte Carlo mode), flip coins for questionable players.
    If stochastic=False (deterministic mode), use expected values.

    Returns the total adj_em penalty applied (for tracking/reporting).
    """
    total_adj_em_penalty = 0.0
    total_shooting_penalty = 0.0
    total_ast_penalty = 0.0
    total_defense_penalty = 0.0
    total_rebound_penalty = 0.0

    for ip in profile.injured_players:
        if ip.non_effectant:
            continue  # Absence already baked into team stats

        play_prob = ip.play_probs.get(round_name, 0.0)

        if stochastic and rng is not None:
            # Monte Carlo: binary -- player plays or doesn't
            plays = rng.random() < play_prob
            if plays:
                # Player plays -- check if returning from injury (ramp-up)
                if play_prob < 0.80:
                    # Returning player: not at full effectiveness
                    effectiveness = min(RAMPUP_CAP, RAMPUP_BASE + RAMPUP_INCREMENT)
                    # Apply partial penalty (1 - effectiveness)
                    adj_penalty = ip.penalty * (1.0 - effectiveness)
                else:
                    adj_penalty = 0.0  # Healthy enough
            else:
                # Player is out: full penalty
                adj_penalty = ip.penalty
        else:
            # Deterministic: expected penalty = penalty × (1 - play_prob)
            adj_penalty = ip.penalty * (1.0 - play_prob)

        if adj_penalty <= 0:
            continue

        # Distribute penalty across team parameters based on player role
        # AdjEM: always affected (primary metric)
        em_share = adj_penalty * 5.5  # Scale BPR-based penalty to AdjEM units (calibrated from 3.5->4.5->5.5 via SOS-adjusted MASH backtest)
        total_adj_em_penalty += em_share

        # Offense-specific degradation based on OBPR vs DBPR
        if ip.obpr > ip.dbpr:
            # Offensive player: degrade shooting, assists more
            off_ratio = ip.obpr / max(ip.obpr + ip.dbpr, 0.1)
            total_shooting_penalty += adj_penalty * off_ratio * 0.4
            total_ast_penalty += adj_penalty * off_ratio * 0.3
        else:
            # Defensive player: degrade defense, rebounding more
            def_ratio = ip.dbpr / max(ip.obpr + ip.dbpr, 0.1)
            total_defense_penalty += adj_penalty * def_ratio * 0.4
            total_rebound_penalty += adj_penalty * def_ratio * 0.3

    # Cap the penalties
    total_adj_em_penalty = min(total_adj_em_penalty, MAX_ADJEM_PENALTY)

    # Apply to team
    team.adj_em -= total_adj_em_penalty
    team.adj_o -= total_adj_em_penalty * 0.6  # Offense takes 60% of AdjEM hit

    if total_shooting_penalty > 0:
        team.shooting_eff *= max(0.80, 1.0 - total_shooting_penalty * 0.15)
        team.efg_pct *= max(0.85, 1.0 - total_shooting_penalty * 0.10)

    if total_ast_penalty > 0:
        team.ast_pct *= max(0.80, 1.0 - total_ast_penalty * 0.15)

    if total_defense_penalty > 0:
        team.adj_d += total_defense_penalty * 2.0  # Higher AdjD = worse defense
        team.dvi *= max(0.75, 1.0 - total_defense_penalty * 0.20)
        team.rpi_rim *= max(0.75, 1.0 - total_defense_penalty * 0.15)

    if total_rebound_penalty > 0:
        team.rbm -= total_rebound_penalty * 1.5
        team.orb_pct *= max(0.85, 1.0 - total_rebound_penalty * 0.10)
        team.drb_pct *= max(0.85, 1.0 - total_rebound_penalty * 0.10)

    # Update profile for reporting
    profile.adj_em_penalty = total_adj_em_penalty
    profile.shooting_penalty = total_shooting_penalty
    profile.ast_penalty = total_ast_penalty
    profile.defense_penalty = total_defense_penalty
    profile.rebound_penalty = total_rebound_penalty

    return total_adj_em_penalty


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Monte Carlo Round-by-Round Integration
# ─────────────────────────────────────────────────────────────────────────────

def build_injury_mc_prob_func(base_prob_func, profiles: Dict[str, TeamInjuryProfile],
                              round_name: str = "R64",
                              rng: Optional[np.random.Generator] = None):
    """Wrap a base prob_func with injury adjustments for a specific round.

    For Monte Carlo: called once per round with a new round_name.
    Uses stochastic player availability (coin flips per simulation).
    """
    # Pre-compute expected team penalties for this round (deterministic fallback)
    _round_cache: Dict[str, float] = {}

    def _injury_adjusted_prob(team_a: Team, team_b: Team) -> float:
        """Get base probability, then adjust for injury impact."""
        p_base = base_prob_func(team_a, team_b)

        # Compute injury-based adjustments
        penalty_a = _get_round_penalty(team_a.name, profiles, round_name)
        penalty_b = _get_round_penalty(team_b.name, profiles, round_name)

        # Convert penalties to probability adjustments
        # A team with higher penalty should have lower win probability
        net_penalty = penalty_a - penalty_b  # positive = team_a more hurt

        if abs(net_penalty) < 0.001:
            return p_base

        # Shift probability: each unit of net penalty shifts ~2% toward opponent
        shift = net_penalty * 0.02
        p_adjusted = p_base - shift
        p_adjusted = np.clip(p_adjusted, 0.02, 0.98)

        return p_adjusted

    return _injury_adjusted_prob


def compute_star_isolation(profiles: Dict[str, TeamInjuryProfile],
                           player_df: pd.DataFrame) -> Dict[str, float]:
    """Compute Star Isolation Index (SII) for each team.

    SII = (top_player_bpr_share) * (top_player_poss_share)

    This catches the BYU/Dybantsa problem: when a team's #2 option is
    injured, the star becomes the ONLY option. Predictable and fatigued.

    Returns: dict of team_name → SII value.  Also adds extra penalty to
    the team's profile if SII exceeds thresholds.
    """
    sii_results = {}

    if player_df.empty:
        return sii_results

    for team_name, profile in profiles.items():
        # Find healthy top player's dominance on this team
        team_players = player_df[player_df["team"] == team_name]
        if team_players.empty:
            continue

        # Get BPR and poss columns
        if "bpr" not in team_players.columns or "poss" not in team_players.columns:
            continue

        team_total_bpr = team_players["bpr"].sum()
        team_total_poss = team_players["poss"].sum()

        if team_total_bpr <= 0 or team_total_poss <= 0:
            continue

        # Exclude injured players (OUT status) from the "available" pool
        out_players = {ip.player for ip in profile.injured_players
                       if ip.status == "OUT" and not ip.non_effectant}

        available = team_players[~team_players["name"].isin(out_players)]
        if available.empty:
            continue

        # Find the top remaining player by BPR
        top_idx = available["bpr"].idxmax()
        top_bpr = available.loc[top_idx, "bpr"]
        top_poss = available.loc[top_idx, "poss"]

        # SII = share of remaining production × share of remaining usage
        remaining_bpr = available["bpr"].sum()
        remaining_poss = available["poss"].sum()

        if remaining_bpr <= 0 or remaining_poss <= 0:
            continue

        bpr_share = top_bpr / remaining_bpr
        poss_share = top_poss / remaining_poss
        sii = bpr_share * poss_share

        sii_results[team_name] = sii

        # Apply penalty if SII exceeds thresholds
        if sii >= STAR_ISOLATION_THRESHOLD_SEVERE:
            profile.adj_em_penalty += STAR_ISOLATION_PENALTY_SEVERE
        elif sii >= STAR_ISOLATION_THRESHOLD_MILD:
            profile.adj_em_penalty += STAR_ISOLATION_PENALTY_MILD

    return sii_results


def _get_round_penalty(team_name: str, profiles: Dict[str, TeamInjuryProfile],
                       round_name: str) -> float:
    """Get the expected injury penalty for a team in a specific round."""
    profile = profiles.get(team_name)
    if profile is None:
        return 0.0

    total = 0.0
    for ip in profile.injured_players:
        if ip.non_effectant:
            continue  # Absence already baked into team stats
        play_prob = ip.play_probs.get(round_name, 0.0)
        total += ip.penalty * (1.0 - play_prob)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Usage Vacuum & Star Dependency (Post-Ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def apply_star_vacuum_penalty(matchups: List[Matchup],
                              profiles: Dict[str, TeamInjuryProfile],
                              round_name: str = "R64") -> List[Matchup]:
    """Apply post-ensemble penalty for teams with star carriers out.

    When a star carrier (>30% BPR share) is out, the Usage Vacuum
    compounds the degradation: remaining players take more shots at
    lower efficiency.

    Args:
        matchups: Matchups with ensemble probabilities already computed.
        profiles: TeamInjuryProfile dict with star carrier flags.
        round_name: Current round for play probability lookup.
    """
    for m in matchups:
        # Check team A
        vacuum_a = _compute_vacuum_penalty(m.team_a.name, profiles, round_name)
        vacuum_b = _compute_vacuum_penalty(m.team_b.name, profiles, round_name)

        if vacuum_a > 0 or vacuum_b > 0:
            p = m.win_prob_a_ensemble

            # Vacuum on A reduces A's probability
            if vacuum_a > 0:
                p = p * (1.0 - vacuum_a)

            # Vacuum on B increases A's probability
            if vacuum_b > 0:
                p = 1.0 - (1.0 - p) * (1.0 - vacuum_b)

            m.win_prob_a_ensemble = np.clip(p, 0.02, 0.98)

    return matchups


def _compute_vacuum_penalty(team_name: str,
                            profiles: Dict[str, TeamInjuryProfile],
                            round_name: str) -> float:
    """Compute the Usage Vacuum penalty for a team.

    Returns a probability multiplier (0 = no penalty, 0.15 = 15% prob reduction).
    """
    profile = profiles.get(team_name)
    if profile is None or not profile.has_star_carrier_out:
        return 0.0

    max_vacuum = 0.0
    for ip in profile.injured_players:
        if not ip.is_star_carrier:
            continue

        play_prob = ip.play_probs.get(round_name, 0.0)
        out_prob = 1.0 - play_prob

        if out_prob < 0.10:
            continue  # Player very likely to play, skip

        # Vacuum penalty scales with how much of a star carrier they are
        # and how poor the replacement is
        excess_share = ip.bpr_share - STAR_CARRIER_THRESHOLD
        replacement_gap = 1.0 - ip.replacement_factor

        # Base vacuum: scales with excess BPR share × replacement gap × out probability
        vacuum = excess_share * replacement_gap * out_prob * 0.50
        vacuum = min(vacuum, 0.15)  # Cap at 15% probability reduction

        max_vacuum = max(max_vacuum, vacuum)

    return max_vacuum


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def injury_impact_report(profiles: Dict[str, TeamInjuryProfile],
                         round_name: str = "R64") -> str:
    """Generate human-readable injury impact report."""
    lines = [
        "=" * 70,
        "  INJURY IMPACT REPORT",
        f"  Round: {round_name}",
        "=" * 70,
        "",
    ]

    # Sort by total penalty (most impacted first)
    sorted_profiles = sorted(profiles.values(),
                             key=lambda p: _get_round_penalty(p.team_name, profiles, round_name),
                             reverse=True)

    for profile in sorted_profiles:
        total_pen = _get_round_penalty(profile.team_name, profiles, round_name)
        if total_pen < 0.01:
            continue

        has_collapse = any(ip.collapse_risk for ip in profile.injured_players
                           if not ip.non_effectant)
        star_flag = " *** STAR CARRIER OUT ***" if profile.has_star_carrier_out else ""
        collapse_flag = " *** COLLAPSE RISK ***" if has_collapse else ""
        lines.append(f"  {profile.team_name}{star_flag}{collapse_flag}")
        lines.append(f"  Expected AdjEM penalty: -{total_pen * 3.5:.1f}")

        for ip in profile.injured_players:
            play_prob = ip.play_probs.get(round_name, 0.0)
            status_str = f"[{ip.status}]"
            prob_str = f"Play%: {play_prob:.0%}"
            bpr_str = f"BPR: {ip.bpr:.1f}" if ip.bpr > 0 else ""
            share_str = f"Share: {ip.bpr_share:.0%}" if ip.bpr_share > 0 else ""
            carrier = " STAR-CARRIER" if ip.is_star_carrier else ""
            cat_str = (f"Leads {ip.categories_led} cats (x{ip.multi_cat_amplifier:.2f})"
                       if ip.categories_led >= 2 else "")

            parts = [f"    {ip.player} ({ip.position})", status_str, prob_str]
            if bpr_str:
                parts.append(bpr_str)
            if share_str:
                parts.append(share_str)
            if carrier:
                parts.append(carrier)
            if cat_str:
                parts.append(cat_str)
            if ip.collapse_risk:
                parts.append("COLLAPSE-RISK")
            lines.append("  ".join(parts))

            if ip.notes:
                lines.append(f"      {ip.notes}")

        lines.append("")

    if len([p for p in sorted_profiles
            if _get_round_penalty(p.team_name, profiles, round_name) >= 0.01]) == 0:
        lines.append("  No significant injury impacts detected for this round.")

    # ── Non-effectant players (filtered out) ──────────────────────────
    non_effectant_all = []
    for profile in profiles.values():
        for ip in profile.injured_players:
            if ip.non_effectant:
                non_effectant_all.append((profile.team_name, ip))

    if non_effectant_all:
        lines.append("")
        lines.append("-" * 70)
        lines.append("  NON-EFFECTANT (filtered — absence already in team baseline):")
        lines.append("-" * 70)
        for team_name, ip in sorted(non_effectant_all, key=lambda x: x[0]):
            lines.append(f"    {team_name}: {ip.player} [{ip.status}]")
            lines.append(f"      Reason: {ip.non_effectant_reason}")

    return "\n".join(lines)


def injury_matchup_flags(matchups: List[Matchup],
                         profiles: Dict[str, TeamInjuryProfile],
                         round_name: str = "R64",
                         threshold: float = 0.03) -> str:
    """Flag matchups where injuries shift win probability by > threshold.

    Returns formatted string for inclusion in main output.
    """
    lines = [
        "=" * 70,
        "  INJURY-AFFECTED MATCHUPS",
        f"  (showing matchups where injuries shift probability by >{threshold:.0%})",
        "=" * 70,
        "",
    ]

    flagged = []
    for m in matchups:
        pen_a = _get_round_penalty(m.team_a.name, profiles, round_name)
        pen_b = _get_round_penalty(m.team_b.name, profiles, round_name)
        net_shift = abs(pen_a - pen_b) * 0.02  # probability shift

        if net_shift > threshold:
            flagged.append((m, pen_a, pen_b, net_shift))

    flagged.sort(key=lambda x: x[3], reverse=True)

    for m, pen_a, pen_b, shift in flagged:
        more_hurt = m.team_a.name if pen_a > pen_b else m.team_b.name
        lines.append(f"  ({m.team_a.seed}) {m.team_a.name} vs ({m.team_b.seed}) {m.team_b.name}")
        lines.append(f"    Ensemble prob: {m.win_prob_a_ensemble:.1%}")
        lines.append(f"    Injury shift: ~{shift:.1%} toward {m.team_b.name if pen_a > pen_b else m.team_a.name}")

        # List affected players
        for team_name in [m.team_a.name, m.team_b.name]:
            profile = profiles.get(team_name)
            if profile:
                for ip in profile.injured_players:
                    if ip.play_probs.get(round_name, 1.0) < 0.90:
                        lines.append(f"    ⚠ {ip.player} ({team_name}) [{ip.status}] - {ip.injury_type}")
        lines.append("")

    if not flagged:
        lines.append("  No matchups significantly affected by injuries.")

    return "\n".join(lines)
