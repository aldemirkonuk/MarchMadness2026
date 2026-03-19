"""Player matchup sandbox -- team-level player impact parameters.

Computes 7 core player-based parameters, 1 style_clash parameter,
P&R tactical counter metrics, and 5 matchup flags from EvanMiya
player BPR data.

The P&R proxy (big_man_offense, rim_defense_bpr) IS integrated into
the ensemble prob_func as a small z-adjustment. All other sandbox
params remain display-only until manual review confirms value.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from src.models import Team


def load_player_data() -> pd.DataFrame:
    """Load EvanMiya player data with team name normalization."""
    import os
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _path = os.path.join(_base, "data", "EvanMiya_Players.csv")
    try:
        df = pd.read_csv(_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

    name_map = {
        "Michigan State": "Michigan St.",
        "Iowa State": "Iowa St.",
        "Ohio State": "Ohio St.",
        "North Carolina State": "North Carolina St.",
        "North Dakota State": "North Dakota St.",
        "Utah State": "Utah St.",
        "McNeese State": "McNeese St.",
        "Wright State": "Wright St.",
        "Tennessee State": "Tennessee St.",
        "Kennesaw State": "Kennesaw St.",
    }
    df["team"] = df["team"].replace(name_map)
    return df


def compute_pnr_metrics(teams: list, player_df: pd.DataFrame) -> None:
    """Populate big_man_offense and rim_defense_bpr on each Team object.

    Heuristic for identifying "bigs": among the top 8 BPR players on a team,
    those with DBPR > OBPR and DBPR > 3.0 are classified as defensive bigs
    (rim protectors / roll men).  We take the top-2 by DBPR.

    big_man_offense = sum of OBPR for top-2 bigs (how much offense comes
                      from big-man actions like P&R rolls)
    rim_defense_bpr = sum of DBPR for top-2 bigs (rim protection quality
                      beyond the team-level rpi_rim stat)
    """
    for team in teams:
        players = player_df[player_df["team"] == team.name].nlargest(8, "bpr")
        if players.empty:
            team.big_man_offense = 0.0
            team.rim_defense_bpr = 0.0
            continue

        bigs = players[(players["dbpr"] > players["obpr"]) & (players["dbpr"] > 3.0)]
        bigs = bigs.nlargest(2, "dbpr")

        if bigs.empty:
            team.big_man_offense = 0.0
            team.rim_defense_bpr = 0.0
        else:
            team.big_man_offense = float(bigs["obpr"].sum())
            team.rim_defense_bpr = float(bigs["dbpr"].sum())


def compute_player_matchup_params(team_a: Team, team_b: Team,
                                   player_df: pd.DataFrame
                                   ) -> Dict[str, float]:
    """Compute all 8 player matchup parameters for team_a vs team_b.

    Returns dict with keys:
        star_bpr_mismatch, defensive_pressure, depth_advantage,
        two_way_threat, bpr_concentration, star_carry_ratio,
        star_vs_opp_defense, style_clash_a, style_clash_b
    """
    pa = player_df[player_df["team"] == team_a.name].nlargest(8, "bpr")
    pb = player_df[player_df["team"] == team_b.name].nlargest(8, "bpr")

    if pa.empty or pb.empty:
        return _default_params()

    # Top player stats
    star_a = pa.iloc[0] if len(pa) > 0 else None
    star_b = pb.iloc[0] if len(pb) > 0 else None

    top3_a = pa.head(3)
    top3_b = pb.head(3)

    all_a = pa
    all_b = pb

    # 1. star_bpr_mismatch: gap between best players
    bpr_a = star_a["bpr"] if star_a is not None else 0
    bpr_b = star_b["bpr"] if star_b is not None else 0
    star_bpr_mismatch = bpr_a - bpr_b

    # 2. defensive_pressure: A's best defender vs B's best scorer
    best_def_a = pa.nlargest(1, "dbpr")["dbpr"].iloc[0] if len(pa) > 0 else 0
    best_off_b = pb.nlargest(1, "obpr")["obpr"].iloc[0] if len(pb) > 0 else 0
    best_def_b = pb.nlargest(1, "dbpr")["dbpr"].iloc[0] if len(pb) > 0 else 0
    best_off_a = pa.nlargest(1, "obpr")["obpr"].iloc[0] if len(pa) > 0 else 0
    defensive_pressure = (best_def_a - best_off_b + best_def_b - best_off_a) / 2.0

    # 3. depth_advantage: sum of BPR beyond top 3
    bench_bpr_a = all_a.iloc[3:]["bpr"].sum() if len(all_a) > 3 else 0
    bench_bpr_b = all_b.iloc[3:]["bpr"].sum() if len(all_b) > 3 else 0
    depth_advantage = bench_bpr_a - bench_bpr_b

    # 4. two_way_threat: count of players with OBPR > 3 AND DBPR > 2
    two_way_a = len(all_a[(all_a["obpr"] > 3.0) & (all_a["dbpr"] > 2.0)])
    two_way_b = len(all_b[(all_b["obpr"] > 3.0) & (all_b["dbpr"] > 2.0)])
    two_way_threat = two_way_a - two_way_b

    # 5. bpr_concentration: how top-heavy is the team?
    total_bpr_a = all_a["bpr"].sum()
    total_bpr_b = all_b["bpr"].sum()
    if total_bpr_a > 0 and total_bpr_b > 0:
        conc_a = top3_a["bpr"].sum() / total_bpr_a
        conc_b = top3_b["bpr"].sum() / total_bpr_b
    else:
        conc_a = conc_b = 0.5
    bpr_concentration = conc_a - conc_b

    # 6. star_carry_ratio: star BPR / team average BPR
    avg_a = all_a["bpr"].mean() if len(all_a) > 0 else 1
    avg_b = all_b["bpr"].mean() if len(all_b) > 0 else 1
    carry_a = bpr_a / max(avg_a, 0.1)
    carry_b = bpr_b / max(avg_b, 0.1)
    star_carry_ratio = carry_a - carry_b

    # 7. star_vs_opp_defense: A's star OBPR - B's best DBPR
    star_vs_def_a = (star_a["obpr"] if star_a is not None else 0) - best_def_b
    star_vs_def_b = (star_b["obpr"] if star_b is not None else 0) - best_def_a
    star_vs_opp_defense = star_vs_def_a - star_vs_def_b

    # 8. style_clash: offensive vs defensive identity mismatch
    off_identity_a = top3_a["obpr"].mean() if len(top3_a) > 0 else 0
    def_identity_a = top3_a["dbpr"].mean() if len(top3_a) > 0 else 0
    off_identity_b = top3_b["obpr"].mean() if len(top3_b) > 0 else 0
    def_identity_b = top3_b["dbpr"].mean() if len(top3_b) > 0 else 0

    style_clash_a = off_identity_a - def_identity_b
    style_clash_b = off_identity_b - def_identity_a

    return {
        "star_bpr_mismatch": star_bpr_mismatch,
        "defensive_pressure": defensive_pressure,
        "depth_advantage": depth_advantage,
        "two_way_threat": float(two_way_threat),
        "bpr_concentration": bpr_concentration,
        "star_carry_ratio": star_carry_ratio,
        "star_vs_opp_defense": star_vs_opp_defense,
        "style_clash_a": style_clash_a,
        "style_clash_b": style_clash_b,
    }


def _default_params() -> Dict[str, float]:
    return {k: 0.0 for k in [
        "star_bpr_mismatch", "defensive_pressure", "depth_advantage",
        "two_way_threat", "bpr_concentration", "star_carry_ratio",
        "star_vs_opp_defense", "style_clash_a", "style_clash_b",
    ]}


def compute_matchup_flags(team_a: Team, team_b: Team,
                           params: Dict[str, float],
                           player_df: pd.DataFrame
                           ) -> List[str]:
    """Generate matchup flags for manual review.

    Flag types:
    1. Low star BPR gap with large seed gap
    2. Favorite lacks two-way player, underdog has one
    3. Favorite has high BPR concentration
    4. Underdog's defender matches up well against favorite's scorer
    """
    flags = []
    fav = team_a if team_a.seed < team_b.seed else team_b
    dog = team_b if team_a.seed < team_b.seed else team_a
    seed_gap = abs(team_a.seed - team_b.seed)

    fav_players = player_df[player_df["team"] == fav.name].nlargest(5, "bpr")
    dog_players = player_df[player_df["team"] == dog.name].nlargest(5, "bpr")

    if fav_players.empty or dog_players.empty:
        return flags

    fav_star_bpr = fav_players.iloc[0]["bpr"] if len(fav_players) > 0 else 0
    dog_star_bpr = dog_players.iloc[0]["bpr"] if len(dog_players) > 0 else 0

    # Flag 1: Small star gap despite large seed gap
    bpr_gap = abs(fav_star_bpr - dog_star_bpr)
    if seed_gap >= 5 and bpr_gap < 2.0:
        flags.append(
            f"FLAG-1: Small star gap ({bpr_gap:.1f} BPR) despite {seed_gap}-seed spread. "
            f"{dog.name}'s {dog_players.iloc[0]['name']} ({dog_star_bpr:.1f}) "
            f"vs {fav.name}'s {fav_players.iloc[0]['name']} ({fav_star_bpr:.1f})"
        )

    # Flag 2: Favorite lacks two-way, underdog has one
    fav_two_way = len(fav_players[(fav_players["obpr"] > 3) & (fav_players["dbpr"] > 2)])
    dog_two_way = len(dog_players[(dog_players["obpr"] > 3) & (dog_players["dbpr"] > 2)])
    if fav_two_way == 0 and dog_two_way >= 1:
        tw_name = dog_players[(dog_players["obpr"] > 3) & (dog_players["dbpr"] > 2)].iloc[0]["name"]
        flags.append(
            f"FLAG-2: {fav.name} has NO two-way threats; "
            f"{dog.name} has {dog_two_way} (e.g. {tw_name})"
        )

    # Flag 3: Favorite's BPR is top-heavy (concentration > 0.7)
    total_fav = fav_players["bpr"].sum()
    if total_fav > 0:
        top3_share = fav_players.head(3)["bpr"].sum() / total_fav
        if top3_share > 0.70 and seed_gap >= 4:
            flags.append(
                f"FLAG-3: {fav.name} BPR is top-heavy ({top3_share:.0%} in top 3). "
                f"If star neutralized, depth drops off sharply."
            )

    # Flag 4: Underdog's best defender vs favorite's best scorer
    fav_best_scorer = fav_players.nlargest(1, "obpr").iloc[0]
    dog_best_defender = dog_players.nlargest(1, "dbpr").iloc[0]
    if dog_best_defender["dbpr"] > fav_best_scorer["obpr"] * 0.5:
        flags.append(
            f"FLAG-4: {dog.name}'s {dog_best_defender['name']} "
            f"(DBPR={dog_best_defender['dbpr']:.1f}) could disrupt "
            f"{fav.name}'s {fav_best_scorer['name']} "
            f"(OBPR={fav_best_scorer['obpr']:.1f})"
        )

    # Flag 5: P&R hard counter -- team's big-man offense vs opponent rim defense
    bmo_a = getattr(team_a, "big_man_offense", 0.0)
    bmo_b = getattr(team_b, "big_man_offense", 0.0)
    rd_a = getattr(team_a, "rim_defense_bpr", 0.0)
    rd_b = getattr(team_b, "rim_defense_bpr", 0.0)
    if bmo_a > 5.0 and rd_b > 8.0:
        flags.append(
            f"FLAG-5: P&R HARD COUNTER -- {team_a.name}'s big-man offense "
            f"(OBPR sum={bmo_a:.1f}) vs {team_b.name}'s rim defense "
            f"(DBPR sum={rd_b:.1f}). Primary engine may be neutralized."
        )
    if bmo_b > 5.0 and rd_a > 8.0:
        flags.append(
            f"FLAG-5: P&R HARD COUNTER -- {team_b.name}'s big-man offense "
            f"(OBPR sum={bmo_b:.1f}) vs {team_a.name}'s rim defense "
            f"(DBPR sum={rd_a:.1f}). Primary engine may be neutralized."
        )

    return flags


def player_matchup_report(matchups, teams: List[Team]) -> str:
    """Generate the full player matchup sandbox report for all R64 matchups."""
    player_df = load_player_data()
    if player_df.empty:
        return "  Player data not available."

    lines = [
        "=" * 70,
        "  PLAYER MATCHUP SANDBOX (8 params + style_clash + P&R + flags)",
        "  NOTE: P&R metrics integrated into ensemble; other params display-only.",
        "=" * 70,
        "",
    ]

    for m in matchups:
        params = compute_player_matchup_params(m.team_a, m.team_b, player_df)
        flags = compute_matchup_flags(m.team_a, m.team_b, params, player_df)

        lines.append(f"  ({m.team_a.seed}) {m.team_a.name} vs ({m.team_b.seed}) {m.team_b.name}")
        lines.append(f"    star_mismatch={params['star_bpr_mismatch']:+.2f}  "
                      f"def_pressure={params['defensive_pressure']:+.2f}  "
                      f"depth={params['depth_advantage']:+.2f}")
        lines.append(f"    two_way={params['two_way_threat']:+.0f}  "
                      f"concentration={params['bpr_concentration']:+.3f}  "
                      f"carry_ratio={params['star_carry_ratio']:+.2f}")
        lines.append(f"    star_vs_def={params['star_vs_opp_defense']:+.2f}  "
                      f"style_A={params['style_clash_a']:+.2f}  "
                      f"style_B={params['style_clash_b']:+.2f}")

        if flags:
            for f in flags:
                lines.append(f"    >> {f}")
        lines.append("")

    flagged_count = sum(1 for m in matchups for f in
                        compute_matchup_flags(m.team_a, m.team_b,
                                              compute_player_matchup_params(m.team_a, m.team_b, player_df),
                                              player_df)
                        if f)
    lines.append(f"  Total flags generated: {flagged_count}")

    return "\n".join(lines)
