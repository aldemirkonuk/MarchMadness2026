"""Tournament box-score loader: reads live results, computes per-team four-factors
and trajectory slopes, and exposes them for upstream blending and scenario engine.

Data source: data/live_results/tournament_box_scores.csv
"""

import os
import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TournamentGameStats:
    """Box score stats for one team in one game."""
    round: str = ""
    date: str = ""
    opponent: str = ""
    score: int = 0
    opp_score: int = 0
    won: bool = False
    h1_score: int = 0
    h1_opp_score: int = 0
    fg: int = 0
    fga: int = 0
    fg_pct: float = 0.0
    three_made: int = 0
    three_att: int = 0
    three_pct: float = 0.0
    ft: int = 0
    fta: int = 0
    ft_pct: float = 0.0
    oreb: int = 0
    dreb: int = 0
    total_reb: int = 0
    opp_dreb: int = 0
    ast: int = 0
    tov: int = 0
    stl: int = 0
    blk: int = 0
    pf: int = 0
    pts_in_paint: int = 0
    fast_break_pts: int = 0
    second_chance_pts: int = 0
    bench_pts: int = 0


@dataclass
class TeamTournamentProfile:
    """Aggregated tournament performance for one team."""
    team: str = ""
    games: List[TournamentGameStats] = field(default_factory=list)
    n_games: int = 0

    efg: float = 0.0
    tov_pct: float = 0.0
    orb_pct: float = 0.0
    ftr: float = 0.0
    ast_rate: float = 0.0
    paint_pct: float = 0.0
    bench_pct: float = 0.0

    # Trajectory slopes (per-game change)
    traj_fg_pct: float = 0.0
    traj_ast: float = 0.0
    traj_tov: float = 0.0
    traj_paint: float = 0.0
    traj_bench: float = 0.0
    n_accelerating: int = 0


def _safe_int(val: str, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_fg(val: str):
    """Parse '31-60' into (made, attempted)."""
    if not val or '-' not in val:
        return 0, 0
    parts = val.split('-')
    return _safe_int(parts[0]), _safe_int(parts[1])


def _slope(values: List[float]) -> float:
    """Simple linear regression slope."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def load_tournament_box_scores(base_dir: str = None) -> Dict[str, TeamTournamentProfile]:
    """Load tournament box scores and compute per-team profiles.

    Returns dict of team_name -> TeamTournamentProfile.
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    path = os.path.join(base_dir, "data", "live_results", "tournament_box_scores.csv")
    if not os.path.isfile(path):
        return {}

    # Parse all rows into per-team game lists
    team_games: Dict[str, List[TournamentGameStats]] = {}

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_a = _safe_int(row.get('score_a'))
            score_b = _safe_int(row.get('score_b'))
            winner = row.get('winner', '').strip()
            if score_a == 0 and score_b == 0:
                continue

            team_a = row.get('team_a', '').strip()
            team_b = row.get('team_b', '').strip()
            rnd = row.get('round', '').strip()
            date = row.get('date', '').strip()

            h1_a = _safe_int(row.get('h1_score_a'))
            h1_b = _safe_int(row.get('h1_score_b'))
            fg_pct_a = _safe_float(row.get('fg_pct_a'))
            fg_pct_b = _safe_float(row.get('fg_pct_b'))
            oreb_a = _safe_int(row.get('oreb_a'))
            oreb_b = _safe_int(row.get('oreb_b'))
            dreb_a = _safe_int(row.get('dreb_a'))
            dreb_b = _safe_int(row.get('dreb_b'))
            ast_a = _safe_int(row.get('ast_a'))
            ast_b = _safe_int(row.get('ast_b'))
            tov_a = _safe_int(row.get('tov_a'))
            tov_b = _safe_int(row.get('tov_b'))
            stl_a = _safe_int(row.get('stl_a'))
            stl_b = _safe_int(row.get('stl_b'))
            blk_a = _safe_int(row.get('blk_a'))
            blk_b = _safe_int(row.get('blk_b'))
            pf_a = _safe_int(row.get('pf_a'))
            pf_b = _safe_int(row.get('pf_b'))
            paint_a = _safe_int(row.get('pts_in_paint_a'))
            paint_b = _safe_int(row.get('pts_in_paint_b'))
            fb_a = _safe_int(row.get('fast_break_pts_a'))
            fb_b = _safe_int(row.get('fast_break_pts_b'))
            sc_a = _safe_int(row.get('second_chance_pts_a'))
            sc_b = _safe_int(row.get('second_chance_pts_b'))
            bench_a = _safe_int(row.get('bench_pts_a'))
            bench_b = _safe_int(row.get('bench_pts_b'))

            three_made_a = _safe_int(row.get('three_pt_made_a'))
            three_att_a = _safe_int(row.get('three_pt_att_a'))
            three_pct_a = _safe_float(row.get('three_pt_pct_a'))
            three_made_b = _safe_int(row.get('three_pt_made_b'))
            three_att_b = _safe_int(row.get('three_pt_att_b'))
            three_pct_b = _safe_float(row.get('three_pt_pct_b'))

            ft_a = _safe_int(row.get('ft_made_a'))
            fta_a = _safe_int(row.get('ft_att_a'))
            ft_b = _safe_int(row.get('ft_made_b'))
            fta_b = _safe_int(row.get('ft_att_b'))

            fg_a = _safe_int(row.get('fg_made_a', 0))
            fga_a = _safe_int(row.get('fg_att_a', 0))
            fg_b = _safe_int(row.get('fg_made_b', 0))
            fga_b = _safe_int(row.get('fg_att_b', 0))

            # If fg/fga not directly available, try to derive from fg_pct + score
            # (tournament_box_scores.csv may store fg_pct but not fg/fga directly)
            if fga_a == 0 and fg_pct_a > 0 and score_a > 0:
                # Estimate: score = 2*FG + 3PT + FT => FG = (score - 3PT - FT) / 2 roughly
                pass  # We'll use fg_pct directly when computing eFG

            has_box_data = (oreb_a > 0 or ast_a > 0 or tov_a > 0 or paint_a > 0)

            if team_a and has_box_data:
                game_a = TournamentGameStats(
                    round=rnd, date=date, opponent=team_b,
                    score=score_a, opp_score=score_b,
                    won=winner == team_a,
                    h1_score=h1_a, h1_opp_score=h1_b,
                    fg_pct=fg_pct_a,
                    three_made=three_made_a, three_att=three_att_a, three_pct=three_pct_a,
                    ft=ft_a, fta=fta_a,
                    oreb=oreb_a, dreb=dreb_a, total_reb=oreb_a + dreb_a,
                    opp_dreb=dreb_b,
                    ast=ast_a, tov=tov_a, stl=stl_a, blk=blk_a, pf=pf_a,
                    pts_in_paint=paint_a, fast_break_pts=fb_a,
                    second_chance_pts=sc_a, bench_pts=bench_a,
                )
                team_games.setdefault(team_a, []).append(game_a)

            has_box_data_b = (oreb_b > 0 or ast_b > 0 or tov_b > 0 or paint_b > 0)
            if team_b and has_box_data_b:
                game_b = TournamentGameStats(
                    round=rnd, date=date, opponent=team_a,
                    score=score_b, opp_score=score_a,
                    won=winner == team_b,
                    h1_score=h1_b, h1_opp_score=h1_a,
                    fg_pct=fg_pct_b,
                    three_made=three_made_b, three_att=three_att_b, three_pct=three_pct_b,
                    ft=ft_b, fta=fta_b,
                    oreb=oreb_b, dreb=dreb_b, total_reb=oreb_b + dreb_b,
                    opp_dreb=dreb_a,
                    ast=ast_b, tov=tov_b, stl=stl_b, blk=blk_b, pf=pf_b,
                    pts_in_paint=paint_b, fast_break_pts=fb_b,
                    second_chance_pts=sc_b, bench_pts=bench_b,
                )
                team_games.setdefault(team_b, []).append(game_b)

    # Build profiles
    profiles: Dict[str, TeamTournamentProfile] = {}

    for team, games in team_games.items():
        games.sort(key=lambda g: g.date)
        profile = TeamTournamentProfile(team=team, games=games, n_games=len(games))

        total_fg_pct = [g.fg_pct for g in games if g.fg_pct > 0]
        total_ast = [g.ast for g in games]
        total_tov = [g.tov for g in games]
        total_paint = [g.pts_in_paint for g in games if g.score > 0]
        total_bench = [g.bench_pts for g in games if g.score > 0]
        total_score = [g.score for g in games]

        if total_fg_pct:
            avg_fg_pct = sum(total_fg_pct) / len(total_fg_pct)
            avg_3pt_pct = sum(g.three_pct for g in games if g.three_pct > 0) / max(1, sum(1 for g in games if g.three_pct > 0))
            profile.efg = avg_fg_pct + 0.5 * (avg_3pt_pct / 100.0 if avg_3pt_pct > 1 else avg_3pt_pct) * 0.3

        if total_tov and total_score:
            total_t = sum(total_tov)
            est_poss = sum(total_score) * 0.88
            profile.tov_pct = total_t / max(1, est_poss + total_t)

        total_orb = sum(g.oreb for g in games)
        total_opp_drb = sum(g.opp_dreb for g in games)
        if total_orb + total_opp_drb > 0:
            profile.orb_pct = total_orb / (total_orb + total_opp_drb)

        total_fta = sum(g.fta for g in games)
        est_fga = sum(g.score / max(0.01, g.fg_pct / 100.0 if g.fg_pct > 1 else g.fg_pct * 100 / 100.0)
                       for g in games if g.fg_pct > 0 and g.score > 0)
        if est_fga > 0:
            profile.ftr = total_fta / est_fga

        total_ast_sum = sum(total_ast)
        if total_fg_pct and sum(total_score) > 0:
            est_fg_made = sum(g.score * (g.fg_pct / 100.0 if g.fg_pct > 1 else g.fg_pct) / 2.0
                             for g in games if g.fg_pct > 0 and g.score > 0)
            if est_fg_made > 0:
                profile.ast_rate = total_ast_sum / est_fg_made

        if total_paint and total_score:
            profile.paint_pct = sum(total_paint) / sum(total_score)

        if total_bench and total_score:
            profile.bench_pct = sum(total_bench) / sum(total_score)

        # Trajectory slopes
        if len(games) >= 2:
            profile.traj_fg_pct = _slope([g.fg_pct for g in games if g.fg_pct > 0]) if total_fg_pct else 0.0
            profile.traj_ast = _slope([float(g.ast) for g in games])
            profile.traj_tov = _slope([float(g.tov) for g in games])
            profile.traj_paint = _slope([float(g.pts_in_paint) for g in games if g.score > 0]) if total_paint else 0.0
            profile.traj_bench = _slope([float(g.bench_pts) for g in games if g.score > 0]) if total_bench else 0.0

            n_accel = 0
            if profile.traj_fg_pct > 2.0:
                n_accel += 1
            if profile.traj_ast > 1.5:
                n_accel += 1
            if profile.traj_tov < -1.0:
                n_accel += 1
            if profile.traj_paint > 2.0:
                n_accel += 1
            if profile.traj_bench > 1.0:
                n_accel += 1
            profile.n_accelerating = n_accel

        profiles[team] = profile

    return profiles


def compute_comeback_rates(profiles: Dict[str, TeamTournamentProfile],
                           teams=None) -> Dict[str, tuple]:
    """Compute comeback rate for each team from tournament + season data.

    Returns dict of team -> (comeback_rate, games_trailing_at_half).
    comeback_rate = fraction of trailing-at-half games that were won.
    """
    rates: Dict[str, tuple] = {}

    for team, p in profiles.items():
        trailing_games = 0
        comeback_wins = 0
        for g in p.games:
            if g.h1_score > 0 and g.h1_opp_score > 0:
                if g.h1_score < g.h1_opp_score:
                    trailing_games += 1
                    if g.won:
                        comeback_wins += 1

        # Supplement with season-level proxy: q3_adj_strength > 0 means team
        # improves from H1 to H2 (a correlate of comeback ability).
        if teams:
            t_obj = next((t for t in teams.values() if t.name == team), None)
            if t_obj and hasattr(t_obj, 'q3_adj_strength') and t_obj.q3_adj_strength > 1.5:
                trailing_games += 3
                comeback_wins += 2

        if trailing_games > 0:
            rates[team] = (comeback_wins / trailing_games, trailing_games)
        else:
            rates[team] = (0.0, 0)

    return rates


def tournament_profile_report(profiles: Dict[str, TeamTournamentProfile]) -> str:
    """Print tournament profile summary for all teams with data."""
    lines = []
    lines.append("=" * 80)
    lines.append("  TOURNAMENT BOX-SCORE PROFILES")
    lines.append("=" * 80)

    for team, p in sorted(profiles.items()):
        if p.n_games == 0:
            continue
        lines.append(f"\n  {team} ({p.n_games} tournament games):")
        lines.append(f"    eFG: {p.efg:.3f} | TOV%: {p.tov_pct:.3f} | ORB%: {p.orb_pct:.3f} | FTR: {p.ftr:.3f}")
        lines.append(f"    AST rate: {p.ast_rate:.2f} | Paint%: {p.paint_pct:.1%} | Bench%: {p.bench_pct:.1%}")

        if p.n_games >= 2:
            accel_label = f"{p.n_accelerating}/5 stats ACCELERATING" if p.n_accelerating >= 2 else "stable"
            lines.append(f"    Trajectory: FG% {p.traj_fg_pct:+.1f}/game, AST {p.traj_ast:+.1f}, "
                        f"TOV {p.traj_tov:+.1f}, Paint {p.traj_paint:+.1f}, Bench {p.traj_bench:+.1f}")
            lines.append(f"    Assessment: {accel_label}")

        for g in p.games:
            marker = "W" if g.won else "L"
            lines.append(f"      {g.round} {g.date}: {marker} {g.score}-{g.opp_score} vs {g.opponent} "
                        f"(FG={g.fg_pct:.1f}%, AST={g.ast}, TOV={g.tov}, REB={g.total_reb}, Paint={g.pts_in_paint})")

    return "\n".join(lines)
