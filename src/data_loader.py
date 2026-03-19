"""Data collection pipeline: loads archive CSVs and builds Team objects."""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.models import Team
from src.equations import (
    seed_score, three_point_reliability, coaching_factor, legacy_factor,
    defensive_versatility, chaos_index,
)
from src.utils import canonical_name, safe_float, SEED_EXPECTED_ROUND

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(BASE_DIR, "archive")
ARCHIVE3 = os.path.join(BASE_DIR, "archive-3")
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_csv(directory: str, filename: str) -> Optional[pd.DataFrame]:
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError(
            f"\n\n❌  EMPTY DATA FILE: {filename}\n"
            f"   Path: {path}\n"
            f"   The file exists but contains no data (0 bytes).\n"
            f"   Fix: Re-export this file from your data source (KenPom/Barttorvik)\n"
            f"        or restore it from a backup.\n"
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # File exists but is unreadable (corrupted, wrong format, or macOS quarantine).
        # Return None so callers can fall back to defaults instead of crashing.
        import warnings
        warnings.warn(
            f"Unreadable data file: {filename} ({path}) — "
            f"{size} bytes but pandas cannot parse. Using defaults. "
            f"Re-export from your data source to restore full data.",
            UserWarning,
            stacklevel=2,
        )
        return None


def _build_coach_map() -> Dict[str, str]:
    """Map team name -> coach name from archive."""
    df = _load_csv(ARCHIVE, "REF _ Current NCAAM Coaches (2026).csv")
    if df is None:
        return {}
    coach_map = {}
    for _, row in df.iterrows():
        team = str(row.get("Join Team", "")).strip()
        coach = str(row.get("Current Coach", "")).strip()
        if team and coach:
            coach_map[canonical_name(team)] = coach
    return coach_map


def _build_coach_results() -> Dict[str, dict]:
    """Coach tournament performance from archive-3."""
    df = _load_csv(ARCHIVE3, "Coach Results.csv")
    if df is None:
        return {}
    results = {}
    for _, row in df.iterrows():
        name = str(row["COACH"]).strip()
        results[name] = {
            "wins": int(safe_float(row.get("W", 0))),
            "games": int(safe_float(row.get("GAMES", 0))),
            "pake": safe_float(row.get("PAKE", 0)),
            "f4_pct": safe_float(row.get("F4%", "0").replace("%", "")) / 100,
        }
    return results


def _build_team_legacy(current_year: int = 2026) -> Dict[str, float]:
    """Seed-normalized, decay-weighted PASE from Tournament Matchups.

    Three corrections over the old raw-PASE approach:
      1. Difficulty coefficient: normalize outperformance by the max possible
         for that seed so 1-seeds aren't structurally penalized.
      2. Coaching decay: 0.85^(years_ago) so stale runs fade.
      3. sqrt(N) normalization + cap [-3, +3].
    """
    from math import sqrt

    ROUND_TO_WINS = {64: 0, 32: 1, 16: 2, 8: 3, 4: 4, 2: 5, 1: 6}

    tm = _load_csv(ARCHIVE3, "Tournament Matchups.csv")
    if tm is None:
        return {}

    cols = ["YEAR", "TEAM", "SEED", "ROUND"]
    df = tm[cols].dropna().copy()
    df["YEAR"] = df["YEAR"].astype(int)
    df["SEED"] = df["SEED"].astype(int)
    df["ROUND"] = df["ROUND"].astype(int)

    # Keep deepest round per (team, year) — smallest ROUND value = deepest.
    df = (
        df.sort_values(["YEAR", "TEAM", "ROUND"])
        .groupby(["YEAR", "TEAM"], as_index=False)
        .first()
    )

    team_appearances: Dict[str, list] = {}
    for _, row in df.iterrows():
        year = int(row["YEAR"])
        seed = int(row["SEED"])
        actual_wins = ROUND_TO_WINS.get(int(row["ROUND"]), 0)
        expected = SEED_EXPECTED_ROUND.get(seed, 0.0)
        max_pase = 6.0 - expected
        if max_pase < 0.5:
            max_pase = 0.5

        normalized_outperf = (actual_wins - expected) / max_pase
        decay = 0.85 ** max(0, current_year - year)

        team_name = canonical_name(str(row["TEAM"]).strip())
        team_appearances.setdefault(team_name, []).append(
            (normalized_outperf, decay)
        )

    legacy: Dict[str, float] = {}
    for team, appearances in team_appearances.items():
        n = len(appearances)
        if n == 0:
            legacy[team] = 0.0
            continue
        weighted_sum = sum(outperf * decay for outperf, decay in appearances)
        weight_total = sum(decay for _, decay in appearances)
        if weight_total > 0:
            weighted_avg = weighted_sum / weight_total
        else:
            weighted_avg = 0.0
        pase = weighted_avg * sqrt(n)
        legacy[team] = max(-3.0, min(3.0, pase))

    return legacy


def build_season_h2h() -> Dict[tuple, float]:
    """Build head-to-head season results from game logs.

    Returns {(canonical_a, canonical_b): avg_margin_for_a} for every
    pair of tournament-eligible teams that met this season.  The margin
    is signed: positive means team_a won by that many points on average.
    """
    import glob as _glob
    logs_dir = os.path.join(ARCHIVE3, "game-logs")
    if not os.path.isdir(logs_dir):
        return {}

    raw: Dict[tuple, list] = {}
    for fpath in _glob.glob(os.path.join(logs_dir, "*.csv")):
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if df.empty:
            continue
        team = canonical_name(str(df.iloc[0]["team"]).strip())
        for _, row in df.iterrows():
            opp = canonical_name(str(row["opponent"]).strip())
            try:
                margin = float(row["score_t"]) - float(row["score_o"])
            except (ValueError, TypeError, KeyError):
                continue
            raw.setdefault((team, opp), []).append(margin)

    h2h: Dict[tuple, float] = {}
    seen = set()
    for (a, b), margins in raw.items():
        pair = tuple(sorted([a, b]))
        if pair in seen:
            continue
        seen.add(pair)
        rev = raw.get((b, a), [])
        all_margins = margins + [-m for m in rev]
        if all_margins:
            avg = sum(all_margins) / len(all_margins)
            h2h[(a, b)] = avg
            h2h[(b, a)] = -avg
    return h2h


def _build_resume_data() -> Dict[str, dict]:
    """Resume/quality wins from archive-3."""
    df = _load_csv(ARCHIVE3, "Resumes.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    resumes = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        q1w = safe_float(row.get("Q1 W", 0))
        q1q2w = safe_float(row.get("Q1 PLUS Q2 W", 0))
        total = q1q2w + safe_float(row.get("Q3 Q4 L", 0))
        resumes[team] = {
            "q1_wins": q1w,
            "q1q2_wins": q1q2w,
            "top50_games": max(total, 1),
            "top50_win_pct": q1q2w / max(total, 1),
        }
    return resumes


def _build_evan_miya() -> Dict[str, dict]:
    """Killshots and ratings from EvanMiya (archive-3)."""
    df = _load_csv(ARCHIVE3, "EvanMiya.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    miya = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        miya[team] = {
            "killshots": safe_float(row.get("KILLSHOTS PER GAME", 0)),
            "killshots_conceded": safe_float(row.get("KILL SHOTS CONCEDED PER GAME", 0)),
            "o_rate": safe_float(row.get("O RATE", 0)),
            "d_rate": safe_float(row.get("D RATE", 0)),
            "injury_rank": safe_float(row.get("INJURY RANK", 35)),
            "roster_rank": safe_float(row.get("ROSTER RANK", 35)),
        }
    return miya


def _build_evanmiya_players() -> Dict[str, dict]:
    """Real player BPR data from EvanMiya_Players.csv.

    Returns per-team: best_bpr, best_poss, team_avg_bpr, player_count.
    """
    path = os.path.join(DATA_DIR, "EvanMiya_Players.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    result = {}
    for team_name, grp in df.groupby("team"):
        cname = canonical_name(str(team_name).strip())
        bprs = grp["bpr"].astype(float)
        poss = grp["poss"].astype(float)
        best_idx = bprs.idxmax()
        result[cname] = {
            "best_bpr": float(bprs.loc[best_idx]),
            "best_poss": float(poss.loc[best_idx]),
            "best_player": str(grp.loc[best_idx, "name"]),
            "team_avg_bpr": float(bprs.mean()),
            "player_count": len(grp),
        }
    return result


def _build_evanmiya_teams() -> Dict[str, dict]:
    """Real team-level data from EvanMiya_Teams.csv.

    Has rank_inj, roster_rank, runs_margin, opponent_adjust, pace_adjust.
    """
    path = os.path.join(DATA_DIR, "EvanMiya_Teams.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    result = {}
    for _, row in df.iterrows():
        cname = canonical_name(str(row["team"]).strip())
        result[cname] = {
            "rank_inj": safe_float(row.get("rank_inj", 35)),
            "roster_rank": safe_float(row.get("roster_rank", 35)),
            "runs_per_game": safe_float(row.get("runs_per_game", 0)),
            "runs_conceded_per_game": safe_float(row.get("runs_conceded_per_game", 0)),
            "runs_margin": safe_float(row.get("runs_margin", 0)),
            "opponent_adjust": safe_float(row.get("opponent_adjust", 0)),
            "pace_adjust": safe_float(row.get("pace_adjust", 0)),
            "team_bpr": safe_float(row.get("bpr", 0)),
            "wins": safe_float(row.get("wins", 0)),
            "losses": safe_float(row.get("losses", 0)),
        }
    return result


def _build_game_log_stats() -> Dict[str, dict]:
    """Compute real momentum and clutch from per-game logs in archive-3/game-logs/.

    Returns per-team: last10_wpct, close_game_wpct, games_played.
    """
    log_dir = os.path.join(ARCHIVE3, "game-logs")
    if not os.path.isdir(log_dir):
        return {}
    result = {}
    for fn in os.listdir(log_dir):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(log_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "team" not in df.columns:
            continue
        team_name = canonical_name(str(df.iloc[0]["team"]).strip())
        n = len(df)

        # Last 10 games record
        last10 = df.tail(10)
        last10_wins = (last10["result"] == "W").sum()
        last10_wpct = last10_wins / max(len(last10), 1)

        # Close games (decided by <= 5 points) + scoring margin variance
        margin_std = 0.0
        close_wpct = 0.5
        close_count = 0
        if "score_t" in df.columns and "score_o" in df.columns:
            signed_margins = df["score_t"].astype(float) - df["score_o"].astype(float)
            margins = signed_margins.abs()
            close_mask = margins <= 5
            close_games = df[close_mask]
            close_count = int(close_mask.sum())
            if len(close_games) > 0:
                close_wins = (close_games["result"] == "W").sum()
                close_wpct = close_wins / len(close_games)

            if len(signed_margins) >= 3:
                margin_std = float(signed_margins.std())

        result[team_name] = {
            "last10_wpct": float(last10_wpct),
            "close_game_wpct": float(close_wpct),
            "close_game_count": close_count,
            "games_played": n,
            "margin_std": margin_std,
        }
    return result


def _build_location_data() -> Dict[str, dict]:
    """Travel distance from archive-3 Tournament Locations."""
    df = _load_csv(ARCHIVE3, "Tournament Locations.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    locations = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        cur_round = safe_float(row.get("CURRENT ROUND", 1))
        if cur_round == 1:
            locations[team] = {
                "distance_mi": safe_float(row.get("DISTANCE (MI)", 500)),
            }
    return locations


def _build_shooting_splits() -> Dict[str, dict]:
    """Rim protection data from Shooting Splits."""
    df = _load_csv(ARCHIVE3, "Shooting Splits.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    splits = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        splits[team] = {
            "dunks_fgd": safe_float(row.get("DUNKS FG%D", 0)) / 100,
            "close_twos_fgd": safe_float(row.get("CLOSE TWOS FG%D", 0)) / 100,
            "opp_rim_fg": (safe_float(row.get("DUNKS FG%D", 0)) +
                           safe_float(row.get("CLOSE TWOS FG%D", 0))) / 200,
        }
    return splits


def _build_bench_exp() -> Dict[str, dict]:
    """Bench scoring % and experience from KenPom Height."""
    df = _load_csv(ARCHIVE, "INT _ KenPom _ Height.csv")
    if df is None:
        return {}
    df_2026 = df[df["Season"] == 2026]
    data = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TeamName"]).strip())
        data[team] = {
            "bench_pct": safe_float(row.get("Bench", 25)) / 100,
            "experience": safe_float(row.get("Experience", 2.0)),
        }
    return data


def _build_misc_stats() -> Dict[str, dict]:
    """Steal rate and other misc stats from KenPom Miscellaneous."""
    df = _load_csv(ARCHIVE, "INT _ KenPom _ Miscellaneous Team Stats.csv")
    if df is None:
        return {}
    df_2026 = df[df["Season"] == 2026]
    data = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TeamName"]).strip())
        data[team] = {
            "stl_rate": safe_float(row.get("StlRate", 0)) / 100,
            "opp_fg3_pct": safe_float(row.get("OppFG3Pct", 33)) / 100,
        }
    return data


def _build_teamsheet_ranks() -> Dict[str, dict]:
    """NET rating, KPI, SOR, BPI from Teamsheet Ranks."""
    df = _load_csv(ARCHIVE3, "Teamsheet Ranks.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    data = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        q1w = int(safe_float(row.get("Q1 W", 0)))
        q1l = int(safe_float(row.get("Q1 L", 0)))
        q3w = int(safe_float(row.get("Q3 W", 0)))
        q3l = int(safe_float(row.get("Q3 L", 0)))
        q4w = int(safe_float(row.get("Q4 W", 0)))
        q4l = int(safe_float(row.get("Q4 L", 0)))
        data[team] = {
            "net_rating": int(safe_float(row.get("NET", 50))),
            "kpi": int(safe_float(row.get("KPI", 50))),
            "sor": int(safe_float(row.get("SOR", 50))),
            "bpi": int(safe_float(row.get("BPI", 50))),
            "quality_avg": safe_float(row.get("QUALITY AVG", 50)),
            "q1a_wins": int(safe_float(row.get("Q1A W", 0))),
            "q1a_losses": int(safe_float(row.get("Q1A L", 0))),
            "q1_wins": q1w,
            "q1_losses": q1l,
            "q1_record": q1w / max(q1w + q1l, 1),
            "q3_losses": q3l,
            "q4_losses": q4l,
            "q34_games": q3w + q3l + q4w + q4l,
            "q34_loss_rate": (q3l + q4l) / max(q3w + q3l + q4w + q4l, 1),
        }
    return data


def _build_teamrankings() -> Dict[str, dict]:
    """Momentum, consistency, and neutral court data from TeamRankings."""
    df = _load_csv(ARCHIVE3, "TeamRankings.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    data = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        data[team] = {
            "tr_rating": safe_float(row.get("TR RATING", 10)),
            "tr_last_rank": int(safe_float(row.get("LAST", 50))),
            "tr_hi_rank": int(safe_float(row.get("HI", 50))),
            "tr_lo_rank": int(safe_float(row.get("LO", 100))),
            "sos_last_rank": int(safe_float(row.get("SOS LAST", 50))),
            "consistency_rating": safe_float(row.get("CONSISTENCY TR RATING", 10)),
        }

    # Also load neutral court data
    ndf = _load_csv(ARCHIVE3, "TeamRankings Neutral.csv")
    if ndf is not None:
        ndf_2026 = ndf[ndf["YEAR"] == 2026]
        for _, row in ndf_2026.iterrows():
            team = canonical_name(str(row["TEAM"]).strip())
            if team in data:
                data[team]["neutral_rating"] = safe_float(row.get("TR RATING", 10))
            else:
                data[team] = {"neutral_rating": safe_float(row.get("TR RATING", 10))}
    return data


def _build_z_ratings() -> Dict[str, float]:
    """Z Rating from archive-3. 2026 not available, so derive from 2025 coefficients.

    Z Rating = f(AdjEM, SOS) -- correlates r=0.903 with AdjEM and r=0.830 with SOS.
    Using 2025 data to calibrate the linear fit, then apply to 2026 teams.
    """
    df = _load_csv(ARCHIVE3, "Z Rating Teams.csv")
    if df is None:
        return {}

    # Try 2026 first
    df_2026 = df[df["YEAR"] == 2026]
    if len(df_2026) > 0:
        ratings = {}
        for _, row in df_2026.iterrows():
            team = canonical_name(str(row["TEAM"]).strip())
            ratings[team] = safe_float(row.get("Z RATING", 10))
        return ratings

    # No 2026 data: derive from AdjEM + SOS using 2025 coefficients
    # From correlation analysis: Z ≈ 0.45 * AdjEM + 0.35 * SOS + 3.0
    return {}


def _build_public_picks() -> Dict[str, dict]:
    """Public pick % as sentiment proxy."""
    df = _load_csv(ARCHIVE3, "Public Picks.csv")
    if df is None:
        return {}
    df_2026 = df[df["YEAR"] == 2026]
    picks = {}
    for _, row in df_2026.iterrows():
        team = canonical_name(str(row["TEAM"]).strip())
        picks[team] = {
            "r64_pick": safe_float(row.get("R64", 50)) / 100,
            "champ_pick": safe_float(row.get("FINALS", 0)) / 100,
        }
    return picks


def load_all_teams() -> List[Team]:
    """Master data loader: build all 68 Team objects from archive data."""
    kb = _load_csv(ARCHIVE3, "KenPom Barttorvik.csv")
    if kb is None:
        raise FileNotFoundError("KenPom Barttorvik.csv not found in archive-3")

    kb_2026 = kb[(kb["YEAR"] == 2026) & (kb["SEED"].notna()) & (kb["SEED"] > 0)].copy()

    coach_map = _build_coach_map()
    coach_results = _build_coach_results()
    team_legacy = _build_team_legacy()
    resumes = _build_resume_data()
    evan_miya = _build_evan_miya()
    locations = _build_location_data()
    shooting = _build_shooting_splits()
    public_picks = _build_public_picks()
    bench_exp = _build_bench_exp()
    misc_stats = _build_misc_stats()
    teamsheet = _build_teamsheet_ranks()
    teamrankings = _build_teamrankings()
    z_ratings = _build_z_ratings()
    em_players = _build_evanmiya_players()
    em_teams = _build_evanmiya_teams()
    game_log_stats = _build_game_log_stats()

    matchups_df = pd.read_csv(os.path.join(DATA_DIR, "matchups.csv"))
    team_region_map = {}
    for _, row in matchups_df.iterrows():
        team_region_map[canonical_name(str(row["team_a"]).strip())] = row["region"]
        team_region_map[canonical_name(str(row["team_b"]).strip())] = row["region"]

    teams = []
    for _, row in kb_2026.iterrows():
        raw_name = str(row["TEAM"]).strip()
        name = canonical_name(raw_name)
        sd = int(row["SEED"])

        t = Team(
            name=name,
            seed=sd,
            region=team_region_map.get(name, ""),
        )

        # Tier 1
        t.adj_em = safe_float(row.get("KADJ EM", 0))
        t.adj_o = safe_float(row.get("KADJ O", 0))
        t.adj_d = safe_float(row.get("KADJ D", 0))
        t.efg_pct = safe_float(row.get("EFG%", 0)) / 100
        t.to_pct = safe_float(row.get("TOV%", 0)) / 100
        t.orb_pct = safe_float(row.get("OREB%", 0)) / 100
        t.drb_pct = safe_float(row.get("DREB%", 0)) / 100
        t.opp_to_pct = safe_float(row.get("TOV%D", 0)) / 100
        t.seed_score = seed_score(sd)

        # Shooting
        t.three_p_pct = safe_float(row.get("3PT%", 0)) / 100
        three_ptr = safe_float(row.get("3PTR", 0)) / 100
        t.three_pa_fga = three_ptr
        t.three_pri = three_point_reliability(t.three_p_pct, three_ptr)
        t.ft_pct = safe_float(row.get("FT%", 0)) / 100
        ftr_raw = safe_float(row.get("FTR", 0)) / 100
        t.fta_fga = ftr_raw
        t.ftr = t.ft_pct * ftr_raw

        # 2PT data for scoring balance
        t.two_pt_pct = safe_float(row.get("2PT%", 50)) / 100
        t.two_pt_rate = safe_float(row.get("2PTR", 50)) / 100

        # Scoring balance: rewards teams efficient from BOTH inside and outside
        t.scoring_balance = (t.two_pt_pct * t.two_pt_rate +
                             t.three_p_pct * t.three_pa_fga)

        # TS% approximation: eFG% adjusted for FT
        t.ts_pct = t.efg_pct * 0.85 + t.ftr * 0.15

        # Merged shooting efficiency (single param from eFG% + TS%)
        t.shooting_eff = 0.6 * t.efg_pct + 0.4 * t.ts_pct

        # Height in meters (EFF HGT = minutes-weighted, in inches)
        eff_hgt_inches = safe_float(row.get("EFF HGT", 80))
        t.eff_height = eff_hgt_inches * 0.0254  # convert to meters

        # SOS from Elite SOS
        t.sos = safe_float(row.get("ELITE SOS", 0))

        # Top-50 performance from Resumes
        res = resumes.get(name, {})
        t.top25_perf = res.get("top50_win_pct", 0.5)

        # Tier 2
        t.ast_pct = safe_float(row.get("AST%", 0)) / 100
        t.exp = safe_float(row.get("EXP", 2.0))
        t.blk_pct = safe_float(row.get("BLK%", 0)) / 100
        ms = misc_stats.get(name, {})
        t.stl_rate = ms.get("stl_rate", safe_float(row.get("BLKED%", 0)) / 100)
        t.opp_3p_pct = ms.get("opp_fg3_pct",
                               safe_float(row.get("3PT%D", 33)) / 100)

        # Rim protection from Shooting Splits
        ss = shooting.get(name, {})
        t.opp_fg_pct_rim = ss.get("opp_rim_fg", 0.5)

        # Pace
        t.pace = safe_float(row.get("BADJ T", safe_float(row.get("RAW T", 68))))

        # Record
        t.wins = int(safe_float(row.get("W", 0)))
        t.losses = int(safe_float(row.get("L", 0)))
        win_pct = t.wins / max(t.wins + t.losses, 1)

        # AdjEM rank
        t.adj_em_rank = int(safe_float(row.get("KADJ EM RANK", 50)))

        # Coaching Factor (Bayesian-smoothed; real coach tournament record)
        coach_name = coach_map.get(name, "")
        cr = coach_results.get(coach_name, {})
        t.ctf = coaching_factor(cr.get("wins", 0), cr.get("games", 0))

        # Legacy Factor (reformed PASE: difficulty-normalized, decay-weighted, capped [-3, +3])
        t.legacy_factor = team_legacy.get(name, 0.0)

        # EvanMiya: killshots + real runs_margin for MSRP
        em = evan_miya.get(name, {})
        emt = em_teams.get(name, {})
        t.killshots_per_game = em.get("killshots", 0)
        t.killshots_conceded = em.get("killshots_conceded", 0)
        # Prefer real runs_margin from EvanMiya_Teams.csv over killshots
        if emt.get("runs_margin", 0) != 0:
            t.msrp = emt["runs_margin"]
        else:
            t.msrp = t.killshots_per_game - t.killshots_conceded

        # Proximity
        loc = locations.get(name, {})
        dist = loc.get("distance_mi", 500)
        t.proximity = 1.0 / (1.0 + dist / 500.0)

        # BDS (bench scoring) from KenPom Height
        be = bench_exp.get(name, {})
        t.bds = be.get("bench_pct", 0.25)
        if be.get("experience"):
            t.exp = be["experience"]

        # ── SPI: REAL from EvanMiya player BPR ──
        emp = em_players.get(name)
        if emp:
            league_avg_bpr = 2.0  # avg D-I starter BPR ~2.0
            t.spi = (emp["best_bpr"] * emp["best_poss"] / 1500.0) / max(league_avg_bpr, 0.1)
        else:
            roster_rank_val = int(em.get("roster_rank", 35) if em else 35)
            talent_factor = max(0, (68 - roster_rank_val) / 68.0)
            t.spi = talent_factor * t.adj_em / 15.0

        # ── Q1/Q3 (old fake scores removed; real records set after ts load below) ──

        # ── Rebounding margin ──
        t.rbm = (t.orb_pct + t.drb_pct) / 2 - 0.5

        # Chaos Index
        t.three_pt_std = t.three_pa_fga * 0.08
        t.chaos_index = chaos_index(t.three_pa_fga, t.three_pt_std)

        # Blend SPI with public championship sentiment
        pp = public_picks.get(name, {})
        champ_pick = pp.get("champ_pick", 0.0)
        t.spi = max(t.spi, 0) + champ_pick * 0.5

        # NET rating, quadrant records from Teamsheet Ranks
        ts = teamsheet.get(name, {})
        t.net_rating = ts.get("net_rating", 50)
        q1a_w = ts.get("q1a_wins", 0)
        q1a_l = ts.get("q1a_losses", 0)
        if q1a_w + q1a_l > 0:
            t.top25_perf = max(t.top25_perf, q1a_w / (q1a_w + q1a_l))

        # Real quadrant records (replace fake AdjEM-derived Q1/Q3)
        t.q1_record = ts.get("q1_record", 0.0)
        t.q34_loss_rate = ts.get("q34_loss_rate", 0.0)

        # Injury rank: prefer real EvanMiya_Teams data, fallback to archive
        if emt.get("rank_inj", 0) > 0:
            t.injury_rank = int(emt["rank_inj"])
        else:
            t.injury_rank = int(em.get("injury_rank", 35) if em else 35)
        if emt.get("roster_rank", 0) > 0:
            t.roster_rank = int(emt["roster_rank"])
        else:
            t.roster_rank = int(em.get("roster_rank", 35) if em else 35)

        # Points per game = PPPO (points per possession) * pace (possessions per game)
        pppo = safe_float(row.get("PPPO", 0))
        pppd = safe_float(row.get("PPPD", 0))
        t.ppg = pppo * t.pace if pppo > 0 else t.adj_o * t.pace / 100.0
        t.opp_ppg = pppd * t.pace if pppd > 0 else t.adj_d * t.pace / 100.0

        # BARTHAG (probability of beating average D-I team)
        t.barthag = safe_float(row.get("BARTHAG", 0.5))

        # Best player above average: REAL from EvanMiya player BPR
        emp = em_players.get(name)
        if emp:
            t.best_player_above_avg_pts = emp["best_bpr"] - emp["team_avg_bpr"]
            t.best_player_above_avg_reb = t.orb_pct + t.drb_pct - 0.5
            t.best_player_above_avg_ast = t.ast_pct
        elif t.roster_rank > 0:
            t.best_player_above_avg_pts = max(0, (68 - t.roster_rank) / 68.0)
            t.best_player_above_avg_reb = t.orb_pct + t.drb_pct - 0.5
            t.best_player_above_avg_ast = t.ast_pct
        t.star_player_win_pct = win_pct

        # (SPI now computed for ALL teams above using roster_rank)

        # ── TeamRankings: real momentum, clutch, blowout resilience ──
        trk = teamrankings.get(name, {})
        t.tr_rating = trk.get("tr_rating", 10)
        t.tr_last_rank = trk.get("tr_last_rank", 50)
        t.tr_hi_rank = trk.get("tr_hi_rank", 50)
        t.tr_lo_rank = trk.get("tr_lo_rank", 100)
        t.sos_last_rank = trk.get("sos_last_rank", 50)
        t.consistency_rating = trk.get("consistency_rating", 10)
        t.neutral_rating = trk.get("neutral_rating", 10)

        # ── MOMENTUM: blend game-log last-10 + TeamRankings rank trend ──
        gl = game_log_stats.get(name, {})
        hi = t.tr_hi_rank
        lo = t.tr_lo_rank
        last = t.tr_last_rank
        rank_range = max(lo - hi, 1)
        momentum_rank = max(0, min(1, (lo - last) / rank_range))

        last10_wpct = gl.get("last10_wpct", win_pct)
        q1a_rate = q1a_w / max(q1a_w + q1a_l, 1) if q1a_w + q1a_l > 0 else 0.5
        sos_hard = t.sos > 30
        if sd >= 5 and sos_hard:
            t.momentum = 0.25 * momentum_rank + 0.35 * last10_wpct + 0.20 * win_pct + 0.20 * q1a_rate
        else:
            t.momentum = 0.30 * momentum_rank + 0.30 * last10_wpct + 0.20 * win_pct + 0.20 * q1a_rate

        # ── CLUTCH: REAL close-game win% from game logs + consistency ──
        close_wpct = gl.get("close_game_wpct", 0.5)
        close_count = gl.get("close_game_count", 0)
        killshots_margin = t.killshots_per_game - t.killshots_conceded
        consistency_norm = min(t.consistency_rating / 15.0, 1.0)
        killshots_norm = min(max((killshots_margin + 1) / 2.0, 0), 1)
        # Weight close_game_wpct more if we have enough close games
        close_weight = min(0.30, close_count * 0.05) if close_count > 0 else 0.0
        remaining = 1.0 - close_weight
        t.clutch_factor = (close_weight * close_wpct +
                           remaining * 0.30 * win_pct +
                           remaining * 0.25 * t.barthag +
                           remaining * 0.25 * consistency_norm +
                           remaining * 0.20 * killshots_norm)

        # Scoring margin variance from game logs (lower = more consistent)
        t.scoring_margin_std = gl.get("margin_std", 0.0)

        # BLOWOUT RESILIENCE from killshots + consistency
        t.blowout_resilience = min(1.0, max(0.0,
            0.30 * killshots_norm +
            0.30 * consistency_norm +
            0.20 * (t.adj_em / 40.0) +
            0.20 * t.bds
        ))

        # FOUL TROUBLE IMPACT: bench depth offsets star dependency
        t.foul_trouble_impact = -(1.0 - t.bds) * (t.spi / max(t.spi, 0.01))
        t.foul_trouble_impact = max(-1.0, min(0.0, t.foul_trouble_impact))

        # CWP: Star player 17+ at halftime -> win%
        star_dominance = max(0, (68 - t.roster_rank) / 68.0) if t.roster_rank > 0 else 0.5
        t.cwp_star_17_half_win_pct = (0.40 * t.barthag +
                                       0.30 * star_dominance +
                                       0.15 * q1a_rate +
                                       0.15 * win_pct)

        # ── CWP composites ──
        margin_normalized = min(max(t.adj_em / 30.0, 0), 1)
        # Fragility: blend of low margin, low consistency, and high variance
        # Lower = more fragile (inverted in normalization)
        t.fragility_score = max(0, min(1.0,
            0.35 * (1 - margin_normalized) +
            0.25 * (1 - min(t.consistency_rating / 15.0, 1.0)) +
            0.20 * (1 - win_pct) +
            0.20 * t.chaos_index * 10
        ))
        neutral_contrib = min(trk.get("neutral_rating", 10) / 30.0, 1.0) if trk else 0.0
        t.march_readiness = (0.20 * win_pct +
                             0.15 * min(win_pct + 0.1, 1.0) +
                             0.15 * (1 - t.three_pa_fga) +
                             0.10 * win_pct +
                             0.10 * t.drb_pct +
                             0.10 * (1 - t.opp_3p_pct) +
                             0.10 * neutral_contrib +
                             0.10 * t.cwp_star_17_half_win_pct)
        # versatility_score removed from weights (was just AdjEM/30 = redundant)

        teams.append(t)

    # SOS-adjust opp_to_pct and to_pct across the field
    avg_sos = sum(t.sos for t in teams) / len(teams) if teams else 1.0
    for t in teams:
        sos_ratio = t.sos / max(avg_sos, 1.0)
        # Credit teams that force TOs against tough opponents
        t.opp_to_pct = t.opp_to_pct * sos_ratio
        # Credit teams with low TO rate against tough opponents
        t.to_pct = t.to_pct / max(sos_ratio, 0.5)

    # Half-scoring data (REAL from StatSharp): H1 point diff, H2 adjustment
    half_df = _load_csv(ARCHIVE3, "HalfScoring.csv")
    half_map: Dict[str, dict] = {}
    if half_df is not None:
        for _, row in half_df.iterrows():
            tname = canonical_name(str(row.get("TEAM", "")))
            half_map[tname] = {
                "h1_pd": safe_float(row.get("H1_PD", 0)),
                "h2_pd": safe_float(row.get("H2_PD", 0)),
            }

    for t in teams:
        hd = half_map.get(canonical_name(t.name))
        if hd:
            t.offensive_burst = hd["h1_pd"]
            t.q3_adj_strength = hd["h2_pd"] - hd["h1_pd"]
        else:
            # Fallback proxy if team not found in StatSharp
            adj_o_vals = [x.adj_o for x in teams]
            ao_min, ao_max = min(adj_o_vals), max(adj_o_vals)
            pace_vals = [x.pace for x in teams]
            pa_min, pa_max = min(pace_vals), max(pace_vals)
            ao_n = (t.adj_o - ao_min) / max(ao_max - ao_min, 1)
            pa_n = (t.pace - pa_min) / max(pa_max - pa_min, 1)
            t.offensive_burst = 0.5 * ao_n + 0.3 * pa_n + 0.2 * t.three_pri
            ctf_n = t.ctf
            exp_n = min(t.exp / 3.0, 1.0)
            con_n = min(t.consistency_rating / 15.0, 1.0)
            t.q3_adj_strength = (0.4 * ctf_n + 0.3 * exp_n +
                                 0.2 * con_n + 0.1 * t.clutch_factor)

    # Derived scores that weights reference (set before normalization)
    for t in teams:
        t.net_score = max(0, (68 - t.net_rating) / 68.0) if t.net_rating > 0 else 0.5
        t.ppg_margin = t.ppg - t.opp_ppg
        # injury_health: lower rank = healthier = BETTER (inverted in normalization)
        t.injury_health = float(t.injury_rank)
        # star_above_avg: real BPR differential (best player BPR - team avg BPR)
        t.star_above_avg = t.best_player_above_avg_pts

        if t.name in z_ratings:
            t.z_rating = z_ratings[t.name]
        else:
            t.z_rating = 0.45 * t.adj_em + 0.35 * t.sos + 3.0

        t.cwp_star_17_half = t.cwp_star_17_half_win_pct
        t.consistency = t.consistency_rating

    # Now compute derived normalized metrics that need the full field
    _compute_dvi_and_rpi(teams)
    _compute_ranks(teams)

    return teams


def _compute_dvi_and_rpi(teams: List[Team]) -> None:
    """Compute DVI and RPI_rim using field-normalized stats."""
    blk_vals = np.array([t.blk_pct for t in teams])
    stl_vals = np.array([t.stl_rate for t in teams])
    opp3_vals = np.array([1 - t.opp_3p_pct for t in teams])
    rim_vals = np.array([1 - t.opp_fg_pct_rim for t in teams])

    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)

    blk_n = _norm(blk_vals)
    stl_n = _norm(stl_vals)
    opp3_n = _norm(opp3_vals)
    rim_n = _norm(rim_vals)

    for i, t in enumerate(teams):
        t.dvi = defensive_versatility(blk_n[i], stl_n[i], opp3_n[i])
        t.rpi_rim = float(blk_n[i] + rim_n[i]) / 2.0


def _compute_ranks(teams: List[Team]) -> None:
    """Compute AdjEM rank across tournament field."""
    sorted_teams = sorted(teams, key=lambda t: t.adj_em, reverse=True)
    for rank, t in enumerate(sorted_teams, 1):
        t.adj_em_rank = rank


def load_matchups(teams: List[Team], h2h_lookup: Dict[tuple, float] = None) -> List:
    """Load Round of 64 matchups from CSV."""
    from src.models import Matchup

    if h2h_lookup is None:
        h2h_lookup = build_season_h2h()

    df = pd.read_csv(os.path.join(DATA_DIR, "matchups.csv"))
    team_dict = {t.name: t for t in teams}

    matchups = []
    for _, row in df.iterrows():
        name_a = canonical_name(str(row["team_a"]).strip())
        name_b = canonical_name(str(row["team_b"]).strip())

        # Handle play-in games: use first listed team as placeholder
        if "/" in name_a:
            name_a = canonical_name(name_a.split("/")[0].strip())
        if "/" in name_b:
            name_b = canonical_name(name_b.split("/")[0].strip())

        ta = team_dict.get(name_a)
        tb = team_dict.get(name_b)

        if ta is None:
            for t in teams:
                if name_a.lower() in t.name.lower() or t.name.lower() in name_a.lower():
                    ta = t
                    break
        if tb is None:
            for t in teams:
                if name_b.lower() in t.name.lower() or t.name.lower() in name_b.lower():
                    tb = t
                    break

        if ta is None or tb is None:
            print(f"WARNING: Could not find teams for {name_a} vs {name_b}")
            continue

        m = Matchup(
            team_a=ta,
            team_b=tb,
            round_name="R64",
            region=str(row["region"]),
        )
        m.h2h_season_edge = h2h_lookup.get((ta.name, tb.name), 0.0)
        matchups.append(m)

    return matchups


def load_historical_games(start_year: int = 2010,
                          end_year: int = 2025) -> pd.DataFrame:
    """Load historical tournament data for XGBoost training."""
    kb = _load_csv(ARCHIVE3, "KenPom Barttorvik.csv")
    if kb is None:
        return pd.DataFrame()

    hist = kb[(kb["YEAR"] >= start_year) & (kb["YEAR"] <= end_year) &
              (kb["SEED"].notna()) & (kb["SEED"] > 0)].copy()

    tm = _load_csv(ARCHIVE3, "Tournament Matchups.csv")

    return hist, tm
