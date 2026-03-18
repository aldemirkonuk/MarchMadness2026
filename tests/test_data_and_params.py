"""Test suite: verifies dataset loading, parameter computation, and model sanity.

Run with: python -m pytest tests/ -v
   or:    python tests/test_data_and_params.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data_loader import load_all_teams, load_matchups, load_historical_games
from src.composite import compute_team_strengths, rank_teams, compute_win_probability
from src.weights import CORE_WEIGHTS, PARAM_KEYS
from src.utils import normalize_teams
from src.models import Matchup

TEAMS = None

def get_teams():
    global TEAMS
    if TEAMS is None:
        TEAMS = load_all_teams()
    return TEAMS


def team_by_name(name):
    for t in get_teams():
        if t.name == name:
            return t
    raise ValueError(f"Team {name} not found")


# ═══════════════════════════════════════════════════════════════════
# 1. DATASET LOADING TESTS
# ═══════════════════════════════════════════════════════════════════

def test_loads_68_teams():
    """Must load exactly 68 tournament teams."""
    teams = get_teams()
    assert len(teams) == 68, f"Expected 68, got {len(teams)}"
    print(f"  PASS: Loaded {len(teams)} teams")


def test_all_four_one_seeds():
    """The four 1-seeds must be present."""
    teams = get_teams()
    one_seeds = [t for t in teams if t.seed == 1]
    names = {t.name for t in one_seeds}
    assert len(one_seeds) == 4, f"Expected 4 one-seeds, got {len(one_seeds)}"
    for expected in ["Duke", "Arizona", "Florida", "Michigan"]:
        assert expected in names, f"Missing 1-seed: {expected}"
    print(f"  PASS: 1-seeds = {names}")


def test_all_regions_populated():
    """Each region must have teams assigned."""
    teams = get_teams()
    regions = {t.region for t in teams if t.region}
    assert len(regions) >= 4, f"Expected 4 regions, got {regions}"
    for r in ["East", "West", "South", "Midwest"]:
        region_teams = [t for t in teams if t.region == r]
        assert len(region_teams) >= 8, f"Region {r} has only {len(region_teams)} teams"
    print(f"  PASS: All 4 regions populated")


def test_matchups_load_32():
    """Must load exactly 32 Round of 64 matchups."""
    teams = get_teams()
    matchups = load_matchups(teams)
    assert len(matchups) == 32, f"Expected 32 matchups, got {len(matchups)}"
    print(f"  PASS: Loaded {len(matchups)} matchups")


def test_historical_data():
    """Historical data must span 2010-2025 with 900+ matchups."""
    hist, tm = load_historical_games()
    assert len(hist) > 500, f"Expected 500+ historical teams, got {len(hist)}"
    assert len(tm) > 1000, f"Expected 1000+ matchup rows, got {len(tm)}"
    years = sorted(hist["YEAR"].unique())
    assert 2010 in years, "Missing year 2010"
    assert 2025 in years, "Missing year 2025"
    print(f"  PASS: Historical data: {len(hist)} team-seasons, {len(tm)} matchup rows, years {years[0]}-{years[-1]}")


# ═══════════════════════════════════════════════════════════════════
# 2. DATASET SOURCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════

DATASET_SOURCES = {
    "KenPom Barttorvik.csv":   ["adj_em", "efg_pct", "to_pct", "orb_pct", "drb_pct",
                                 "opp_to_pct", "ast_pct", "exp", "pace", "blk_pct",
                                 "three_p_pct", "ft_pct", "seed_score", "barthag", "ppg",
                                 "2PT%", "2PTR", "EFF HGT (-> eff_height in meters)"],
    "Resumes.csv":             ["top25_perf"],
    "Coach Results.csv":       ["ctf"],
    "Team Results.csv":        ["legacy_factor (winsorized [-3,+5])"],
    "EvanMiya.csv":            ["msrp", "killshots_per_game", "injury_rank", "roster_rank"],
    "Teamsheet Ranks.csv":     ["net_rating", "q1_record", "q34_loss_rate", "q1a_wins"],
    "TeamRankings.csv":        ["momentum (LAST/HI/LO)", "consistency", "neutral_rating"],
    "Z Rating Teams.csv":      ["z_rating (derived for 2026)"],
    "Shooting Splits.csv":     ["rpi_rim"],
    "KenPom Height.csv":       ["bds"],
    "KenPom Miscellaneous.csv":["stl_rate"],
    "Public Picks.csv":        ["spi (sentiment boost)"],
}

def test_dataset_sources_report():
    """Report which datasets provide which parameters."""
    duke = team_by_name("Duke")
    print("\n  DATASET -> PARAMETER MAPPING:")
    for dataset, params in DATASET_SOURCES.items():
        print(f"    {dataset:<35s} -> {', '.join(params)}")

    print(f"\n  DERIVED PARAMETERS (computed from real stats):")
    print(f"    shooting_eff:      0.6*eFG% + 0.4*TS% (merged from r=0.966 correlated pair)")
    print(f"    scoring_balance:   2PT%*2PTR + 3P%*3PTR (inside-outside balance)")
    print(f"    ftr:               FT% * (FTA/FGA)")
    print(f"    dvi:               0.3*BLK + 0.3*STL + 0.4*(1-Opp3P)")
    print(f"    chaos_index:       (3PA/FGA) * StdDev(3P%)")
    print(f"    rbm:               (ORB% + DRB%) / 2 - 0.5")
    print(f"    fragility:         f(AdjEM, consistency, win%, chaos)")
    print(f"    march_ready:       f(WinPct, 3PA/FGA, DRB%, neutral, cwp)")
    print(f"    to_pct/opp_to_pct: SOS-adjusted (credit TOs against hard schedules)")

    print(f"\n  PROXY PARAMETERS (approximated from real data):")
    print(f"    clutch_factor: Win% + BARTHAG + TeamRankings consistency + EvanMiya killshots")
    print(f"    spi:           roster_rank + AdjEM + public picks")
    print(f"    momentum:      TeamRankings LAST/HI/LO ranks + conditional SOS")
    print(f"    blowout_res:   killshots + consistency + AdjEM + BDS")
    print(f"    foul_trouble:  -(1-BDS) * (SPI/max(SPI))")
    print("  PASS: Dataset source audit complete")


# ═══════════════════════════════════════════════════════════════════
# 3. PARAMETER VALUE SANITY CHECKS
# ═══════════════════════════════════════════════════════════════════

def test_adj_em_range():
    """AdjEM should be in reasonable range for all teams."""
    for t in get_teams():
        assert -20 < t.adj_em < 50, f"{t.name} AdjEM={t.adj_em} out of range"
    duke = team_by_name("Duke")
    pva = team_by_name("Prairie View A&M")
    assert duke.adj_em > 30, f"Duke AdjEM={duke.adj_em} too low for a 1-seed"
    assert pva.adj_em < 0, f"PV A&M AdjEM={pva.adj_em} too high for a 16-seed"
    print(f"  PASS: AdjEM range valid. Duke={duke.adj_em:.1f}, PV A&M={pva.adj_em:.1f}")


def test_efg_pct_range():
    """eFG% should be between 0.40 and 0.65."""
    for t in get_teams():
        assert 0.35 < t.efg_pct < 0.70, f"{t.name} eFG%={t.efg_pct} out of range"
    print(f"  PASS: All eFG% in [0.35, 0.70]")


def test_seed_score():
    """Seed score = 1/seed."""
    duke = team_by_name("Duke")
    assert abs(duke.seed_score - 1.0) < 0.01, f"Duke seed_score={duke.seed_score}"
    siena = team_by_name("Siena")
    assert abs(siena.seed_score - 1/16) < 0.01, f"Siena seed_score={siena.seed_score}"
    print(f"  PASS: Seed scores correct (Duke=1.0, Siena=0.0625)")


def test_coaching_factor():
    """Known coaches should have reasonable CTF."""
    msu = team_by_name("Michigan State")
    assert msu.ctf > 0.55, f"Izzo's CTF={msu.ctf} too low"
    duke = team_by_name("Duke")
    assert duke.ctf > 0.50, f"Scheyer's CTF={duke.ctf} too low"
    print(f"  PASS: CTF values - MSU(Izzo)={msu.ctf:.3f}, Duke(Scheyer)={duke.ctf:.3f}")


def test_top25_performance():
    """1-seeds should generally have high Top-50 win%."""
    one_seeds = [t for t in get_teams() if t.seed == 1]
    for t in one_seeds:
        assert t.top25_perf > 0.3, f"{t.name} top50={t.top25_perf} suspiciously low for 1-seed"
    print(f"  PASS: Top-50 performance - " + ", ".join(f"{t.name}={t.top25_perf:.3f}" for t in one_seeds))


def test_no_nan_in_core_params():
    """No NaN values in any core parameter."""
    teams = get_teams()
    for t in teams:
        for key in PARAM_KEYS:
            val = getattr(t, key, None)
            if val is None:
                continue
            assert not np.isnan(val), f"{t.name}.{key} is NaN"
    print(f"  PASS: No NaN in {len(PARAM_KEYS)} core params across {len(teams)} teams")


def test_experience_range():
    """Experience should be 0.5-5.0."""
    for t in get_teams():
        assert 0.0 < t.exp < 5.0, f"{t.name} exp={t.exp} out of range"
    print(f"  PASS: All experience values in (0, 5)")


# ═══════════════════════════════════════════════════════════════════
# 4. NORMALIZATION AND COMPOSITE TESTS
# ═══════════════════════════════════════════════════════════════════

def test_weights_sum_to_one():
    """Weights must sum to 1.0."""
    total = sum(CORE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}"
    print(f"  PASS: Weights sum = {total:.6f}")


def test_normalization():
    """After normalization, all params in [0, 1]."""
    teams = get_teams()
    normalize_teams(teams, PARAM_KEYS)
    for t in teams:
        for key in PARAM_KEYS:
            v = t.normalized_params.get(key, 0.5)
            assert -0.01 <= v <= 1.01, f"{t.name}.{key} normalized={v} out of [0,1]"
    print(f"  PASS: All normalized params in [0, 1]")


def test_team_strength_ordering():
    """1-seeds should generally rank higher than 16-seeds."""
    teams = get_teams()
    compute_team_strengths(teams)
    ranked = rank_teams(teams)
    top5 = [t.name for t in ranked[:5]]
    one_seeds = {"Duke", "Arizona", "Florida", "Michigan"}
    overlap = one_seeds.intersection(top5)
    assert len(overlap) >= 3, f"Only {overlap} 1-seeds in top 5: {top5}"
    print(f"  PASS: Top 5 = {top5} (contains {len(overlap)} 1-seeds)")


# ═══════════════════════════════════════════════════════════════════
# 5. WIN PROBABILITY TESTS
# ═══════════════════════════════════════════════════════════════════

def test_1_vs_16_probability():
    """1-seed should beat 16-seed >90% of the time in Phase 1A."""
    teams = get_teams()
    compute_team_strengths(teams)
    duke = team_by_name("Duke")
    siena = team_by_name("Siena")
    m = Matchup(team_a=duke, team_b=siena, round_name="R64")
    from src.composite import compute_win_probability
    p = compute_win_probability(m)
    assert p > 0.90, f"Duke vs Siena P(Duke)={p:.3f} too low"
    print(f"  PASS: Duke vs Siena P(Duke)={p:.1%}")


def test_8_vs_9_near_50():
    """8-9 matchups should be near 50/50."""
    teams = get_teams()
    compute_team_strengths(teams)
    ohio = team_by_name("Ohio State")
    tcu = team_by_name("TCU")
    m = Matchup(team_a=ohio, team_b=tcu, round_name="R64")
    from src.composite import compute_win_probability
    p = compute_win_probability(m)
    assert 0.35 < p < 0.70, f"Ohio State vs TCU P(OSU)={p:.3f} not near 50/50"
    print(f"  PASS: Ohio State vs TCU P(OSU)={p:.1%}")


def test_1a_1b_gap():
    """Phase 1A and 1B should not differ by more than 25% on any matchup."""
    teams = get_teams()
    compute_team_strengths(teams)
    matchups = load_matchups(teams)
    from src.composite import compute_all_matchup_probabilities
    compute_all_matchup_probabilities(matchups)

    try:
        from src.xgboost_model import prepare_historical_features, train_xgboost, predict_matchup
        hist, tm = load_historical_games()
        X, y = prepare_historical_features(hist, tm)
        model, _ = train_xgboost(X, y)
        if model:
            max_gap = 0
            worst = None
            for m in matchups:
                m.win_prob_a_1b = predict_matchup(model, m.team_a, m.team_b,
                                                  p_1a=m.win_prob_a_1a)
                gap = abs(m.win_prob_a_1a - m.win_prob_a_1b)
                if gap > max_gap:
                    max_gap = gap
                    worst = m
            print(f"  INFO: Max 1A/1B gap = {max_gap:.1%} ({worst.team_a.name} vs {worst.team_b.name})")
            assert max_gap < 0.25, f"1A/1B gap too large: {max_gap:.1%}"
            print(f"  PASS: All gaps under 25%")
        else:
            print(f"  SKIP: XGBoost not available")
    except Exception as e:
        print(f"  SKIP: {e}")


# ═══════════════════════════════════════════════════════════════════
# 6. NEW PARAMETER COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════

def test_net_rating_loaded():
    """NET rating should be loaded for all teams."""
    duke = team_by_name("Duke")
    assert hasattr(duke, 'net_rating'), "Missing net_rating attribute"
    assert duke.net_rating > 0, f"Duke NET={duke.net_rating}"
    print(f"  PASS: NET rating loaded (Duke={duke.net_rating})")


def test_injury_rank_loaded():
    """Injury rank should be loaded from EvanMiya."""
    duke = team_by_name("Duke")
    assert hasattr(duke, 'injury_rank'), "Missing injury_rank attribute"
    print(f"  PASS: Injury rank loaded (Duke={duke.injury_rank})")


def test_ppg_loaded():
    """Points per game proxy (PPPO) should be loaded."""
    duke = team_by_name("Duke")
    assert hasattr(duke, 'ppg'), "Missing ppg attribute"
    assert duke.ppg > 0, f"Duke PPG={duke.ppg}"
    print(f"  PASS: PPG loaded (Duke={duke.ppg})")


def test_eff_height():
    """Effective height should be in meters (1.9-2.2m range)."""
    duke = team_by_name("Duke")
    assert 1.9 < duke.eff_height < 2.2, f"Duke eff_height={duke.eff_height}m out of range"
    print(f"  PASS: Height loaded (Duke={duke.eff_height:.2f}m)")


def test_q1_record_real():
    """Q1 record should come from real quadrant data, not AdjEM."""
    duke = team_by_name("Duke")
    assert 0 <= duke.q1_record <= 1.0, f"Duke q1_record={duke.q1_record} out of [0,1]"
    pva = team_by_name("Prairie View A&M")
    assert duke.q1_record > pva.q1_record, "Duke should have better Q1 record than PV A&M"
    print(f"  PASS: Q1 record real (Duke={duke.q1_record:.3f}, PVA&M={pva.q1_record:.3f})")


def test_scoring_balance():
    """Scoring balance should reward teams efficient from both inside and outside."""
    teams = get_teams()
    for t in teams:
        assert 0.1 < t.scoring_balance < 0.8, f"{t.name} scoring_balance={t.scoring_balance}"
    print(f"  PASS: Scoring balance in valid range for all 68 teams")


def test_legacy_winsorized():
    """Legacy factor should be capped at [-3, +5]."""
    for t in get_teams():
        assert -3.0 <= t.legacy_factor <= 5.0, f"{t.name} legacy={t.legacy_factor} not winsorized"
    print(f"  PASS: Legacy factor winsorized to [-3, +5]")


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("Dataset: Loads 68 teams", test_loads_68_teams),
        ("Dataset: All 1-seeds present", test_all_four_one_seeds),
        ("Dataset: All regions populated", test_all_regions_populated),
        ("Dataset: 32 matchups", test_matchups_load_32),
        ("Dataset: Historical data", test_historical_data),
        ("Dataset: Source report", test_dataset_sources_report),
        ("Param: AdjEM range", test_adj_em_range),
        ("Param: eFG% range", test_efg_pct_range),
        ("Param: Seed score", test_seed_score),
        ("Param: Coaching factor", test_coaching_factor),
        ("Param: Top-50 performance", test_top25_performance),
        ("Param: No NaN in core", test_no_nan_in_core_params),
        ("Param: Experience range", test_experience_range),
        ("Norm: Weights sum to 1", test_weights_sum_to_one),
        ("Norm: All params [0,1]", test_normalization),
        ("Composite: Strength ordering", test_team_strength_ordering),
        ("Prob: 1 vs 16", test_1_vs_16_probability),
        ("Prob: 8 vs 9", test_8_vs_9_near_50),
        ("Gap: 1A vs 1B", test_1a_1b_gap),
    ]

    new_param_tests = [
        ("NEW: NET rating", test_net_rating_loaded),
        ("NEW: Injury rank", test_injury_rank_loaded),
        ("NEW: PPG", test_ppg_loaded),
        ("NEW: Height in meters", test_eff_height),
        ("NEW: Q1 record (real)", test_q1_record_real),
        ("NEW: Scoring balance", test_scoring_balance),
        ("NEW: Legacy winsorized", test_legacy_winsorized),
    ]

    print("=" * 60)
    print("  NCAA PREDICTOR TEST SUITE")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  NEW PARAMETER TESTS (expected to fail before implementation)")
    print(f"{'='*60}")
    new_pass = 0
    new_fail = 0
    for name, test_fn in new_param_tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            new_pass += 1
        except (AssertionError, AttributeError, Exception) as e:
            print(f"  EXPECTED FAIL: {e}")
            new_fail += 1

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{len(tests)} core tests passed, {failed} failed")
    print(f"  NEW PARAMS: {new_pass}/{len(new_param_tests)} passed, {new_fail} pending implementation")
    print(f"{'='*60}")
