"""Full Tournament Simulation + Historical Backtest

Part 1: Historical Backtest (2008-2025)
  - Runs ROOT logistic model on ALL rounds: R64, R32, S16, E8, F4
  - Applies Round Survival Filter per round
  - Reports accuracy per round and per layer
  - Compares to 2026 actual results (R64=32 games, R32=16 games)

Part 2: 2026 Season Accuracy Check
  - Loads actual R64 + R32 results
  - Shows how each pipeline layer performed

Part 3: Monte Carlo Bracket Simulation (10,000 runs)
  - Uses full 4-layer pipeline (ROOT → Branch → Survival → Form Divergence)
  - Simulates E8 → F4 → Championship from S16 forward
  - Reports: E8 odds, F4 odds, Championship odds, expected bracket path
"""

import csv, math, os, random, sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

random.seed(42)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
base = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED: Historical team object + data loading
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HTeam:
    name: str; seed: int; year: int
    adj_em: float = 0.0; adj_o: float = 0.0; adj_d: float = 0.0
    three_p_pct: float = 0.0; efg_pct: float = 0.0
    orb_pct: float = 0.0; drb_pct: float = 0.0
    tov_pct: float = 0.0; ast_pct: float = 0.0; ftr: float = 0.0
    exp: float = 0.0; pace: float = 0.0
    wab: float = 0.0; sos: float = 0.0; barthag: float = 0.0
    q1_record: float = 0.5

YEARS = ['2008','2009','2010','2011','2012','2013','2014',
         '2015','2016','2017','2018','2019','2021','2022','2023','2024','2025']
ROUND_LABELS = {'64': 'R64', '32': 'R32', '16': 'S16', '8': 'E8', '4': 'F4', '2': 'CHAMP'}

def load_kp(year: str) -> Dict[str, HTeam]:
    teams = {}
    with open(os.path.join(base, "archive-3", "KenPom Barttorvik.csv")) as f:
        for row in csv.DictReader(f):
            if row.get("YEAR","").strip() != year: continue
            name = row.get("TEAM","").strip()
            try:
                t = HTeam(name=name, seed=int(row.get("SEED",0) or 0), year=int(year),
                    adj_em=float(row.get("KADJ EM",0) or 0),
                    adj_o=float(row.get("KADJ O",0) or 0),
                    adj_d=float(row.get("KADJ D",0) or 0),
                    three_p_pct=float(row.get("3PT%",0) or 0)/100,
                    efg_pct=float(row.get("EFG%",0) or 0)/100,
                    orb_pct=float(row.get("OREB%",0) or 0),
                    drb_pct=float(row.get("DREB%",0) or 0),
                    tov_pct=float(row.get("TOV%",0) or 0),
                    ast_pct=float(row.get("AST%",0) or 0),
                    ftr=float(row.get("FTR",0) or 0),
                    exp=float(row.get("EXP",0) or 0),
                    pace=float(row.get("K TEMPO",0) or 0),
                    wab=float(row.get("WAB",0) or 0),
                    sos=float(row.get("ELITE SOS",0) or 0),
                    barthag=float(row.get("BARTHAG",0) or 0))
                t.q1_record = min(1.0, max(0.0, (t.wab + 5) / 20))
                teams[name] = t
            except: pass
    return teams

def load_hist_matchups(cur_round: str) -> List[Tuple]:
    """Returns list of (year, team_a, team_b, winner) tuples."""
    rows = []
    with open(os.path.join(base, "archive-3", "Tournament Matchups.csv")) as f:
        rows = list(csv.DictReader(f))
    kp_cache = {y: load_kp(y) for y in YEARS}
    results = []
    for year in YEARS:
        yr = [r for r in rows if r.get("YEAR","").strip() == year
              and r.get("CURRENT ROUND","").strip() == cur_round]
        kp = kp_cache[year]
        for i in range(0, len(yr)-1, 2):
            ra, rb = yr[i], yr[i+1]
            na, nb = ra["TEAM"].strip(), rb["TEAM"].strip()
            sa, sb = int(ra["SCORE"] or 0), int(rb["SCORE"] or 0)
            ta = kp.get(na) or HTeam(name=na, seed=int(ra.get("SEED",8) or 8), year=int(year))
            tb = kp.get(nb) or HTeam(name=nb, seed=int(rb.get("SEED",8) or 8), year=int(year))
            ta.seed = int(ra.get("SEED", ta.seed) or ta.seed)
            tb.seed = int(rb.get("SEED", tb.seed) or tb.seed)
            winner = na if sa > sb else nb
            results.append((int(year), ta, tb, winner))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT + Survival probability functions
# ═══════════════════════════════════════════════════════════════════════════════

K = 0.135

def root_prob(ta: HTeam, tb: HTeam) -> float:
    z = (ta.adj_em - tb.adj_em)*0.55 + (ta.barthag - tb.barthag)*20*0.25 \
      + (ta.wab - tb.wab)*0.8*0.15 + (ta.sos - tb.sos)*0.3*0.05
    return 1/(1+math.exp(-z*K))

SURVIVAL_PROFILES = {
    'R64': [
        ('adj_em',   lambda a,b: a.adj_em > b.adj_em,    0.30),
        ('adj_d',    lambda a,b: a.adj_d < b.adj_d,      0.25),
        ('sos',      lambda a,b: a.sos > b.sos,           0.20),
        ('tov',      lambda a,b: a.tov_pct < b.tov_pct,  0.15),
        ('oreb',     lambda a,b: a.orb_pct > b.orb_pct,  0.10),
    ],
    'R32': [
        ('adj_o',    lambda a,b: a.adj_o > b.adj_o,      0.25),
        ('efg',      lambda a,b: a.efg_pct > b.efg_pct,  0.20),
        ('adj_em',   lambda a,b: a.adj_em > b.adj_em,    0.15),
        ('pace',     lambda a,b: a.pace > b.pace,         0.15),
        ('2pt_proxy',lambda a,b: a.efg_pct > b.efg_pct, 0.15),  # proxy
        ('adj_d',    lambda a,b: a.adj_d < b.adj_d,      0.10),
    ],
    'S16': [
        ('wab',      lambda a,b: a.q1_record > b.q1_record, 0.25),
        ('3pt',      lambda a,b: a.three_p_pct > b.three_p_pct, 0.20),
        ('oreb',     lambda a,b: a.orb_pct > b.orb_pct,  0.20),
        ('adj_o',    lambda a,b: a.adj_o > b.adj_o,      0.20),
        ('adj_d',    lambda a,b: a.adj_d < b.adj_d,      0.15),
    ],
    'E8':  [
        ('3pt',      lambda a,b: a.three_p_pct > b.three_p_pct, 0.30),
        ('pace_slow',lambda a,b: a.pace < b.pace,         0.25),  # slower wins E8
        ('ast',      lambda a,b: a.ast_pct > b.ast_pct,  0.20),
        ('adj_o',    lambda a,b: a.adj_o > b.adj_o,      0.15),
        ('ftr_low',  lambda a,b: a.ftr < b.ftr,           0.10),
    ],
    'F4':  [
        ('adj_em',   lambda a,b: a.adj_em > b.adj_em,    0.30),
        ('3pt',      lambda a,b: a.three_p_pct > b.three_p_pct, 0.25),
        ('adj_d',    lambda a,b: a.adj_d < b.adj_d,      0.25),
        ('exp',      lambda a,b: a.exp > b.exp,           0.20),
    ],
}
SURVIVAL_MAX = {'R64':0.06,'R32':0.08,'S16':0.07,'E8':0.06,'F4':0.04}

def survival_prob(ta: HTeam, tb: HTeam, p_root: float, rnd: str) -> float:
    profile = SURVIVAL_PROFILES.get(rnd)
    if not profile: return p_root
    def sev(team_is_a):
        triggered_w = sum(w for _, fn, w in profile
                          if fn(ta, tb) == team_is_a)
        total_w = sum(w for _, _, w in profile)
        n_trig = sum(1 for _, fn, w in profile if fn(ta, tb) == team_is_a)
        raw = triggered_w / total_w
        if n_trig >= 3:
            cascade = 1.0
            for _, fn, w in profile:
                if fn(ta, tb) == team_is_a:
                    cascade *= (1 + w * 0.4)
            raw = min(raw * cascade * 0.5, 1.0)
        return math.tanh(raw * 2.0)
    net = sev(True) - sev(False)
    shift = math.tanh(net * 1.5) * SURVIVAL_MAX.get(rnd, 0.05)
    return max(0.01, min(0.99, p_root + shift))


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Historical Backtest — ALL rounds 2008-2025
# ═══════════════════════════════════════════════════════════════════════════════

print("="*90)
print("  PART 1: HISTORICAL BACKTEST — ALL ROUNDS 2008-2025")
print("="*90)

all_results = {}
round_order = [('64','R64'),('32','R32'),('16','S16'),('8','E8'),('4','F4')]

for cr, rnd_label in round_order:
    matchups = load_hist_matchups(cr)
    seed_c = root_c = surv_c = 0
    n = len(matchups)
    year_acc = defaultdict(lambda: {'root':0,'surv':0,'n':0})

    for year, ta, tb, winner in matchups:
        winner_is_a = (winner == ta.name)
        seed_c += (ta.seed < tb.seed) == winner_is_a
        p_r = root_prob(ta, tb)
        root_ok = (p_r > 0.5) == winner_is_a
        root_c += root_ok
        p_s = survival_prob(ta, tb, p_r, rnd_label)
        surv_ok = (p_s > 0.5) == winner_is_a
        surv_c += surv_ok
        year_acc[year]['root'] += root_ok
        year_acc[year]['surv'] += surv_ok
        year_acc[year]['n'] += 1

    all_results[rnd_label] = {
        'n': n, 'seed': seed_c, 'root': root_c, 'surv': surv_c,
        'year_acc': dict(year_acc)
    }

# Print summary table
print(f"\n  {'Round':<8} {'N':>5}  {'Seed%':>8}  {'ROOT%':>8}  {'ROOT+Surv':>10}  {'Δ Surv':>8}  {'Upset%':>8}")
print("  " + "─"*72)
for rnd_label in ['R64','R32','S16','E8','F4']:
    r = all_results[rnd_label]
    n = r['n']
    seed_acc = r['seed']/n
    root_acc = r['root']/n
    surv_acc = r['surv']/n
    # Upset rate: how often does higher seed win
    matchups = load_hist_matchups({'R64':'64','R32':'32','S16':'16','E8':'8','F4':'4'}[rnd_label])
    upsets = sum(1 for _, ta, tb, winner in matchups
                 if (winner == ta.name) != (ta.seed < tb.seed))
    upset_rate = upsets / len(matchups) if matchups else 0
    delta = surv_acc - root_acc
    marker = "▲" if delta > 0.005 else "▼" if delta < -0.005 else "─"
    print(f"  {rnd_label:<8} {n:>5}  {seed_acc:>7.1%}  {root_acc:>7.1%}  {surv_acc:>9.1%}  "
          f"{marker}{abs(delta):>6.1%}  {upset_rate:>7.1%}")

# Year-by-year for each round
print(f"\n\n  YEAR-BY-YEAR BREAKDOWN BY ROUND:")
print(f"  {'Year':<6}", end="")
for rnd in ['R64','R32','S16','E8','F4']:
    print(f"  {rnd:>10}", end="")
print()
print("  " + "─"*60)
all_years = sorted(set(y for rnd in all_results for y in all_results[rnd]['year_acc']))
for year in all_years:
    print(f"  {year:<6}", end="")
    for rnd in ['R64','R32','S16','E8','F4']:
        ya = all_results[rnd]['year_acc'].get(year, {'root':0,'surv':0,'n':0})
        if ya['n'] > 0:
            acc = ya['surv']/ya['n']
            print(f"  {ya['surv']:>2}/{ya['n']:>2} {acc:>4.0%}", end="")
        else:
            print(f"  {'':>10}", end="")
    print()

# Overall pipeline accuracy
print(f"\n  TOTAL GAMES BACKTESTED: {sum(r['n'] for r in all_results.values())}")
total_root = sum(r['root'] for r in all_results.values())
total_surv = sum(r['surv'] for r in all_results.values())
total_n = sum(r['n'] for r in all_results.values())
print(f"  ROOT (all rounds):      {total_root}/{total_n} = {total_root/total_n:.1%}")
print(f"  ROOT+Survival:          {total_surv}/{total_n} = {total_surv/total_n:.1%}  (Δ{(total_surv-total_root)/total_n:+.1%})")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: 2026 Season Accuracy — actual R64 + R32 results
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print("  PART 2: 2026 SEASON ACCURACY — ACTUAL vs PREDICTED")
print(f"{'='*90}")

# Load actual results
actual_r64 = {}
with open(os.path.join(base, "data/live_results/day1_thursday.csv")) as f:
    for row in csv.DictReader(f):
        actual_r64[(row['team_a'].strip(), row['team_b'].strip())] = row['winner'].strip()
with open(os.path.join(base, "data/live_results/day2_friday.csv")) as f:
    for row in csv.DictReader(f):
        actual_r64[(row['team_a'].strip(), row['team_b'].strip())] = row['winner'].strip()

actual_r32 = {}
for fname in ['r32_saturday.csv', 'r32_sunday.csv']:
    with open(os.path.join(base, f"data/live_results/{fname}")) as f:
        for row in csv.DictReader(f):
            actual_r32[(row['team_a'].strip(), row['team_b'].strip())] = row['winner'].strip()

# Load S16 predictions
s16_preds = {}
with open(os.path.join(base, "data/live_results/s16_predictions.csv")) as f:
    for row in csv.DictReader(f):
        s16_preds[(row['team_a'], row['team_b'])] = row

# Load full pipeline for 2026 (use the same pipeline as run_s16_predictions.py)
print("\n  Loading 2026 team data through full pipeline...")
from src.data_loader import load_all_teams, load_matchups, build_season_h2h
from src.composite import compute_team_strengths, compute_all_matchup_probabilities, generate_pros_cons
from src.niche import enrich_niche
from src.weights import ACTIVE_WEIGHTS, ENSEMBLE_LAMBDA, DATASET_CONFIG
from src.ensemble import blend_all_matchups
from src.models import Matchup
from src.branch_engine import MatchupBranchContext, evaluate_all_branches
from src.round_survival import evaluate_round_survival
from src.form_divergence import load_all_game_logs, evaluate_form_divergence

teams_2026 = load_all_teams()
enrich_niche(teams_2026)
injury_profiles = {}
if DATASET_CONFIG.get("use_injury_model", False):
    try:
        from src.injury_model import load_injuries, quantify_player_impacts, apply_injury_degradation
        from src.player_matchup import load_player_data
        injuries = load_injuries()
        player_df = load_player_data()
        if injuries:
            injury_profiles = quantify_player_impacts(injuries, player_df)
            teams_2026 = apply_injury_degradation(teams_2026, injury_profiles)
    except: pass
teams_2026 = compute_team_strengths(teams_2026)
team_map = {t.name: t for t in teams_2026}
game_logs = load_all_game_logs([t.name for t in teams_2026],
                                os.path.join(base, "archive-3", "game-logs"))
h2h = build_season_h2h()

def build_ctx(m, rnd, sii_r=None):
    a, b = m.team_a, m.team_b
    pa = injury_profiles.get(a.name)
    pb = injury_profiles.get(b.name)
    return MatchupBranchContext(
        team_a_name=a.name, team_b_name=b.name, round_name=rnd,
        p_base=m.win_prob_a_ensemble, seed_a=a.seed, seed_b=b.seed,
        sii_a=0.0, sii_b=0.0,
        star_lost_a=pa.has_star_carrier_out if pa else False,
        star_lost_b=pb.has_star_carrier_out if pb else False,
        star_bpr_share_a=getattr(pa,'top_player_bpr_share',0.0) if pa else 0.0,
        star_bpr_share_b=getattr(pb,'top_player_bpr_share',0.0) if pb else 0.0,
        team_exp_a=a.exp, team_exp_b=b.exp,
        star_ppg_a=a.best_player_above_avg_pts+12 if a.best_player_above_avg_pts>0 else 0,
        star_ppg_b=b.best_player_above_avg_pts+12 if b.best_player_above_avg_pts>0 else 0,
        orb_pct_a=a.orb_pct, orb_pct_b=b.orb_pct,
        drb_pct_a=a.drb_pct, drb_pct_b=b.drb_pct,
        rbm_a=a.rbm, rbm_b=b.rbm,
        offensive_burst_a=a.offensive_burst, offensive_burst_b=b.offensive_burst,
        q3_adj_a=a.q3_adj_strength, q3_adj_b=b.q3_adj_strength,
        form_trend_a=a.form_trend, form_trend_b=b.form_trend,
        momentum_a=a.momentum, momentum_b=b.momentum,
        bench_depth_a=a.bds, bench_depth_b=b.bds,
        three_pt_share_a=a.three_pa_fga, three_pt_share_b=b.three_pa_fga,
        three_pt_pct_a=a.three_p_pct, three_pt_pct_b=b.three_p_pct,
        three_pt_std_a=a.three_pt_std, three_pt_std_b=b.three_pt_std,
        first_tourney_a=a.exp<1.5, first_tourney_b=b.exp<1.5,
        foul_trouble_rate_a=a.foul_trouble_impact, foul_trouble_rate_b=b.foul_trouble_impact,
        crippled_roster_a=getattr(pa,'has_crippled_roster',False) if pa else False,
        crippled_roster_b=getattr(pb,'has_crippled_roster',False) if pb else False,
        crippled_weeks_out_a=getattr(pa,'crippled_weeks_out',0.0) if pa else 0.0,
        crippled_weeks_out_b=getattr(pb,'crippled_weeks_out',0.0) if pb else 0.0,
        crippled_ppg_lost_a=getattr(pa,'crippled_ppg_lost',0.0) if pa else 0.0,
        crippled_ppg_lost_b=getattr(pb,'crippled_ppg_lost',0.0) if pb else 0.0,
        sos_rank_a=getattr(a,'sos_rank',50), sos_rank_b=getattr(b,'sos_rank',50),
        q1_wins_a=int(getattr(a,'q1_record',0.5)*10), q1_wins_b=int(getattr(b,'q1_record',0.5)*10),
    )

def full_pipeline_prob(ta, tb, rnd):
    """Run full 4-layer pipeline, return probability for team_a."""
    m = Matchup(team_a=ta, team_b=tb, round_name=rnd, region="")
    m.h2h_season_edge = h2h.get((ta.name, tb.name), 0.0)
    ms = compute_all_matchup_probabilities([m])
    ms = blend_all_matchups(ms, ENSEMBLE_LAMBDA)
    m = ms[0]
    p_root = m.win_prob_a_ensemble
    ctx = build_ctx(m, rnd)
    ra, rb = evaluate_all_branches(ctx)
    p_branch = ra.p_final
    p_surv, _, _ = evaluate_round_survival(ta, tb, rnd, p_branch)
    la, lb = game_logs.get(ta.name), game_logs.get(tb.name)
    p_final, _, _ = evaluate_form_divergence(ta, tb, la, lb, rnd, p_surv)
    return p_final, p_root, p_branch, p_surv

# Evaluate R64 accuracy
def get_actual(name_a, name_b, d):
    w = d.get((name_a, name_b)) or d.get((name_b, name_a))
    return w

# Load R64 matchups
r64_matchups = load_matchups(teams_2026, h2h)
r64_matchups = compute_all_matchup_probabilities(r64_matchups)
for m in r64_matchups: generate_pros_cons(m)
r64_matchups = blend_all_matchups(r64_matchups, ENSEMBLE_LAMBDA)

print(f"\n  {'Round':<6}  {'Layer':<20}  {'Correct':>9}  {'Total':>6}  {'Accuracy':>9}")
print("  " + "─"*56)

# R64
r64_root_c = r64_full_c = 0
for m in r64_matchups:
    a, b = m.team_a, m.team_b
    actual = get_actual(a.name, b.name, actual_r64)
    if not actual: continue
    p_r = m.win_prob_a_ensemble
    if (p_r > 0.5) == (actual == a.name): r64_root_c += 1
    # Full pipeline
    p_s, _, _ = evaluate_round_survival(a, b, "R64", p_r)
    la, lb = game_logs.get(a.name), game_logs.get(b.name)
    p_f, _, _ = evaluate_form_divergence(a, b, la, lb, "R64", p_s)
    if (p_f > 0.5) == (actual == a.name): r64_full_c += 1
n64 = len([m for m in r64_matchups if get_actual(m.team_a.name, m.team_b.name, actual_r64)])
print(f"  {'R64':<6}  {'ROOT only':<20}  {r64_root_c:>9}  {n64:>6}  {r64_root_c/n64:>8.1%}")
print(f"  {'R64':<6}  {'ROOT+Surv+Form':<20}  {r64_full_c:>9}  {n64:>6}  {r64_full_c/n64:>8.1%}")

# R32 — from r32_full_predictions.csv + survival + form
r32_branch_c = r32_surv_c = r32_full_c = 0
r32_probs = {}
with open(os.path.join(base, "data/live_results/r32_full_predictions.csv")) as f:
    for row in csv.DictReader(f):
        r32_probs[(row['team_a'], row['team_b'])] = float(row['final_prob_a'])

r32_games = [
    ("East","Duke","TCU"),("East","Louisville","Michigan State"),
    ("East","St. John's","Kansas"),("East","UCLA","Connecticut"),
    ("South","Vanderbilt","Nebraska"),("South","VCU","Illinois"),
    ("South","Texas A&M","Houston"),("South","Florida","Iowa"),
    ("West","High Point","Arkansas"),("West","Texas","Gonzaga"),
    ("West","Arizona","Utah State"),("West","Miami FL","Purdue"),
    ("Midwest","Michigan","Saint Louis"),("Midwest","Texas Tech","Alabama"),
    ("Midwest","Tennessee","Virginia"),("Midwest","Kentucky","Iowa State"),
]
n32 = 0
for region, na, nb in r32_games:
    ta, tb = team_map.get(na), team_map.get(nb)
    if not ta or not tb: continue
    actual = get_actual(na, nb, actual_r32)
    if not actual: continue
    n32 += 1
    p_branch = r32_probs.get((na, nb), 0.5)
    if (p_branch > 0.5) == (actual == na): r32_branch_c += 1
    m = Matchup(team_a=ta, team_b=tb, round_name="R32", region=region)
    m.win_prob_a_ensemble = p_branch
    p_s, _, _ = evaluate_round_survival(ta, tb, "R32", p_branch)
    la, lb = game_logs.get(na), game_logs.get(nb)
    p_f, _, _ = evaluate_form_divergence(ta, tb, la, lb, "R32", p_s)
    if (p_s > 0.5) == (actual == na): r32_surv_c += 1
    if (p_f > 0.5) == (actual == na): r32_full_c += 1

print(f"  {'R32':<6}  {'Branch only':<20}  {r32_branch_c:>9}  {n32:>6}  {r32_branch_c/n32:>8.1%}")
print(f"  {'R32':<6}  {'Branch+Survival':<20}  {r32_surv_c:>9}  {n32:>6}  {r32_surv_c/n32:>8.1%}")
print(f"  {'R32':<6}  {'Full Pipeline':<20}  {r32_full_c:>9}  {n32:>6}  {r32_full_c/n32:>8.1%}")

total_2026 = n64 + n32
total_full = r64_full_c + r32_full_c
print(f"\n  {'TOTAL':<6}  {'Full Pipeline':<20}  {total_full:>9}  {total_2026:>6}  {total_full/total_2026:>8.1%}")
print(f"  (Historical baseline for same rounds: {all_results['R64']['root']/all_results['R64']['n']:.1%} R64 / {all_results['R32']['root']/all_results['R32']['n']:.1%} R32)")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Monte Carlo Bracket Simulation (10,000 runs)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*90}")
print("  PART 3: MONTE CARLO BRACKET SIMULATION — 10,000 RUNS")
print(f"  From Sweet 16 forward (E8 → F4 → Championship)")
print(f"{'='*90}")

N_SIMS = 10000

# S16 matchups and probabilities (from our pipeline)
print("\n  Computing S16 probabilities through full pipeline...")

s16_bracket = [
    # (region, team_a, team_b, bracket_side)
    # East top half vs East bottom half in F4
    ("East",  "Duke",      "St. John's",    "East_top"),
    ("East",  "Connecticut","Michigan State","East_bot"),
    ("South", "Iowa",       "Nebraska",      "South_top"),
    ("South", "Houston",    "Illinois",      "South_bot"),
    ("West",  "Arizona",    "Arkansas",      "West_top"),
    ("West",  "Purdue",     "Texas",         "West_bot"),
    ("Midwest","Michigan",  "Alabama",       "Midwest_top"),
    ("Midwest","Iowa State","Tennessee",     "Midwest_bot"),
]

# F4 pairings: East winner vs South winner, West winner vs Midwest winner
F4_PAIRINGS = [
    ("East_top","East_bot","East"),
    ("South_top","South_bot","South"),
    ("West_top","West_bot","West"),
    ("Midwest_top","Midwest_bot","Midwest"),
]
CHAMP_PAIRINGS = [
    ("East", "South"),  # East region winner vs South region winner
    ("West", "Midwest"),
]

# Pre-compute pipeline probabilities for all potential matchups we might see
print("  Pre-computing matchup probabilities (this may take a moment)...")

# Get all teams in S16
s16_teams = list(set([na for _, na, nb, _ in s16_bracket] +
                     [nb for _, na, nb, _ in s16_bracket]))

# Cache probabilities
prob_cache = {}

def get_cached_prob(na, nb, rnd):
    key = (na, nb, rnd)
    key_rev = (nb, na, rnd)
    if key in prob_cache: return prob_cache[key]
    if key_rev in prob_cache: return 1 - prob_cache[key_rev]
    ta, tb = team_map.get(na), team_map.get(nb)
    if not ta or not tb: return 0.5
    try:
        p, _, _, _ = full_pipeline_prob(ta, tb, rnd)
    except:
        p = 0.5
    prob_cache[key] = p
    return p

# S16 probs
for _, na, nb, _ in s16_bracket:
    get_cached_prob(na, nb, "S16")
    sys.stdout.write(".")
    sys.stdout.flush()
print(f" S16 done")

# Pre-compute E8 matchups (S16 winner combos within each region)
e8_combos = set()
for i in range(0, len(s16_bracket), 2):
    top_teams = [s16_bracket[i][1], s16_bracket[i][2]]
    bot_teams = [s16_bracket[i+1][1], s16_bracket[i+1][2]]
    for t in top_teams:
        for b in bot_teams:
            e8_combos.add((t, b))

for na, nb in e8_combos:
    get_cached_prob(na, nb, "E8")
    sys.stdout.write(".")
    sys.stdout.flush()
print(f" E8 done")

# ── Monte Carlo ──────────────────────────────────────────────────────────────
champ_counts = defaultdict(int)
f4_counts = defaultdict(int)
e8_counts = defaultdict(int)
s16_win_counts = defaultdict(int)

# Also track expected bracket path
region_e8_winners = defaultdict(lambda: defaultdict(int))
region_f4_winners = defaultdict(lambda: defaultdict(int))

for sim in range(N_SIMS):
    region_winners = {}  # region_side → winner name

    # ── S16: simulate each game ────────────────────────────────────────────
    for region, na, nb, side in s16_bracket:
        p = get_cached_prob(na, nb, "S16")
        winner = na if random.random() < p else nb
        region_winners[side] = winner
        s16_win_counts[winner] += 1

    # ── E8: top half vs bottom half within each region ────────────────────
    e8_region_winners = {}
    for rnd_top, rnd_bot, region_name in F4_PAIRINGS:
        ta_name = region_winners[rnd_top]
        tb_name = region_winners[rnd_bot]
        p = get_cached_prob(ta_name, tb_name, "E8")
        if (ta_name, tb_name, "E8") not in prob_cache and \
           (tb_name, ta_name, "E8") not in prob_cache:
            p = get_cached_prob(ta_name, tb_name, "E8")
        winner = ta_name if random.random() < p else tb_name
        e8_counts[winner] += 1
        e8_region_winners[region_name] = winner

    # ── F4: East vs South, West vs Midwest ───────────────────────────────
    f4_region_winners = {}
    for r1, r2 in CHAMP_PAIRINGS:
        ta_name = e8_region_winners[r1]
        tb_name = e8_region_winners[r2]
        p = get_cached_prob(ta_name, tb_name, "F4")
        winner = ta_name if random.random() < p else tb_name
        f4_counts[winner] += 1
        f4_region_winners[f"{r1}_{r2}"] = winner

    # ── Championship ──────────────────────────────────────────────────────
    side1, side2 = list(f4_region_winners.keys())
    ta_name = f4_region_winners[side1]
    tb_name = f4_region_winners[side2]
    p = get_cached_prob(ta_name, tb_name, "CHAMP")
    winner = ta_name if random.random() < p else tb_name
    champ_counts[winner] += 1

# ── Print results ──────────────────────────────────────────────────────────
all_s16_teams = list(set([na for _, na, nb, _ in s16_bracket] +
                         [nb for _, na, nb, _ in s16_bracket]))

print(f"\n  {'Team':<22} {'Seed':>5}  {'E8%':>7}  {'F4%':>7}  {'Champ%':>8}  Conf")
print("  " + "─"*65)

# Group by region for readability
by_region = defaultdict(list)
for _, na, nb, side in s16_bracket:
    region = side.split("_")[0]
    by_region[region].extend([na, nb])
# deduplicate
for k in by_region: by_region[k] = list(dict.fromkeys(by_region[k]))

for region in ["East", "South", "West", "Midwest"]:
    print(f"\n  ── {region} ──")
    for name in by_region[region]:
        t = team_map.get(name)
        seed = t.seed if t else "?"
        e8_pct = e8_counts[name] / N_SIMS
        f4_pct = f4_counts[name] / N_SIMS
        ch_pct = champ_counts[name] / N_SIMS
        conf = "★★★" if ch_pct > 0.12 else "★★" if ch_pct > 0.06 else "★" if ch_pct > 0.02 else ""
        print(f"  {name:<22} ({seed:>2})  {e8_pct:>6.1%}  {f4_pct:>6.1%}  {ch_pct:>7.1%}  {conf}")

# Top 8 championship contenders
print(f"\n\n  TOP CHAMPIONSHIP CONTENDERS (sorted by title odds):")
print(f"  {'Rank':<5}  {'Team':<22}  {'Seed':>5}  {'E8%':>7}  {'F4%':>7}  {'Champ%':>8}")
print("  " + "─"*60)
top8 = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)[:8]
for i, (name, cnt) in enumerate(top8, 1):
    t = team_map.get(name)
    seed = t.seed if t else "?"
    print(f"  #{i:<4}  {name:<22}  ({seed:>2})  "
          f"{e8_counts[name]/N_SIMS:>6.1%}  {f4_counts[name]/N_SIMS:>6.1%}  {cnt/N_SIMS:>7.1%}")

# Most likely bracket path
print(f"\n\n  MOST LIKELY BRACKET PATH (deterministic — always take higher-prob team):")
print(f"  {'Stage':<8}  {'Region':<12}  {'Matchup':<45}  {'Pick':<22}  {'Prob':>6}")
print("  " + "─"*95)

det_s16_winners = {}
for region, na, nb, side in s16_bracket:
    p = get_cached_prob(na, nb, "S16")
    winner = na if p > 0.5 else nb
    det_s16_winners[side] = winner
    ta = team_map.get(na)
    tb = team_map.get(nb)
    print(f"  {'S16':<8}  {region:<12}  ({ta.seed if ta else '?'}){na:<20} vs ({tb.seed if tb else '?'}){nb:<20}  "
          f"{winner:<22}  {max(p,1-p):>5.1%}")

det_e8_winners = {}
print()
for rnd_top, rnd_bot, region_name in F4_PAIRINGS:
    ta_name = det_s16_winners[rnd_top]
    tb_name = det_s16_winners[rnd_bot]
    p = get_cached_prob(ta_name, tb_name, "E8")
    winner = ta_name if p > 0.5 else tb_name
    det_e8_winners[region_name] = winner
    ta = team_map.get(ta_name)
    tb = team_map.get(tb_name)
    print(f"  {'E8':<8}  {region_name:<12}  ({ta.seed if ta else '?'}){ta_name:<20} vs ({tb.seed if tb else '?'}){tb_name:<20}  "
          f"{winner:<22}  {max(p,1-p):>5.1%}")

det_f4_winners = {}
print()
for r1, r2 in CHAMP_PAIRINGS:
    ta_name = det_e8_winners[r1]
    tb_name = det_e8_winners[r2]
    p = get_cached_prob(ta_name, tb_name, "F4")
    winner = ta_name if p > 0.5 else tb_name
    det_f4_winners[f"{r1}_{r2}"] = winner
    ta = team_map.get(ta_name)
    tb = team_map.get(tb_name)
    print(f"  {'F4':<8}  {r1+'/'+r2:<12}  ({ta.seed if ta else '?'}){ta_name:<20} vs ({tb.seed if tb else '?'}){tb_name:<20}  "
          f"{winner:<22}  {max(p,1-p):>5.1%}")

# Championship
print()
side1, side2 = list(det_f4_winners.keys())
ta_name = det_f4_winners[side1]
tb_name = det_f4_winners[side2]
p = get_cached_prob(ta_name, tb_name, "CHAMP")
winner = ta_name if p > 0.5 else tb_name
ta = team_map.get(ta_name)
tb = team_map.get(tb_name)
print(f"  {'CHAMP':<8}  {'FINAL':<12}  ({ta.seed if ta else '?'}){ta_name:<20} vs ({tb.seed if tb else '?'}){tb_name:<20}  "
      f"★ {winner:<20}  {max(p,1-p):>5.1%}")

print(f"\n\n  ★ PREDICTED 2026 NCAA CHAMPION: {winner} ({max(p,1-p):.1%} in the final)")
print(f"  Simulation championship odds:   {champ_counts[winner]/N_SIMS:.1%} (from {N_SIMS:,} Monte Carlo runs)")

# Save simulation results
sim_path = os.path.join(base, "data/live_results/simulation_results.csv")
with open(sim_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['team','seed','e8_pct','f4_pct','champ_pct'])
    w.writeheader()
    for name in all_s16_teams:
        t = team_map.get(name)
        w.writerow({'team': name, 'seed': t.seed if t else '?',
                    'e8_pct': f"{e8_counts[name]/N_SIMS:.4f}",
                    'f4_pct': f"{f4_counts[name]/N_SIMS:.4f}",
                    'champ_pct': f"{champ_counts[name]/N_SIMS:.4f}"})
print(f"\n  Results saved to: {sim_path}")
