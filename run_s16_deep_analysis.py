"""S16 DEEP ANALYSIS — Every edge, branch, flag, and form shift for all 8 games.
Outputs structured breakdown for each matchup:
  - Raw stat comparison with who holds each edge
  - Branch engine: which branches fired and WHY
  - Round survival: which S16-specific flags triggered
  - Form divergence: trend direction, severity, game log snapshots
  - Net pipeline progression (ROOT → BRANCH → SURV → FORM → FINAL)
"""

import sys, os, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_teams, load_matchups, load_historical_games, build_season_h2h
from src.composite import compute_team_strengths, compute_all_matchup_probabilities, generate_pros_cons
from src.niche import enrich_niche
from src.weights import CORE_WEIGHTS, ACTIVE_WEIGHTS, ENSEMBLE_LAMBDA, DATASET_CONFIG
from src.ensemble import blend_all_matchups
from src.models import Matchup
from src.branch_engine import MatchupBranchContext, evaluate_all_branches, branch_engine_report
from src.round_survival import evaluate_round_survival, round_survival_report
from src.form_divergence import (
    load_all_game_logs, evaluate_form_divergence,
    form_divergence_report, detect_sleeping_giant, NAME_TO_FILE
)

base = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# LOAD PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
print("Loading full pipeline...\n")
teams = load_all_teams()
enrich_niche(teams)

if DATASET_CONFIG.get("use_recency_weighting", False):
    try:
        from src.recency import compute_recency_metrics, enrich_teams_with_recency
        recency_data = compute_recency_metrics(teams)
        enrich_teams_with_recency(teams, recency_data)
    except: pass

injury_profiles = {}
sii_results = {}
if DATASET_CONFIG.get("use_injury_model", False):
    try:
        from src.injury_model import (
            load_injuries, quantify_player_impacts,
            apply_injury_degradation, compute_star_isolation,
        )
        from src.player_matchup import load_player_data
        injuries = load_injuries()
        player_df = load_player_data()
        if injuries:
            injury_profiles = quantify_player_impacts(injuries, player_df)
            sii_results = compute_star_isolation(injury_profiles, player_df)
            teams = apply_injury_degradation(teams, injury_profiles, round_name="S16")
    except Exception as e:
        print(f"  Injury model error: {e}")

teams = compute_team_strengths(teams)
team_map = {t.name: t for t in teams}

game_logs_dir = os.path.join(base, "archive-3", "game-logs")
all_team_names = [t.name for t in teams]
game_logs = load_all_game_logs(all_team_names, game_logs_dir)

h2h_lookup = build_season_h2h()


# S16 matchups
s16_games = [
    ("East", "Duke", "St. John's"),
    ("East", "Connecticut", "Michigan State"),
    ("South", "Iowa", "Nebraska"),
    ("South", "Houston", "Illinois"),
    ("West", "Arizona", "Arkansas"),
    ("West", "Purdue", "Texas"),
    ("Midwest", "Michigan", "Alabama"),
    ("Midwest", "Iowa State", "Tennessee"),
]


def build_branch_ctx(m, round_name="S16"):
    a = m.team_a
    b = m.team_b
    prof_a = injury_profiles.get(a.name)
    prof_b = injury_profiles.get(b.name)
    ctx = MatchupBranchContext(
        team_a_name=a.name, team_b_name=b.name,
        round_name=round_name, p_base=m.win_prob_a_ensemble,
        seed_a=a.seed, seed_b=b.seed,
        sii_a=sii_results.get(a.name, 0.0) if sii_results else 0.0,
        sii_b=sii_results.get(b.name, 0.0) if sii_results else 0.0,
        star_lost_a=prof_a.has_star_carrier_out if prof_a else False,
        star_lost_b=prof_b.has_star_carrier_out if prof_b else False,
        star_bpr_share_a=prof_a.top_player_bpr_share if prof_a and hasattr(prof_a, 'top_player_bpr_share') else 0.0,
        star_bpr_share_b=prof_b.top_player_bpr_share if prof_b and hasattr(prof_b, 'top_player_bpr_share') else 0.0,
        team_exp_a=a.exp, team_exp_b=b.exp,
        star_ppg_a=a.best_player_above_avg_pts + 12.0 if a.best_player_above_avg_pts > 0 else 0.0,
        star_ppg_b=b.best_player_above_avg_pts + 12.0 if b.best_player_above_avg_pts > 0 else 0.0,
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
        first_tourney_a=a.exp < 1.5, first_tourney_b=b.exp < 1.5,
        foul_trouble_rate_a=a.foul_trouble_impact, foul_trouble_rate_b=b.foul_trouble_impact,
        crippled_roster_a=prof_a.has_crippled_roster if prof_a and hasattr(prof_a, 'has_crippled_roster') else False,
        crippled_roster_b=prof_b.has_crippled_roster if prof_b and hasattr(prof_b, 'has_crippled_roster') else False,
        crippled_weeks_out_a=prof_a.crippled_weeks_out if prof_a and hasattr(prof_a, 'crippled_weeks_out') else 0.0,
        crippled_weeks_out_b=prof_b.crippled_weeks_out if prof_b and hasattr(prof_b, 'crippled_weeks_out') else 0.0,
        crippled_ppg_lost_a=prof_a.crippled_ppg_lost if prof_a and hasattr(prof_a, 'crippled_ppg_lost') else 0.0,
        crippled_ppg_lost_b=prof_b.crippled_ppg_lost if prof_b and hasattr(prof_b, 'crippled_ppg_lost') else 0.0,
        sos_rank_a=getattr(a, 'sos_rank', 50), sos_rank_b=getattr(b, 'sos_rank', 50),
        q1_wins_a=int(getattr(a, 'q1_record', 0.5) * 10),
        q1_wins_b=int(getattr(b, 'q1_record', 0.5) * 10),
    )
    return ctx


# ═══════════════════════════════════════════════════════════════════════════
# DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 100)
print("  SWEET 16 — DEEP MATCHUP ANALYSIS")
print("  Every edge, every branch, every flag, every form shift")
print("=" * 100)

# Build matchups through ROOT
s16_matchups = []
for region, na, nb in s16_games:
    ta, tb = team_map.get(na), team_map.get(nb)
    if not ta or not tb:
        print(f"  MISSING: {na if not ta else nb}")
        continue
    m = Matchup(team_a=ta, team_b=tb, round_name="S16", region=region)
    m.h2h_season_edge = h2h_lookup.get((ta.name, tb.name), 0.0)
    s16_matchups.append(m)

s16_matchups = compute_all_matchup_probabilities(s16_matchups)
for m in s16_matchups:
    generate_pros_cons(m)
s16_matchups = blend_all_matchups(s16_matchups, ENSEMBLE_LAMBDA)

if DATASET_CONFIG.get("use_injury_model", False) and injury_profiles:
    try:
        from src.injury_model import apply_star_vacuum_penalty
        s16_matchups = apply_star_vacuum_penalty(s16_matchups, injury_profiles, round_name="S16")
    except: pass


for m in s16_matchups:
    a, b = m.team_a, m.team_b
    root_prob = m.win_prob_a_ensemble

    # LAYER 2: Branch Engine
    ctx = build_branch_ctx(m, "S16")
    res_a, res_b = evaluate_all_branches(ctx)
    branch_prob = res_a.p_final

    # LAYER 3: Round Survival
    p_survival, surv_a, surv_b = evaluate_round_survival(a, b, "S16", branch_prob)

    # LAYER 4: Form Divergence
    log_a = game_logs.get(a.name)
    log_b = game_logs.get(b.name)
    p_final, form_a, form_b = evaluate_form_divergence(a, b, log_a, log_b, "S16", p_survival)

    sg_a = detect_sleeping_giant(a, log_a)
    sg_b = detect_sleeping_giant(b, log_b)

    # ─── HEADER ───
    print(f"\n\n{'█' * 100}")
    print(f"  {m.region} REGION: ({a.seed}) {a.name} vs ({b.seed}) {b.name}")
    print(f"{'█' * 100}")

    # ─── PIPELINE PROGRESSION ───
    deltas = []
    deltas.append(f"ROOT={root_prob:.1%}")
    d1 = branch_prob - root_prob
    deltas.append(f"BRANCH={branch_prob:.1%} ({d1:+.1%})")
    d2 = p_survival - branch_prob
    deltas.append(f"SURV={p_survival:.1%} ({d2:+.1%})")
    d3 = p_final - p_survival
    deltas.append(f"FORM={p_final:.1%} ({d3:+.1%})")
    total_d = p_final - root_prob
    pick = a.name if p_final > 0.5 else b.name
    conf = abs(p_final - 0.5) * 2

    print(f"\n  ╔══ PIPELINE ═══════════════════════════════════════════════════════╗")
    print(f"  ║  {' → '.join(deltas)}")
    print(f"  ║  NET SHIFT: {total_d:+.1%}  |  PICK: {pick} ({p_final:.1%})  |  CONF: {conf:.0%}")
    print(f"  ╚════════════════════════════════════════════════════════════════════╝")

    # ─── STAT COMPARISON ───
    print(f"\n  {'─' * 90}")
    print(f"  STAT EDGE MAP — who holds what advantage and by how much")
    print(f"  {'─' * 90}")
    print(f"  {'STAT':<25} {a.name:>14} {b.name:>14}  {'Δ':>8}  {'Edge':>15}")
    print(f"  {'─' * 80}")

    stat_edges_a = 0
    stat_edges_b = 0

    stats = [
        ("Adj EM",        a.adj_em,         b.adj_em,         True,  "Overall quality"),
        ("Adj O",         a.adj_o,          b.adj_o,          True,  "Offensive efficiency"),
        ("Adj D",         a.adj_d,          b.adj_d,          False, "Defensive efficiency"),
        ("Barthag",       a.barthag,        b.barthag,        True,  "Win probability"),
        ("3PT%",          a.three_p_pct*100, b.three_p_pct*100, True,  "Three-point shooting"),
        ("EFG%",          a.efg_pct*100,    b.efg_pct*100,    True,  "Effective FG%"),
        ("OREB%",         a.orb_pct,        b.orb_pct,        True,  "Offensive rebounding"),
        ("DREB%",         a.drb_pct,        b.drb_pct,        True,  "Defensive rebounding"),
        ("TOV%",          a.to_pct,         b.to_pct,         False, "Turnover rate"),
        ("AST%",          a.ast_pct,        b.ast_pct,        True,  "Ball movement"),
        ("FTR",           a.ftr,            b.ftr,            True,  "Free throw rate"),
        ("Pace",          a.pace,           b.pace,           True,  "Tempo"),
        ("Experience",    a.exp,            b.exp,            True,  "Roster experience"),
        ("Q1 Record",     a.q1_record,      b.q1_record,      True,  "Wins vs top 30"),
        ("SOS",           a.sos,            b.sos,            True,  "Schedule strength"),
        ("WAB",           a.wab if hasattr(a, 'wab') else 0, b.wab if hasattr(b, 'wab') else 0, True, "Wins above bubble"),
        ("Bench Depth",   a.bds,            b.bds,            True,  "Bench contribution"),
        ("3PT Variance",  a.three_pt_std,   b.three_pt_std,   True,  "Shooting volatility"),
        ("Momentum",      a.momentum,       b.momentum,       True,  "Recent form trend"),
        ("Off Burst",     a.offensive_burst, b.offensive_burst, True, "Ceiling scoring"),
    ]

    for name, va, vb, higher_better, desc in stats:
        delta = va - vb
        if higher_better:
            edge_team = a.name if va > vb else b.name
        else:
            edge_team = a.name if va < vb else b.name
            delta = -delta  # flip so positive means edge holder is better

        if edge_team == a.name:
            stat_edges_a += 1
        else:
            stat_edges_b += 1

        marker = "★" if abs(delta) > 3 else "▲" if abs(delta) > 1 else "·"
        print(f"  {name:<25} {va:>14.2f} {vb:>14.2f}  {delta:>+8.2f}  {marker} {edge_team}")

    print(f"\n  STAT EDGE COUNT: {a.name} holds {stat_edges_a}/{len(stats)} edges  |  "
          f"{b.name} holds {stat_edges_b}/{len(stats)} edges")

    # ─── BRANCH ENGINE ───
    print(f"\n  {'─' * 90}")
    print(f"  BRANCH ENGINE — scenario-specific conditional modifiers")
    print(f"  {'─' * 90}")

    branches_a = res_a.branches_fired
    branches_b = res_b.branches_fired

    if not branches_a and not branches_b:
        print(f"  No branches fired — ROOT probability passes through unchanged")
    else:
        print(f"\n  {a.name}: {len(branches_a)} branches, total shift = {res_a.total_shift:+.2%}")
        for br in branches_a:
            print(f"    ┌─ {br.branch_name}")
            print(f"    │  Severity: {br.severity:.3f}  |  Shift: {br.shift*100:+.2f}%  |  Max: {br.max_shift*100:.1f}%")
            print(f"    │  Direction: {br.direction}")
            if br.sub_conditions:
                for sc in br.sub_conditions:
                    status = "✓" if sc.met else "✗"
                    print(f"    │    {status} {sc.name}: {sc.description} (w={sc.weight:.2f})")
            print(f"    └─ {br.explanation}")

        print(f"\n  {b.name}: {len(branches_b)} branches, total shift = {res_b.total_shift:+.2%}")
        for br in branches_b:
            print(f"    ┌─ {br.branch_name}")
            print(f"    │  Severity: {br.severity:.3f}  |  Shift: {br.shift*100:+.2f}%  |  Max: {br.max_shift*100:.1f}%")
            print(f"    │  Direction: {br.direction}")
            if br.sub_conditions:
                for sc in br.sub_conditions:
                    status = "✓" if sc.met else "✗"
                    print(f"    │    {status} {sc.name}: {sc.description} (w={sc.weight:.2f})")
            print(f"    └─ {br.explanation}")

    # ─── ROUND SURVIVAL ───
    print(f"\n  {'─' * 90}")
    print(f"  S16 ROUND SURVIVAL — stat profile for this specific round")
    print(f"  (S16 profile: WAB 65% | 3PT% 52% | OREB% 60% | AdjO 63% | AdjD 60%)")
    print(f"  {'─' * 90}")

    net_surv = surv_a.survival_score - surv_b.survival_score
    surv_shift = p_survival - branch_prob
    print(f"\n  {a.name}: score={surv_a.survival_score:.3f}, "
          f"flags={len(surv_a.flags_triggered)}/{surv_a.flags_total}")
    for f in surv_a.flags_triggered:
        print(f"    ✓ {f.name} (w={f.weight:.2f}): {f.description}")
    missed_a = [f for f in [fl for fl in _get_all_flags(a, b)] if not f.met] if False else []

    print(f"\n  {b.name}: score={surv_b.survival_score:.3f}, "
          f"flags={len(surv_b.flags_triggered)}/{surv_b.flags_total}")
    for f in surv_b.flags_triggered:
        print(f"    ✓ {f.name} (w={f.weight:.2f}): {f.description}")

    print(f"\n  NET SURVIVAL: {net_surv:+.3f} → shift = {surv_shift:+.2%} toward {'favored' if surv_shift > 0 else 'underdog'}")

    # ─── FORM DIVERGENCE ───
    print(f"\n  {'─' * 90}")
    print(f"  FORM DIVERGENCE — are season stats lying?")
    print(f"  {'─' * 90}")

    for team_name, form, sg, log in [(a.name, form_a, sg_a, log_a), (b.name, form_b, sg_b, log_b)]:
        print(f"\n  {team_name}: pattern={form.pattern}, severity={form.severity:.3f}, "
              f"shift={form.shift:+.2%}")
        print(f"    EM divergence (last10 - season): {form.em_divergence:+.1f}")
        print(f"    Margin divergence: {form.margin_divergence:+.1f}")
        print(f"    Early→Late shift: {form.early_late_shift:+.1f}")
        if sg > 0:
            print(f"    ⚡ SLEEPING GIANT score: {sg:.2f}")
        if form.flags:
            print(f"    Flags ({len([f for f in form.flags if f.met])}/{len(form.flags)} triggered):")
            for f in form.flags:
                status = "✓" if f.met else "✗"
                print(f"      {status} {f.name} (w={f.weight:.2f}): {f.description}")

        # Game log snapshot
        if log:
            print(f"\n    GAME LOG SNAPSHOT:")
            print(f"      Season EM:     {log.season_em:>7.1f}")
            print(f"      Last 10 EM:    {log.last10_em:>7.1f}  ({log.last10_em - log.season_em:+.1f})")
            print(f"      Last 5 EM:     {log.last5_em:>7.1f}  ({log.last5_em - log.season_em:+.1f})")
            print(f"      First half EM: {log.first_half_em:>7.1f}")
            print(f"      Second half EM:{log.second_half_em:>7.1f}")
            print(f"      Trend:         {'↗ RISING' if log.last5_em > log.last10_em else '↘ FALLING' if log.last5_em < log.last10_em else '→ FLAT'}")

    form_shift = p_final - p_survival
    print(f"\n  FORM NET SHIFT: {form_shift:+.2%}")

    # ─── EDGE SUMMARY ───
    print(f"\n  {'═' * 90}")
    print(f"  EDGE SUMMARY")
    print(f"  {'═' * 90}")

    edges = []
    # Stat dominance
    if stat_edges_a >= 14:
        edges.append(f"STATISTICAL DOMINANCE: {a.name} holds {stat_edges_a}/20 stat edges")
    elif stat_edges_b >= 14:
        edges.append(f"STATISTICAL DOMINANCE: {b.name} holds {stat_edges_b}/20 stat edges")

    # Key S16 metrics
    if a.q1_record > b.q1_record + 0.1:
        edges.append(f"Q1 RESUME: {a.name} ({a.q1_record:.2f}) >> {b.name} ({b.q1_record:.2f})")
    elif b.q1_record > a.q1_record + 0.1:
        edges.append(f"Q1 RESUME: {b.name} ({b.q1_record:.2f}) >> {a.name} ({a.q1_record:.2f})")

    if abs(a.adj_em - b.adj_em) > 5:
        better = a if a.adj_em > b.adj_em else b
        worse = b if better == a else a
        edges.append(f"EM GAP ({abs(a.adj_em - b.adj_em):.1f}): {better.name} has clear quality advantage")

    if abs(a.orb_pct - b.orb_pct) > 2:
        better = a if a.orb_pct > b.orb_pct else b
        edges.append(f"OREB EDGE: {better.name} ({better.orb_pct:.1f}%) — second chances matter at S16")

    # 3PT% at S16
    if abs(a.three_p_pct - b.three_p_pct) > 0.02:
        better = a if a.three_p_pct > b.three_p_pct else b
        worse = b if better == a else a
        edges.append(f"3PT% EDGE: {better.name} ({better.three_p_pct:.1%} vs {worse.three_p_pct:.1%})")

    # Experience
    if abs(a.exp - b.exp) > 0.3:
        better = a if a.exp > b.exp else b
        worse = b if better == a else a
        edges.append(f"EXPERIENCE: {better.name} ({better.exp:.2f}) vs {worse.name} ({worse.exp:.2f})")

    # Branches
    if branches_a:
        for br in branches_a:
            if abs(br.shift) > 0.01:
                edges.append(f"BRANCH [{br.branch_name}]: {br.shift*100:+.1f}% toward {a.name}")
    if branches_b:
        for br in branches_b:
            if abs(br.shift) > 0.01:
                edges.append(f"BRANCH [{br.branch_name}]: {br.shift*100:+.1f}% toward {b.name}")

    # Form
    if form_a.severity > 0.1:
        edges.append(f"FORM [{form_a.pattern}]: {a.name} sev={form_a.severity:.2f} → {form_a.shift:+.1%}")
    if form_b.severity > 0.1:
        edges.append(f"FORM [{form_b.pattern}]: {b.name} sev={form_b.severity:.2f} → {form_b.shift:+.1%}")

    # Sleeping giant
    if sg_a > 0.3:
        edges.append(f"⚡ SLEEPING GIANT: {a.name} (score={sg_a:.2f})")
    if sg_b > 0.3:
        edges.append(f"⚡ SLEEPING GIANT: {b.name} (score={sg_b:.2f})")

    if not edges:
        print(f"  No significant edges detected — true toss-up")
    else:
        for i, e in enumerate(edges, 1):
            print(f"  {i}. {e}")

    # Toss-up warning
    if conf < 0.15:
        print(f"\n  ⚠️  TOSS-UP ALERT: confidence only {conf:.0%} — model sees this as a coin flip")
    elif conf < 0.30:
        print(f"\n  ⚠️  LOW CONFIDENCE: {conf:.0%} — small edges, high upset potential")


# ═══════════════════════════════════════════════════════════════════════════
# QUICK SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═' * 100}")
print(f"  QUICK REFERENCE — ALL S16 PICKS")
print(f"{'═' * 100}")
print(f"\n  {'Region':<10} {'Matchup':<42} {'Pick':<20} {'Prob':>7} {'Conf':>7} {'Key Edge'}")
print(f"  {'─' * 95}")
