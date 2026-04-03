"""TEXAS DEEP DIVE — What's driving their upset run?

Texas (11-seed) beat:
  R64: ??? (need to check bracket)
  R32: (3) Gonzaga

The model has them at last5_EM = -15.5 (catastrophic collapse).
But the user flags that their TOURNAMENT games show dominance.

This script analyzes:
1. Full season arc — where did they peak, where did they crater
2. Win DNA vs Loss DNA — what stats separate Texas W from Texas L
3. Last 5 games vs their actual TOURNAMENT games (which might not be in game logs)
4. What features make Texas dangerous when they're ON
5. Head-to-head with Purdue's profile
"""

import sys, os, csv, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

base = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
# LOAD TEXAS GAME LOG (+ optional NCAA supplement — same schema as Texas.csv)
# ═══════════════════════════════════════════════════════════════════════════
log_path = os.path.join(base, "archive-3", "game-logs", "Texas.csv")
supplement_path = os.path.join(base, "archive-3", "game-logs", "Texas_NCAA2026_supplement.csv")

primary_rows = []
with open(log_path, 'r') as f:
    for row in csv.DictReader(f):
        row['_source'] = 'regular'
        primary_rows.append(row)

supp_rows = []
if os.path.isfile(supplement_path):
    with open(supplement_path, 'r') as f:
        for row in csv.DictReader(f):
            row['_source'] = 'ncaa_supplement'
            supp_rows.append(row)

games = primary_rows + supp_rows
games.sort(key=lambda r: r.get('date', ''))

print("=" * 100)
print("  TEXAS DEEP DIVE — WHAT MAKES THEM DANGEROUS?")
print("=" * 100)

print(f"\n  Total games in merged log: {len(games)} ({len(primary_rows)} regular + {len(supp_rows)} NCAA supplement)")

# ═══════════════════════════════════════════════════════════════════════════
# PARSE GAME-BY-GAME
# ═══════════════════════════════════════════════════════════════════════════

parsed = []
for g in games:
    try:
        score_t = float(g['score_t'])
        score_o = float(g['score_o'])
        rating_t = float(g['rating_t'])
        rating_o = float(g['rating_o'])
        poss_t = float(g['poss_t'])
        poss_o = float(g['poss_o'])
        opp_rank = int(g['opp_rank'])
        result = g['result']
        opponent = g['opponent']
        date = g['date']
        venue = g['venue']

        margin = score_t - score_o
        em = rating_t - rating_o
        pace = (poss_t + poss_o) / 2

        parsed.append({
            'opponent': opponent, 'date': date, 'venue': venue,
            'result': result, 'opp_rank': opp_rank,
            'score_t': score_t, 'score_o': score_o,
            'margin': margin, 'em': em,
            'rating_t': rating_t, 'rating_o': rating_o,
            'pace': pace,
            'raw_score_t': int(g['t_score_t']), 'raw_score_o': int(g['t_score_o']),
            'raw_rating_t': float(g['t_rating_t']), 'raw_rating_o': float(g['t_rating_o']),
            'source': g.get('_source', 'regular'),
        })
    except (ValueError, KeyError):
        continue

# ═══════════════════════════════════════════════════════════════════════════
# FULL SEASON ARC
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 100}")
print(f"  FULL SEASON ARC — every game")
print(f"{'─' * 100}")
print(f"  {'Date':<12} {'Opponent':<25} {'Rk':>4} {'Venue':<8} {'Result':<4} {'Score':>8} {'Margin':>7} {'Adj EM':>8} {'Off':>7} {'Def':>7} {'Pace':>6}")
print(f"  {'─'*95}")

running_em = []
for i, g in enumerate(parsed):
    running_em.append(g['em'])
    print(f"  {g['date']:<12} {g['opponent']:<25} {g['opp_rank']:>4} {g['venue']:<8} {g['result']:<4} "
          f"{g['raw_score_t']:>3}-{g['raw_score_o']:<3} {g['margin']:>+7.1f} {g['em']:>+8.1f} "
          f"{g['rating_t']:>7.1f} {g['rating_o']:>7.1f} {g['pace']:>6.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# SEASON PHASES
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  SEASON PHASES")
print(f"{'─' * 100}")

# Split into thirds
n = len(parsed)
third = n // 3
phases = {
    'Early (games 1-10)': parsed[:10],
    'Middle (games 11-21)': parsed[10:21],
    'Late (games 22-32)': parsed[21:],
}

for phase_name, phase_games in phases.items():
    wins = [g for g in phase_games if g['result'] == 'W']
    losses = [g for g in phase_games if g['result'] == 'L']
    avg_em = statistics.mean([g['em'] for g in phase_games]) if phase_games else 0
    avg_off = statistics.mean([g['rating_t'] for g in phase_games]) if phase_games else 0
    avg_def = statistics.mean([g['rating_o'] for g in phase_games]) if phase_games else 0
    avg_margin = statistics.mean([g['margin'] for g in phase_games]) if phase_games else 0
    avg_pace = statistics.mean([g['pace'] for g in phase_games]) if phase_games else 0
    avg_opp_rank = statistics.mean([g['opp_rank'] for g in phase_games]) if phase_games else 0

    q1_games = [g for g in phase_games if g['opp_rank'] <= 30]
    q1_wins = [g for g in q1_games if g['result'] == 'W']

    print(f"\n  {phase_name}:")
    print(f"    Record: {len(wins)}-{len(losses)} | Avg EM: {avg_em:+.1f} | Avg Off: {avg_off:.1f} | Avg Def: {avg_def:.1f}")
    print(f"    Avg Margin: {avg_margin:+.1f} | Avg Pace: {avg_pace:.1f} | Avg Opp Rank: {avg_opp_rank:.0f}")
    print(f"    Q1 games: {len(q1_wins)}/{len(q1_games)}")

# ═══════════════════════════════════════════════════════════════════════════
# WIN DNA vs LOSS DNA
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  WIN DNA vs LOSS DNA — What features separate Texas W from Texas L")
print(f"{'─' * 100}")

wins = [g for g in parsed if g['result'] == 'W']
losses = [g for g in parsed if g['result'] == 'L']

def avg(lst, key):
    vals = [g[key] for g in lst]
    return statistics.mean(vals) if vals else 0

print(f"\n  {'Feature':<30} {'WINS ({})'.format(len(wins)):>15} {'LOSSES ({})'.format(len(losses)):>15} {'Δ':>10}")
print(f"  {'─'*75}")

features = [
    ('Adj EM', 'em'),
    ('Offensive Rating', 'rating_t'),
    ('Defensive Rating', 'rating_o'),
    ('Raw Points Scored', 'raw_score_t'),
    ('Raw Points Allowed', 'raw_score_o'),
    ('Margin', 'margin'),
    ('Pace', 'pace'),
    ('Opp Rank', 'opp_rank'),
]

for name, key in features:
    w_avg = avg(wins, key)
    l_avg = avg(losses, key)
    delta = w_avg - l_avg
    print(f"  {name:<30} {w_avg:>15.1f} {l_avg:>15.1f} {delta:>+10.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY WINS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  QUALITY WINS — Texas's best performances")
print(f"{'─' * 100}")

quality_wins = sorted([g for g in wins if g['opp_rank'] <= 60], key=lambda x: x['opp_rank'])
for g in quality_wins:
    print(f"  {g['date']} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"{g['raw_score_t']}-{g['raw_score_o']} ({g['venue']}) "
          f"EM={g['em']:+.1f} Off={g['rating_t']:.1f} Def={g['rating_o']:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# LAST 10 / LAST 5 DETAILED — What does the "collapse" look like
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  LAST 10 & LAST 5 — The alleged 'collapse'")
print(f"{'─' * 100}")

last10 = parsed[-10:]
last5 = parsed[-5:]

print(f"\n  LAST 10 GAMES:")
for g in last10:
    marker = "★" if g['result'] == 'W' else "✗"
    print(f"  {marker} {g['date']} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"{g['raw_score_t']}-{g['raw_score_o']} ({g['venue']}) EM={g['em']:+.1f}")

l10_em = statistics.mean([g['em'] for g in last10])
l10_off = statistics.mean([g['rating_t'] for g in last10])
l10_def = statistics.mean([g['rating_o'] for g in last10])
l10_wins = len([g for g in last10 if g['result'] == 'W'])
print(f"\n  Last 10 summary: {l10_wins}-{10-l10_wins} | EM: {l10_em:+.1f} | Off: {l10_off:.1f} | Def: {l10_def:.1f}")

print(f"\n  LAST 5 GAMES:")
for g in last5:
    marker = "★" if g['result'] == 'W' else "✗"
    print(f"  {marker} {g['date']} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"{g['raw_score_t']}-{g['raw_score_o']} ({g['venue']}) EM={g['em']:+.1f}")

l5_em = statistics.mean([g['em'] for g in last5])
l5_off = statistics.mean([g['rating_t'] for g in last5])
l5_def = statistics.mean([g['rating_o'] for g in last5])
l5_wins = len([g for g in last5 if g['result'] == 'W'])
print(f"\n  Last 5 summary: {l5_wins}-{5-l5_wins} | EM: {l5_em:+.1f} | Off: {l5_off:.1f} | Def: {l5_def:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# TEXAS's UPSET PATTERN — When they beat ranked teams, what's different?
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  UPSET PATTERN — When Texas beats ranked teams, what drives it?")
print(f"{'─' * 100}")

ranked_wins = [g for g in wins if g['opp_rank'] <= 50]
ranked_losses = [g for g in losses if g['opp_rank'] <= 50]

if ranked_wins:
    print(f"\n  RANKED WINS ({len(ranked_wins)} games):")
    for g in ranked_wins:
        print(f"    vs #{g['opp_rank']:>3} {g['opponent']:<20} {g['raw_score_t']}-{g['raw_score_o']} "
              f"EM={g['em']:+.1f} Off={g['rating_t']:.1f} Def={g['rating_o']:.1f} Pace={g['pace']:.1f}")

    print(f"\n    Avg: Off={avg(ranked_wins, 'rating_t'):.1f} Def={avg(ranked_wins, 'rating_o'):.1f} "
          f"Pace={avg(ranked_wins, 'pace'):.1f} Margin={avg(ranked_wins, 'margin'):+.1f}")

if ranked_losses:
    print(f"\n  RANKED LOSSES ({len(ranked_losses)} games):")
    for g in ranked_losses:
        print(f"    vs #{g['opp_rank']:>3} {g['opponent']:<20} {g['raw_score_t']}-{g['raw_score_o']} "
              f"EM={g['em']:+.1f} Off={g['rating_t']:.1f} Def={g['rating_o']:.1f} Pace={g['pace']:.1f}")

    print(f"\n    Avg: Off={avg(ranked_losses, 'rating_t'):.1f} Def={avg(ranked_losses, 'rating_o'):.1f} "
          f"Pace={avg(ranked_losses, 'pace'):.1f} Margin={avg(ranked_losses, 'margin'):+.1f}")

if ranked_wins and ranked_losses:
    print(f"\n  ╔══ KEY DIFFERENCES (Upset Wins vs Ranked Losses) ══╗")
    diff_off = avg(ranked_wins, 'rating_t') - avg(ranked_losses, 'rating_t')
    diff_def = avg(ranked_wins, 'rating_o') - avg(ranked_losses, 'rating_o')
    diff_pace = avg(ranked_wins, 'pace') - avg(ranked_losses, 'pace')
    diff_pts = avg(ranked_wins, 'raw_score_t') - avg(ranked_losses, 'raw_score_t')
    diff_pts_a = avg(ranked_wins, 'raw_score_o') - avg(ranked_losses, 'raw_score_o')

    print(f"  ║  Offensive Rating:  {diff_off:>+8.1f} (higher when winning)")
    print(f"  ║  Defensive Rating:  {diff_def:>+8.1f} ({'better' if diff_def < 0 else 'worse'} when winning)")
    print(f"  ║  Pace:              {diff_pace:>+8.1f} ({'faster' if diff_pace > 0 else 'slower'} when winning)")
    print(f"  ║  Points Scored:     {diff_pts:>+8.1f}")
    print(f"  ║  Points Allowed:    {diff_pts_a:>+8.1f}")
    print(f"  ╚══════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════
# VENUE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  VENUE SPLIT — Texas at home vs away vs neutral")
print(f"{'─' * 100}")

for venue in ['home', 'away', 'neutral']:
    v_games = [g for g in parsed if g['venue'] == venue]
    v_wins = [g for g in v_games if g['result'] == 'W']
    if v_games:
        v_em = statistics.mean([g['em'] for g in v_games])
        v_off = statistics.mean([g['rating_t'] for g in v_games])
        v_def = statistics.mean([g['rating_o'] for g in v_games])
        print(f"  {venue.upper():<10} {len(v_wins)}-{len(v_games)-len(v_wins)} | EM: {v_em:+.1f} | Off: {v_off:.1f} | Def: {v_def:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# TOURNAMENT CONTEXT — R64 and R32 results
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  TOURNAMENT GAMES — What happened in March")
print(f"{'─' * 100}")

last_regular = primary_rows[-1]['date'] if primary_rows else '(none)'
print(f"\n  Base file Texas.csv ends after conference tournament: last row {last_regular}.")
print(f"  NCAA tournament games are not in that export unless you add a supplement CSV.")
if supp_rows:
    print(f"  Loaded NCAA rows from: {os.path.basename(supplement_path)} ({len(supp_rows)} games).")

ncaa_merged = [g for g in parsed if g['source'] == 'ncaa_supplement']
march_games = [g for g in parsed if g['date'] >= '2026-03-01']
print(f"\n  All March games in merged log ({len(march_games)}) — conf + NCAA:")
for g in march_games:
    marker = "★" if g['result'] == 'W' else "✗"
    tag = "[NCAA]" if g['source'] == 'ncaa_supplement' else "[conf/reg]"
    print(f"  {marker} {g['date']} {tag} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"{g['raw_score_t']}-{g['raw_score_o']} ({g['venue']}) EM={g['em']:+.1f}")
if ncaa_merged:
    avg_ncaa_em = statistics.mean([g['em'] for g in ncaa_merged])
    print(f"\n  NCAA-only (supplement): {len(ncaa_merged)} games | mean Adj EM = {avg_ncaa_em:+.1f}")
else:
    print(f"\n  No NCAA supplement rows — only late-season/conf games appear above.")

# ═══════════════════════════════════════════════════════════════════════════
# THE KEY QUESTION: What does Texas look like when they're ON?
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═' * 100}")
print(f"  THE BIG QUESTION: WHAT FEATURES DRIVE TEXAS WHEN THEY'RE ON?")
print(f"{'═' * 100}")

# Separate high-performance games (top quartile by EM) from low-performance
ems = [g['em'] for g in parsed]
median_em = statistics.median(ems)
top_games = sorted(parsed, key=lambda x: x['em'], reverse=True)[:8]  # Top quarter
bot_games = sorted(parsed, key=lambda x: x['em'])[:8]  # Bottom quarter

print(f"\n  TOP 8 GAMES (by Adj EM):")
for g in top_games:
    print(f"    {g['date']} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"EM={g['em']:+.1f} Off={g['rating_t']:.1f} Def={g['rating_o']:.1f} Pace={g['pace']:.1f}")

print(f"\n    TOP 8 AVG: Off={avg(top_games, 'rating_t'):.1f} Def={avg(top_games, 'rating_o'):.1f} "
      f"Pace={avg(top_games, 'pace'):.1f} Score={avg(top_games, 'raw_score_t'):.0f}-{avg(top_games, 'raw_score_o'):.0f}")

print(f"\n  BOTTOM 8 GAMES (by Adj EM):")
for g in bot_games:
    print(f"    {g['date']} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"EM={g['em']:+.1f} Off={g['rating_t']:.1f} Def={g['rating_o']:.1f} Pace={g['pace']:.1f}")

print(f"\n    BOT 8 AVG: Off={avg(bot_games, 'rating_t'):.1f} Def={avg(bot_games, 'rating_o'):.1f} "
      f"Pace={avg(bot_games, 'pace'):.1f} Score={avg(bot_games, 'raw_score_t'):.0f}-{avg(bot_games, 'raw_score_o'):.0f}")

# Feature deltas between top and bottom quartile
print(f"\n  ╔══ FEATURE IMPORTANCE: What changes between Texas's BEST and WORST games ══╗")
diff_off = avg(top_games, 'rating_t') - avg(bot_games, 'rating_t')
diff_def = avg(top_games, 'rating_o') - avg(bot_games, 'rating_o')
diff_pace = avg(top_games, 'pace') - avg(bot_games, 'pace')
diff_pts = avg(top_games, 'raw_score_t') - avg(bot_games, 'raw_score_t')
diff_pts_a = avg(top_games, 'raw_score_o') - avg(bot_games, 'raw_score_o')

features_ranked = [
    (abs(diff_def), f"DEFENSIVE RATING: {diff_def:+.1f} ({'tightens' if diff_def < 0 else 'loosens'} in good games)"),
    (abs(diff_off), f"OFFENSIVE RATING: {diff_off:+.1f} (offense {'rises' if diff_off > 0 else 'drops'} in good games)"),
    (abs(diff_pace), f"PACE:             {diff_pace:+.1f} ({'faster' if diff_pace > 0 else 'slower'} in good games)"),
    (abs(diff_pts), f"POINTS SCORED:    {diff_pts:+.1f}"),
    (abs(diff_pts_a), f"POINTS ALLOWED:   {diff_pts_a:+.1f}"),
]
features_ranked.sort(reverse=True)

for i, (magnitude, desc) in enumerate(features_ranked, 1):
    importance = "★★★" if magnitude > 15 else "★★" if magnitude > 8 else "★"
    print(f"  ║  #{i} {importance} {desc}")

print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════════════
# TEXAS vs ALABAMA (they played Jan 10)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'─' * 100}")
print(f"  HEAD-TO-HEAD: Texas beat Alabama 92-88 (Jan 10, away)")
print(f"{'─' * 100}")

bama_game = [g for g in parsed if g['opponent'] == 'Alabama']
if bama_game:
    g = bama_game[0]
    print(f"  Score: {g['raw_score_t']}-{g['raw_score_o']} (Texas WIN, at Alabama)")
    print(f"  EM: {g['em']:+.1f} | Off: {g['rating_t']:.1f} | Def: {g['rating_o']:.1f} | Pace: {g['pace']:.1f}")
    print(f"  Texas scored 131.4 points per 100 possessions AGAINST Alabama")
    print(f"  This suggests: when Texas's offense is clicking, they can score on ANYONE")

# Also check the teams they beat recently
print(f"\n\n{'─' * 100}")
print(f"  TEXAS vs PURDUE-LIKE PROFILES")
print(f"  (Teams with strong offense + moderate defense, like Purdue)")
print(f"{'─' * 100}")

# Find games against teams ranked 20-50 (similar to Purdue's profile range)
mid_tier = [g for g in parsed if 20 <= g['opp_rank'] <= 50]
print(f"\n  Games vs #20-50 ranked teams ({len(mid_tier)} games):")
for g in mid_tier:
    marker = "★" if g['result'] == 'W' else "✗"
    print(f"  {marker} {g['date']} vs #{g['opp_rank']:>3} {g['opponent']:<20} "
          f"{g['raw_score_t']}-{g['raw_score_o']} EM={g['em']:+.1f}")
mid_wins = [g for g in mid_tier if g['result'] == 'W']
print(f"  Record vs #20-50: {len(mid_wins)}-{len(mid_tier)-len(mid_wins)}")

# ═══════════════════════════════════════════════════════════════════════════
# SLEEPING GIANT INDICATORS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n\n{'═' * 100}")
print(f"  SLEEPING GIANT ASSESSMENT")
print(f"{'═' * 100}")

print(f"""
  CASE FOR TEXAS AS SLEEPING GIANT:
  1. Beat Alabama 92-88 AWAY (Jan 10) — Off rating 131.4
  2. Beat Vanderbilt (#14) 80-64 at home — dominant
  3. Beat NC State (#39) 102-97 neutral — high-scoring affair
  4. Experience: 2.74 (highest on Purdue's side of the bracket)
  5. FTR: 0.35 (gets to the line more than ANY S16 team)
  6. Q3 adjustment: +2.3 (gets BETTER after halftime — unlike Purdue's -2.7)

  CASE AGAINST:
  1. Last 5 season games: 1-4 record
  2. Lost to Oklahoma (home), Ole Miss (neutral), Arkansas (away) — all bad losses
  3. Season EM only 6.5 (lowest in S16 field by far)
  4. Last 5 EM: -15.5 (worst in tournament field)
  5. Defense is season-long liability: Adj D = 105.9

  THE KEY QUESTION: Are the tournament games a NEW Texas or a dead cat bounce?
""")

# Rolling window analysis
print(f"\n{'─' * 100}")
print(f"  ROLLING 5-GAME EM WINDOWS — Finding the pattern")
print(f"{'─' * 100}")

for i in range(len(parsed) - 4):
    window = parsed[i:i+5]
    w_em = statistics.mean([g['em'] for g in window])
    w_off = statistics.mean([g['rating_t'] for g in window])
    w_def = statistics.mean([g['rating_o'] for g in window])
    w_wins = len([g for g in window if g['result'] == 'W'])
    dates = f"{window[0]['date']} to {window[-1]['date']}"
    bar = "█" * int(max(0, (w_em + 30) / 3))  # visual bar
    print(f"  Games {i+1:>2}-{i+5:<2} ({dates}): EM={w_em:>+7.1f} Off={w_off:>6.1f} Def={w_def:>6.1f} W-L={w_wins}-{5-w_wins} {bar}")

print(f"\n\n{'═' * 100}")
print(f"  FINAL ASSESSMENT: WHAT'S THE MOST IMPORTANT FEATURE?")
print(f"{'═' * 100}")
