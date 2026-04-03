# Model Improvement Plan — Post Day 1

## IMMEDIATE PRIORITY: Update for Round of 32

### Step 1: Eliminate Dead Teams, Update Bracket
Teams eliminated Thursday: Ohio State, Siena, South Florida, North Dakota State, Troy, McNeese State, North Carolina, Penn, Saint Mary's, Idaho, Howard, Georgia, Wisconsin, BYU, Kennesaw State, Hawaii

Remaining R32 matchups (Thursday winners):
- **East**: Duke vs TCU, Louisville vs Michigan State
- **West**: High Point vs Gonzaga (or Arizona), Texas vs Gonzaga
- **South**: Nebraska vs Vanderbilt, VCU vs Illinois, Texas A&M vs Houston
- **Midwest**: Michigan vs Saint Louis

Friday games still needed before full R32 bracket

### Step 2: New Parameters to Add

#### A. Star Isolation Index (SII) — addresses BYU/Dybantsa problem
**Data source**: EvanMiya_Players.csv (bpr, poss columns)
**Formula**:
```
SII = (star_bpr / team_total_bpr) * (star_poss / team_total_poss)
```
If SII > 0.25 → flag as "star-dependent"
If SII > 0.35 → flag as "star-isolated" (penalty)
**Rationale**: BYU's offense was 100% Dybantsa after Saunders injury. Model didn't capture that one-man-show = tournament death

#### B. Tournament Rebounding Emphasis (TRE)
**Data source**: KenPom Barttorvik.csv (ORB%, DRB%)
**Current weight**: rbm = 0.0882
**Proposed**: Create tournament-specific rebound metric:
```
TRE = 0.6 * ORB% + 0.4 * DRB%  (offensive boards matter MORE in March)
```
Boost weight from 0.0882 to 0.10-0.11 for tournament predictions
**Rationale**: Every upset Thursday involved rebounding dominance

#### C. Conference Strength Adjustment (CSA)
**Data source**: Conference tournament results + Day 1 outcomes
**Formula**:
```
CSA = (conf_wins_day1 / conf_teams_in_tourney) * conf_sos_rank
```
Big 12 CSA should be higher than Big Ten based on physical play style translating to March
**Implementation**: Multiply team's base composite by (1 + 0.02 * CSA_rank)

#### D. Composure Under Pressure (CUP) — addresses Wisconsin/UNC problem
**Data source**: Close game record (games decided by ≤5 pts), FT% in clutch
**Formula**:
```
CUP = 0.4 * close_game_win_pct + 0.3 * clutch_ft_pct + 0.3 * (1 - late_game_turnover_rate)
```
**Current proxy**: clutch_factor (weight 0.0504) partially captures this
**Enhancement**: Split clutch_factor into regular clutch + tournament pressure adjustment

#### E. Assist Rate Change Post-Injury (Chemistry Proxy)
**Data source**: Team assists pre-injury vs post-injury from game logs
**Formula**:
```
chemistry_delta = (ast_rate_post_injury / ast_rate_pre_injury) - 1.0
```
If chemistry_delta < -0.15 → additional injury penalty multiplier of 1.3x
**Rationale**: UNC's assist rate almost certainly dropped after Wilson injury → selfish play

### Step 3: Recalibrate Existing Weights

#### Overconfident predictions to fix:
| Game | Model Prob | Actual | Issue |
|------|-----------|--------|-------|
| Gonzaga 96.4% | Won 73-64 | Barely survived | Model too confident on 3-seeds |
| Duke 99.6% | Won 71-65 | Trailed at half | Model too confident on 1-seeds with injuries |
| Wisconsin 85.9% | Lost 82-83 | Upset | Model underweighted specialist/unicorn factor |
| UNC 78.5% | Lost 78-82 OT | Upset | Injury model under-penalized star loss on young team |
| St. Mary's 78.2% | Lost 50-63 | Upset | Model over-valued WCC strength |

#### Proposed weight adjustments:
1. **exp (experience)**: 0.0127 → 0.020 — experience proved critical (UNC young = lost, VCU veteran = won)
2. **rbm (rebounding)**: 0.0882 → 0.10 — rebounding decided multiple games
3. **form_trend**: 0.0150 → 0.020 — recent form matters (BYU 2-4 after Saunders)
4. **bds (bench depth)**: 0.0094 → 0.015 — Duke's 7-man rotation nearly cost them; Siena had zero subs
5. **ast_pct**: 0.0364 → 0.04 — ball movement = team chemistry = wins

### Step 4: New Flags for R32 Predictions

#### UNICORN_PLAYER flag
Players who are elite at ONE thing and can single-handedly swing a game:
- Chase Johnston (High Point): 3PT specialist, 45.2% from 3
- Terrence Hill Jr. (VCU): Scoring burst potential (34 pts in R64)
- Tavari Johnson (Akron): 5'11" scoring guard, 20.1 PPG
- Nate Johnson (Akron): MAC POY + DPOY (dual unicorn)

These players widen the outcome distribution → increase upset probability by 3-5%

#### FATIGUE_RISK flag
Teams that relied heavily on one player in R64:
- High Point: Johnston carried scoring
- VCU: Hill Jr. carried scoring (34 pts)
- Texas: Need to check minute distribution
- Any team with a player who logged 38+ minutes in R64

#### REVENGE/MOMENTUM flag
- Texas beat BYU (Big 12 team) → carries conference confidence into R32
- VCU's historic 19-pt comeback → massive psychological boost
- Nebraska's first-ever NCAA tournament win → could either relax or peak

### Step 5: R32 Matchup-Specific Adjustments

#### VCU vs Illinois
- **Key question**: Can Boswell (Illinois lockdown defender) neutralize Terrence Hill Jr.?
- If yes: Illinois wins comfortably. Boswell guarded Penn's best player all game → can do it again
- If no: VCU's scoring burst potential makes this dangerous
- **Model adjustment**: Factor in Illinois's defensive versatility vs VCU's star scorer

#### High Point vs [Arizona/Gonzaga winner]
- Johnston is a one-man 3PT army. If he gets hot, anything can happen
- But: Arizona/Gonzaga have much better perimeter defense than Wisconsin
- **Model adjustment**: Apply 3PT variance factor. Johnston's true upset probability is higher than composite suggests

#### Louisville +/- M. Brown Jr. for R32
- Brown could return (R32 play_prob: 0.40). If he plays even limited minutes:
  - Scoring drought risk drops dramatically
  - Team synergy improves (already showed great flow without him)
- **Model adjustment**: Update Brown Jr. play_prob based on latest reports. Louisville WITH Brown is a different team

### Step 6: Data Pipeline Changes

1. **Create `data/live_results/` directory** ✅ (done)
2. **Save Day 1 results CSV** ✅ (done)
3. **Update matchups.csv** for R32 with actual bracket paths
4. **Update injuries.csv** with post-R64 injury updates
5. **Create `data/tournament_stats.csv`** with game-level stats from Day 1
6. **Re-run main.py** for R32 predictions with updated weights
