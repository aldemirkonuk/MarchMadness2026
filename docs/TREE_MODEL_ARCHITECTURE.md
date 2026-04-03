# Tree Model Architecture — "One Root, Many Branches"

## The Backtest Verdict

Model A (CORE_WEIGHTS): 73.6% on 1,070 historical games (2008-2025)
Model B (TOURNAMENT_WEIGHTS): 71.0% — DROPS 2.5%

But WHERE it drops is revealing:
- High-confidence chalk (>80%): **IDENTICAL** (78.4% both)
- Mid-confidence (65-80%): **Near-identical** (-0.9%)
- Close games (35-65%): **DESTROYED** (-13.3%)

Model B flipped 37 correct Model A calls to wrong, rescued only 10.
The problem: `top25_perf` (7.5% of Model A) was the tiebreaker for close games.
Removing it meant the model lost its best discriminator when teams are similar.

## The Insight

Applying tournament adjustments UNIFORMLY to all games hurts close-game
accuracy. But the tournament adjustments ARE correct for specific scenarios:
- BYU's star isolation (Dybantsa 40 min, still lost)
- UNC's chemistry collapse (19-pt lead → OT loss)
- Wisconsin's composure failure (49 pts from guards, still lost)
- Siena's fatigue trap (zero subs → 7 min scoreless)

These aren't universal truths. They're BRANCH conditions that override the
base prediction only when triggered.

## Architecture: The Tree

```
                    ┌──────────────────────────┐
                    │    ROOT: Model A (73.7%)  │
                    │   CORE_WEIGHTS backbone   │
                    │  All 35 params, optimized │
                    │   Backtested. Bulletproof.│
                    └────────────┬─────────────┘
                                 │
              Base probability P_base from Model A
                                 │
                    ┌────────────┴─────────────┐
                    │   BRANCH DETECTOR ENGINE  │
                    │   Scans for trigger conds │
                    └────────────┬─────────────┘
                                 │
            ┌────────┬───────────┼────────────┬─────────┐
            │        │           │            │         │
            ▼        ▼           ▼            ▼         ▼
       ┌─────────┐ ┌──────┐ ┌────────┐ ┌──────────┐ ┌───────┐
       │STAR     │ │FATIGUE│ │CHEMISTRY│ │EXPERIENCE│ │UNICORN│
       │ISOLATION│ │TRAP   │ │COLLAPSE │ │GAP       │ │PLAYER │
       └────┬────┘ └──┬───┘ └───┬────┘ └────┬─────┘ └──┬────┘
            │         │         │            │          │
            ▼         ▼         ▼            ▼          ▼
       Prob shift  Prob shift  Prob shift  Prob shift  Variance
       toward      toward      toward      toward      widen
       underdog    underdog    underdog    veteran     (+/- upset
       (-0.05      (-0.03      (-0.08      (+0.03      probability)
        to -0.15)   to -0.08)   to -0.15)   to +0.08)
```

## The Key Principle

> The ROOT never changes. It's the 73.7% backtested foundation.
> BRANCHES are conditional modifiers. Each branch has:
>   1. A TRIGGER condition (boolean: does this scenario apply?)
>   2. A MAGNITUDE (how much to shift probability)
>   3. A DIRECTION (toward upset or toward chalk)
>   4. A CONFIDENCE (how sure are we this branch matters?)
>
> Final probability = P_base + Σ(branch_shift × branch_confidence)
> Capped so total shift never exceeds ±0.15 per matchup.

## Branch Catalog (14 branches)

### TIER 1: PLAYER-LEVEL BRANCHES (highest impact)

#### Branch 1: STAR_ISOLATION
**Trigger**: Team's SII > 0.12 (one player IS the offense)
  OR #2 option is injured AND top player's usage > 35%
**Magnitude**: -0.05 to -0.12 (shift toward underdog beating them)
**Evidence**: BYU 2026 (Dybantsa 35 pts, 40 min, still lost by 8)
**Why it works**: Predictable offense. Fatigue in crunch time. One
  player can't sustain 40 minutes of maximum output in March pressure.
**Data source**: EvanMiya_Players.csv (bpr, poss columns)
**Historical validation**: Need to check star-isolated teams' tourney records

#### Branch 2: UNICORN_PLAYER (outcome distribution widener)
**Trigger**: Underdog has a specialist player who:
  - Leads their conference in a specific stat (3PT%, rebounds, etc.)
  - OR has a stat that's top-10 nationally in one category
  - OR has a style that historically creates March magic
    (extreme 3PT specialist, physical rebounder, lockdown defender)
**Magnitude**: NOT a probability shift. Instead WIDENS the distribution.
  Instead of P(upset) = 0.15, model it as P(upset) = 0.15 ± 0.08.
  The unicorn doesn't make the upset likely, but makes it POSSIBLE.
**Evidence**: Johnston (High Point, 45.2% 3PT, 415 career threes),
  Gohlke (Oakland 2024), DJ Burns (NC State 2024)
**Why it works**: Variance. When a 3PT specialist gets hot, they can
  single-handedly outscore a better team. This is a known distributional
  property of 3PT-dependent offenses.
**Key distinction**: This branch says "widen the error bars" not "flip pick."

#### Branch 3: LOCKDOWN_DEFENDER (opponent's star neutralizer)
**Trigger**: Team A has a player who:
  - Guards multiple positions (switchable)
  - Has top-tier individual defensive metrics
  - Will be matched up against opponent's primary scorer
**Magnitude**: -0.03 to -0.06 (reduces opponent's effective star power)
**Evidence**: Illinois's Boswell locked down Penn's best player, guarded
  him all game. If VCU's Terrence Hill faces Boswell in R32, that matchup
  matters enormously.
**Data source**: Player-level defensive metrics (box_dbpr from EvanMiya)

#### Branch 4: ALPHA_VACUUM (chemistry collapse post-star-loss)
**Trigger**: Team lost its best player (by BPR) to injury AND:
  - Assist rate dropped >15% post-injury, OR
  - Team went ≤.500 in games after the injury
**Magnitude**: -0.05 to -0.15 (probability shift toward opponent)
**Evidence**: UNC lost Wilson → went 5-3 → assists likely dropped →
  Trimble played selfish → 19-pt lead evaporated → lost in OT
**Why it's different from injury model**: The injury model captures the
  SKILL LOSS. This branch captures the CHEMISTRY DESTRUCTION — the
  remaining players' inability to coexist without the alpha.

### TIER 2: TEAM-LEVEL BRANCHES

#### Branch 5: FATIGUE_TRAP
**Trigger**: Team A has ≤7 rotation players AND opponent is deeper
  OR Team A's top player logged 38+ minutes in previous round
**Magnitude**: -0.03 to -0.08 (shift toward deeper team)
**Evidence**: Siena had ZERO subs → went 7 min scoreless vs Duke.
  Duke itself is running 7-man rotation (Foster + Ngongba out).
  BYU's Dybantsa played all 40 minutes and faded.
**Scaling**: Gets WORSE in later rounds (cumulative fatigue).
  R64 = base, R32 = 1.2x, S16 = 1.4x

#### Branch 6: EARLY_LEAD_MIRAGE
**Trigger**: Historical pattern — underdog/young team that tends to
  start games fast (high offensive_burst) but fades after halftime
  (negative q3_adj_strength).
**Magnitude**: -0.02 to -0.05 (shift toward comeback by better team)
**Evidence**: UNC led by 19 → lost. McNeese led by 15 → lost.
  Siena led by 11 → lost. ALL three were lower seeds or less
  experienced teams who couldn't sustain early energy.
**Data**: Compare offensive_burst (1H advantage) vs q3_adj_strength
  (2H adjustment). If team has high burst but negative adjustment,
  they start fast but can't close.

#### Branch 7: REBOUNDING_MISMATCH
**Trigger**: ORB% difference > 0.06 (team A gets 6%+ more offensive boards)
  OR total RBM difference > top-quartile of historical matchups
**Magnitude**: +0.02 to +0.05 (shift toward rebounding-dominant team)
**Evidence**: TCU's Punch (13 rebounds), High Point's hustle, Texas
  physical advantage. Every upset on Day 1 had a rebounding story.
**Why branch not root**: Because rebounding matters MORE in close games
  and LESS in blowouts. As a root weight, it hurts chalk predictions.
  As a branch, it only fires when the matchup is competitive.

#### Branch 8: CONFERENCE_PHYSICALITY
**Trigger**: Big 12 team vs non-Big 12, or high-physical-conference
  team vs finesse conference.
**Magnitude**: +0.01 to +0.03 (mild shift toward physical team)
**Evidence**: Big 12 went strong Day 1 (TCU, Texas, Louisville).
  Big 10 lost multiple (Wisconsin, Ohio State).
**Caution**: This is the WEAKEST branch. Conference effects vary year
  to year. Should have lowest confidence multiplier.

### TIER 3: CONTEXTUAL BRANCHES

#### Branch 9: YOUTH_UNDER_PRESSURE
**Trigger**: Team's EXP < 1.5 AND this is their first tournament game
  (or first tournament game for 3+ starters)
**Magnitude**: -0.03 to -0.06 (shift toward experienced opponent)
**Evidence**: UNC young team collapsed. Duke freshmen trailed 16-seed
  at halftime. Nebraska VETERANS played relaxed team ball.
  VCU's experienced squad stayed composed through 19-point deficit.
**Round scaling**: STRONGEST in R64 (first game nerves), decreasing
  each round as nerves fade.

#### Branch 10: FORM_COLLAPSE
**Trigger**: Team lost 3+ of last 6 games heading into tournament
  OR conference tournament exit was embarrassingly early
**Magnitude**: -0.02 to -0.05
**Evidence**: UNC lost 3 of last 8 including ACC tourney QF loss to Clemson.
  BYU went 2-4 after Saunders injury.

#### Branch 11: COACH_TOURNAMENT_DNA
**Trigger**: Coach has 5+ tournament wins AND Elite 8+ appearances
  OR coach is in first-ever tournament (negative signal)
**Magnitude**: +/- 0.02 to 0.04
**Evidence**: VCU's Martelli (first year, but system-coached team).
  Existing ctf weight covers this partially but as a BRANCH it's more
  nuanced — fire only in tight games where coaching adjustments
  (halftime, timeout plays) are decisive.

#### Branch 12: REVENGE_MOMENTUM
**Trigger**: Team beat a higher-seed in previous round AND the win
  was comeback/dramatic (not a blowout)
**Magnitude**: +0.01 to +0.03 (psychological boost)
**Evidence**: VCU's 19-point comeback → massive confidence boost.
  Nebraska's first-ever tournament win → relaxation or peak.
**Caution**: Can go both ways. "Championship hangover" after emotional
  win is real. Needs context.

#### Branch 13: THREE_PT_VARIANCE_BOMB
**Trigger**: Underdog relies on 3PT shooting for >40% of their offense
  AND their 3PT% > 37%
**Magnitude**: Distribution widener (like UNICORN, not a shift)
  P(upset) range increases by ±0.05
**Evidence**: High Point's entire identity. Johnston at 45.2% from 3.
  When they're hot, they can beat anyone. When they're cold, they
  lose to anyone.
**Why branch not root**: Because 3PT variance ONLY matters in the
  specific matchup context. Against elite perimeter defense, the
  hot-shooting scenario is less likely.

#### Branch 14: DEPTH_ADVANTAGE_IN_LATER_ROUNDS
**Trigger**: R32 or later, AND team has BDS (bench depth) > 0.60
  AND opponent has BDS < 0.40
**Magnitude**: +0.02 to +0.05 (compounds each round)
**Evidence**: Teams that go 10+ deep can weather foul trouble, fatigue,
  and matchup adjustments. Becomes critical in S16+ when games are
  two days apart.

---

## Implementation: The Branch Engine

```python
class BranchEngine:
    """Evaluate all branches for a matchup and produce probability shifts."""

    def evaluate(self, matchup, player_df, injury_profiles, round_name):
        """
        Returns:
          total_shift: float — net probability adjustment
          variance_widen: float — how much to widen outcome distribution
          triggered_branches: list — which branches fired and why
        """
        shifts = []
        variance_mods = []
        triggered = []

        for branch in self.branches:
            result = branch.check(matchup, player_df, injury_profiles, round_name)
            if result.triggered:
                shifts.append(result.shift * result.confidence)
                variance_mods.append(result.variance_mod)
                triggered.append(result)

        # Cap total shift at ±0.15
        total_shift = max(-0.15, min(0.15, sum(shifts)))
        total_variance = sum(variance_mods)

        return total_shift, total_variance, triggered
```

## How Final Prediction Works

```
P_final = P_root                          # from Model A (CORE_WEIGHTS, 73.7%)
        + injury_shift                     # from injury model
        + narrative_shift                  # from narrative layer
        + branch_engine_shift             # from tree branches (NEW)
        ± branch_variance_widen           # widens upset probability range
```

The root is NEVER touched. Model A stays at 73.7%.
Branches only fire when specific conditions are met.
Each branch has empirical evidence from actual games.
Total shift is capped to prevent branch stacking from going crazy.

## Critical Self-Critique

### What's strong:
1. Root is backtested and proven. 73.7% over 1,070 games.
2. Branches are evidence-based. Every one maps to a real Day 1 outcome.
3. Variance widening (UNICORN, 3PT_BOMB) is philosophically correct —
   the model says "I'm less sure" rather than "I'm wrong."

### What's weak:
1. **No historical backtest for branches.** We have Day 1 evidence but
   haven't verified these patterns hold over 2008-2025. STAR_ISOLATION
   and ALPHA_VACUUM especially need validation.
2. **Branch stacking risk.** If a team triggers 4 branches simultaneously,
   the combined shift could still distort predictions even with the cap.
   Need interaction terms or diminishing returns.
3. **Confidence calibration.** Each branch's confidence multiplier is
   currently hand-tuned. Ideally we'd backtest each branch independently.
4. **Data availability.** Player-level branches (1-4) require EvanMiya
   data which we only have for 2026. Historical branches can only use
   team-level data.
5. **Unicorn player identification** is inherently subjective. Johnston
   was 0-for-4 on 2-pointers all season — no model would flag him as
   a game-winner. The best we can do is flag HIGH-VARIANCE players.

### What needs empirical testing:
- Does removing top25_perf ONLY in close games (35-65%) improve accuracy?
- Do star-isolated teams historically underperform their seed?
- Is the "early lead mirage" pattern statistically significant?
- Does Big 12 physicality actually predict March success across years?

## Next Steps

1. Keep Model A as the ROOT. Restore ACTIVE_WEIGHTS = CORE_WEIGHTS.
2. Build the BranchEngine as a POST-MODEL modifier (like injury model).
3. Backtest individual branches where historical data allows.
4. For 2026: activate all branches, compare predictions to actual outcomes.
5. Use R32 predictions as the first live test of the tree model.
