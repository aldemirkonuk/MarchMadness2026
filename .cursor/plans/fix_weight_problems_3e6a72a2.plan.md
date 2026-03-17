---
name: Fix Weight Problems
overview: "Fix 10 problematic parameters that are decaying predictions: replace broken Q1/Q3, fix opp_to_pct inversion, restructure the shooting cluster, add height, redesign three_pri as a scoring style param, make to_pct quality-adjusted, strengthen legacy_factor, and clean up dead weight."
todos:
  - id: add-height
    content: "Add eff_height param: load EFF HGT from KenPom Barttorvik, add to Team, wire to weights at 1.5%"
    status: completed
  - id: fix-opp-to
    content: "SOS-adjust opp_to_pct in data_loader: opp_to_pct * (sos / league_avg_sos)"
    status: completed
  - id: replace-three-pri
    content: Replace three_pri with scoring_balance (2PT%*2PTR + 3P%*3PTR) from KenPom Barttorvik
    status: completed
  - id: fix-to-pct
    content: "Quality-adjust to_pct: divide by SOS ratio to reward low TO against hard schedules"
    status: completed
  - id: merge-shooting
    content: Merge efg_pct + ts_pct into single shooting_eff (0.6*eFG + 0.4*TS), remove ts_pct weight
    status: completed
  - id: fix-drb
    content: Add z-score normalization for drb_pct to amplify narrow-range signal
    status: completed
  - id: fix-legacy
    content: Winsorize legacy_factor to [-3, +5], bump weight to 2%
    status: completed
  - id: fix-q1q3
    content: Replace fake q1/q3_score with real quadrant records from Teamsheet Ranks.csv
    status: completed
  - id: remove-neutral
    content: Remove neutral_perf from weights
    status: completed
  - id: rebalance-weights
    content: Rebalance all weights to sum to 1.0, re-run pipeline and tests
    status: completed
isProject: false
---

# Fix Weight Problems and Add Height Parameter

## Key Findings That Drive These Changes

- **Q1/Q3 scores are FAKE**: They're computed as `(AdjO - AdjD) / 20` and `(AdjO - AdjD) / 25` -- literally just AdjEM divided by a constant. Correlation with AdjEM is r=1.000. They contribute ZERO new information. But we have **real quadrant records** in `Teamsheet Ranks.csv` (Q1 W/L, Q2 W/L, Q3 W/L, Q4 W/L, Q1A W/L) that are currently loaded but not used for this.
- **eFG% and TS% are r=0.966 correlated** -- nearly identical. BARTHAG is only r=0.46 with eFG%, so it's genuinely different.
- **Height data exists**: `KenPom Barttorvik.csv` has `AVG HGT` and `EFF HGT` (minutes-weighted effective height). EFF HGT correlates r=0.557 with AdjEM and r=-0.568 with seed. Strong signal.
- **We have 2PT% and 2PTR data** in KenPom Barttorvik for building a paint-vs-perimeter scoring style param.

---

## Changes (10 items)

### 1. Add `eff_height` parameter (NEW, ~1.5% weight)

- Source: `EFF HGT` from `KenPom Barttorvik.csv` (minutes-weighted height, already loaded in data pipeline)
- Add `eff_height` attribute to `Team` in [src/models.py](src/models.py)
- Load from `KenPom Barttorvik.csv` in [src/data_loader.py](src/data_loader.py) `_build_kenpom_barttorvik()`
- r=0.557 with AdjEM, strong enough to justify its own weight. Different signal from BDS (which uses height distribution, not absolute height).

### 2. Fix `opp_to_pct` -- SOS-adjust it (2.5% weight, keep weight)

- **Problem**: 1-seeds face better opponents who turn it over less. Raw metric is confounded by SOS.
- **Fix**: Normalize by opponent quality. Replace raw `opp_to_pct` with `opp_to_pct_adj = opp_to_pct * (team_sos / league_avg_sos)`. This way, a team that forces turnovers against elite opponents gets credit, while a team that forces turnovers against cupcakes gets discounted.
- Change in [src/data_loader.py](src/data_loader.py), in the derived-scores block.

### 3. Replace `three_pri` with `scoring_style` -- paint vs perimeter balance (keep ~3% weight)

- **Problem**: `three_pri` (3P% * 3PA/FGA) has zero separation between 1-seeds and 16-seeds because it only measures 3-point volume+accuracy, and bad teams can still chuck threes.
- **New param**: `scoring_balance = 2PT% * 2PTR + 3P% * 3PTR` -- a composite that rewards teams efficient from BOTH inside and outside. Teams that can score from everywhere are harder to defend in March.
- Source: `2PT%`, `2PTR`, `3PT%`, `3PTR` all exist in `KenPom Barttorvik.csv`.
- Keep `three_pri` as a sub-component but drop its individual weight. The new `scoring_balance` replaces it.

### 4. Make `to_pct` quality-adjusted (keep 4% weight)

- **Problem**: Raw TO% barely separates teams. A team with 15% TO rate against top-50 teams is much better than one with 15% against cupcakes.
- **Fix**: `to_pct_adj = to_pct / max(sos / league_avg_sos, 0.5)`. Teams with low turnovers against hard schedules get rewarded more. This amplifies the spread without changing the metric's meaning.
- Change in [src/data_loader.py](src/data_loader.py).

### 5. Merge `efg_pct` + `ts_pct` into one `shooting_eff` (combined ~7% weight, save ~2.5%)

- **Problem**: r=0.966 correlation. Two params carrying 9% combined weight for the same information.
- **Fix**: Keep ONE composite: `shooting_eff = 0.6 * efg_pct + 0.4 * ts_pct`. This preserves both signals in one param at ~7% weight. The freed 2.5% gets redistributed to new params (height, scoring_balance).
- BARTHAG stays separate (r=0.46 with eFG%, genuinely different).

### 6. Fix `drb_pct` with rebound margin edge-case amplifier (keep 2.5% weight)

- **Problem**: Raw range 0.650-0.774, too narrow.
- **Fix**: Apply a z-score transformation instead of min-max normalization for this specific param. `drb_z = (drb_pct - mean_drb) / std_drb`. This mathematically amplifies the small but real differences. In matchup prediction, the rebounding margin between two teams gets extra significance when one team has a z-score > 1.5 (outlier rebounder).
- Implementation: Add `drb_pct` to a new `Z_SCORE_PARAMS` set in [src/weights.py](src/weights.py). Modify normalization in [src/composite.py](src/composite.py) to use z-score for listed params.

### 7. Strengthen `legacy_factor` -- cap outliers, bump weight to 2%

- **Problem**: Range -7.6 to +13.5 with CV=8.0. Wild outliers like UConn (+13.5) and Virginia (-7.6) dominate.
- **Fix**: Winsorize to [-3, +5] range before normalization. This preserves the "winner instinct" signal (UConn, Michigan State, Gonzaga, Arkansas) while preventing one team from anchoring the scale. Keep weight at 1%.
- The teams that benefit most: UConn (+13.5 -> +5), Michigan St (+10.3 -> +5), Michigan (+8.4 -> +5), Florida (+7.7 -> +5), North Carolina (+7.6 -> +5). These are exactly the teams the user identified as having "winner instinct."

### 8. Replace fake `q1_score`/`q3_score` with real quadrant records (combined 2% weight)

- **Problem**: Currently `q1_score = (AdjO - AdjD) / 20` and `q3_score = (AdjO - AdjD) / 25`. Correlation with AdjEM = **r=1.000**. They are literally AdjEM scaled by a constant. Zero new information.
- **Fix**: Replace with REAL quadrant records from `Teamsheet Ranks.csv`:
  - `q1_record = Q1 Wins / (Q1 Wins + Q1 Losses)` -- win% against quadrant 1 opponents
  - `q34_loss_rate = (Q3 Losses + Q4 Losses) / (Q3 Games + Q4 Games)` -- bad loss rate (inverted)
- These are genuinely independent from AdjEM and capture "can you beat good teams?" and "do you lose to teams you shouldn't?"

### 9. Remove `neutral_perf` (free up 1%)

- Per user request. Remove from weights and PARAM_KEYS. The neutral court signal is already partially captured by `march_readiness`.

### 10. Keep `msrp` (0.5% weight)

- MSRP correlates r=0.656 with AdjEM -- moderately correlated but not redundant. It provides a forward-looking trajectory signal that AdjEM alone doesn't. High Point (12-seed) and Saint Louis (9-seed) score high on MSRP, identifying mid-majors with upside. At 0.5% weight the risk is minimal. Keep it.

---

## Weight Redistribution Summary

Freed weight:

- `efg_pct` + `ts_pct` merge: saves ~2.5%
- `neutral_perf` removal: saves 1%
- `three_pri` replaced by `scoring_balance`: net 0% (same slot)
- Total freed: ~3.5%

New allocations:

- `eff_height`: +1.5%
- `legacy_factor` bump: +1% (from 1% to 2%)
- `scoring_balance`: absorbs `three_pri` slot at ~3%
- Remaining ~1% redistributed to strengthen `q1_record` and `q34_loss_rate`

Final param count: 40 -> 39 (merge shooting_eff, remove neutral_perf, add height, replace q1/q3 with quadrant records, replace three_pri with scoring_balance)