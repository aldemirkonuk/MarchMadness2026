---
name: Model Improvement Experiments
overview: Run 8 sequential improvement experiments, measuring each independently. Fix historical proxies first, then grid search, calibration tuning, XGBoost feature expansion, volatility integration, and player matchup sandbox design.
todos:
  - id: exp1-proxies
    content: "Experiment 1: Fix 5 historical proxy params in weight_optimizer _row_to_param_dict()"
    status: completed
  - id: exp2-grid
    content: "Experiment 2: Run grid search + Bayesian optimization with fixed proxies"
    status: completed
  - id: exp3a-k
    content: "Experiment 3a: Sweep logistic K parameter (3.0-10.0) for Brier score"
    status: completed
  - id: exp3b-lambda
    content: "Experiment 3b: Sweep ensemble lambda (0.3-0.8)"
    status: completed
  - id: exp3c-platt
    content: "Experiment 3c: Test Platt scaling / isotonic calibration"
    status: completed
  - id: exp4-xgb
    content: "Experiment 4: Expand XGBoost ML_FEATURE_KEYS to 25-30 params"
    status: completed
  - id: exp5-volatile
    content: "Experiment 5: Volatile-favorite flagging + WTH volatility integration"
    status: completed
  - id: exp6-merge
    content: "Experiment 6: Test merging adj_em/barthag/z_rating correlation cluster"
    status: completed
  - id: exp7-upset
    content: "Experiment 7: Build focused upset detection model for high-upset matchups"
    status: completed
  - id: exp8-player
    content: "Experiment 8: Player matchup sandbox (8 params + style_clash + 4 flags, inline output)"
    status: completed
  - id: exp9-metrics
    content: "Experiment 9: Add AUC-ROC, Average Precision, LOYO cross-validation to evaluation pipeline"
    status: in_progress
isProject: false
---

# Model Improvement: Sequential Experiments

Run each experiment independently, measure accuracy before/after, and keep only what improves the model. Current baseline: **73.7% on 1,070 historical games (CORE_WEIGHTS)**.

**Key decisions from brainstorm**:

- CTF and legacy_factor: fix historical proxies but keep weights low (current ~0.8% and ~0.2%)
- Player matchups: sandbox first, 8 params + style_clash, flag all 4 types, inline in main.py output
- Evaluation: add AUC-ROC, Average Precision, and LOYO cross-validation AFTER experiments complete
- Each experiment measured independently, keep only what improves

## Experiment 1: Fix Historical Proxies in Optimizer

**Problem**: 6 params in `_row_to_param_dict()` in [src/weight_optimizer.py](src/weight_optimizer.py) are hardcoded constants for all historical teams, making the optimizer blind to their real signal.

**Fixes** (we HAVE the data for all of these):

- `**ctf`** (currently `= 0.5`): Wire in Coach Results.csv -- match coach name by team+year, compute `(tourney_wins+2)/(tourney_games+4)`
- `**legacy_factor`** (currently `= 0.0`): Wire in Team Results.csv PASE data -- look up `PASE` column by team+year, winsorize to [-3, +5]
- `**consistency`** (currently `= 0.5`): Approximate from KenPom data -- use `1 / (1 + |AdjEM| * 0.02)` as a proxy (dominant teams are more consistent)
- `**scoring_margin_std`** (currently missing): Add proxy: `base_std * (1 + |AdjEM|/30)^(-0.5)` where `base_std = 14.0` (NCAA average). Teams with high AdjEM have lower variance.
- `**msrp`** (currently `= 0.0`): Approximate as `AdjEM / 20.0` (correlated with scoring run differential)
- `**injury_health**` (currently `= 35.0`): Leave as-is (no historical injury data exists)

**Files**: [src/weight_optimizer.py](src/weight_optimizer.py) lines 100-187 (`_row_to_param_dict`)

**Measure**: Run `_evaluate_weights(CORE_WEIGHTS, games, PARAM_KEYS)` before and after proxy fixes.

## Experiment 2: Grid Search with Fixed Proxies + scoring_margin_std

**What**: Run `stage1_grid_search()` then `stage3_bayesian()` on the fixed historical data. Now the optimizer can properly value all 41 params including `scoring_margin_std`.

**Output**: New optimized weight dict. Compare accuracy to current 73.7%.

**Files**: Run optimizer from [src/weight_optimizer.py](src/weight_optimizer.py), update [src/weights.py](src/weights.py) if improved.

## Experiment 3: Calibration Tuning (3 sub-experiments)

Run sequentially, each measured independently on Brier score:

- **3a: Logistic K tuning** -- Sweep K from 3.0 to 10.0 in steps of 0.5. Currently K=6.0. Lower K = softer predictions (closer to 50%). Higher K = sharper (more extreme). Find K that minimizes Brier score.
  - File: K is used in `_predict_winner()` in [src/weight_optimizer.py](src/weight_optimizer.py) line 236 and `LOGISTIC_K` in [src/weights.py](src/weights.py) line 409.
- **3b: Ensemble lambda tuning** -- Sweep `ENSEMBLE_LAMBDA` from 0.3 to 0.8. Currently 0.5 (equal blend of Phase 1A and 1B). Find blend that minimizes prediction error on historical data.
  - File: [src/weights.py](src/weights.py) line 411.
- **3c: Platt scaling** -- After predicting, apply isotonic regression or Platt scaling to map raw probabilities to calibrated ones. This is a post-processing step that doesn't change weights.
  - File: Add to [src/composite.py](src/composite.py) as an optional calibration function.

## Experiment 4: Expand XGBoost Feature Set

**Problem**: XGBoost only uses 18 features (`ML_FEATURE_KEYS`) and is missing `net_score`, `ppg_margin`, `z_rating`, `momentum`, `clutch_factor`, `march_readiness`, and other high-weight params. This causes SHAP to overweight `barthag` (it absorbs signal from missing correlated features).

**Fix**: Expand `ML_FEATURE_KEYS` to include the top 25-30 params from CORE_WEIGHTS. Add corresponding mappings in `_row_to_stats()`.

**Measure**: Compare Phase 1B accuracy and the Phase 1A/1B divergence gap before and after.

**File**: [src/xgboost_model.py](src/xgboost_model.py) lines 17-21 and 186-214.

## Experiment 5: Volatile-Favorite Flagging + WTH Integration

**What**: In `compute_win_probability()`, when a high-seed team (1-4 seed) has `scoring_margin_std` above the field median:

- Apply a volatility penalty that shrinks their win probability slightly toward 50%
- Feed `scoring_margin_std` into the WTH chaos layer as an additional signal alongside `chaos_index`

**File**: [src/composite.py](src/composite.py) `compute_win_probability()`, WTH block.

**Measure**: Re-run Monte Carlo, compare R64 upset prediction rate to historical.

## Experiment 6: Correlation Cluster Experiment

**What**: Try merging the `adj_em / barthag / z_rating` cluster into a single composite (like we did with efg+ts -> shooting_eff). Test whether reducing 3 params to 1 composite improves or hurts accuracy.

**If it hurts**: Revert -- the optimizer already manages the redundancy through low weights.

**File**: [src/data_loader.py](src/data_loader.py), [src/weights.py](src/weights.py).

## Experiment 7: Upset Detection Model

**What**: Train a lightweight logistic regression specifically on 5-vs-12, 6-vs-11, 7-vs-10, and 8-vs-9 matchups (the high-upset-rate zone). Use a focused feature set: `scoring_margin_std`, `momentum`, `consistency`, `sos`, `exp`. Blend its output with the main model's probability for these specific matchup types.

**File**: New function in [src/composite.py](src/composite.py) or [src/xgboost_model.py](src/xgboost_model.py).

## Experiment 8: Player Matchup Sandbox

**Sandbox**: Build and show results on current matchups BEFORE wiring into weights. Data source: `data/EvanMiya_Players.csv` (3,151 players, BPR/OBPR/DBPR/possessions).

**8 matchup params** (computed per matchup, not per team):

- `star_bpr_mismatch`: Team A's best BPR - Team B's best BPR
- `defensive_pressure`: Team B's best DBPR / Team A's best OBPR (0-1, higher = more resistance)
- `depth_advantage`: avg BPR of Team A's top 5 - avg BPR of Team B's top 5
- `two_way_threat`: binary, does team have player with OBPR > 5 AND DBPR > 3?
- `bpr_concentration`: best_BPR / sum(top_5_BPR) -- star dependency ratio
- `star_carry_ratio`: best_player_BPR / team_AdjEM -- how much team depends on one player
- `star_vs_opp_defense`: best_OBPR - (opponent_AdjD / 100) -- can star score against this defense?

**Style clash param** (computed both directions, net = A's disadvantage - B's disadvantage):

- `style_clash`: max(Team_A_3PT_rate * Team_B_opp_3PT_defense, Team_A_paint_rate * Team_B_rim_protection)

**4 flag types** (ALL flagged, inline in main.py output):

1. Star BPR gap < 3.0 AND seed gap >= 5 (favorite's star barely outclasses underdog)
2. Favorite has NO two-way player but underdog DOES
3. Favorite's bpr_concentration > 40% (one-man team)
4. Underdog's best defender DBPR within 60% of favorite's best scorer OBPR

**Output format**: Inline after each R64 matchup in main.py:
`*** FLAGGED: star mismatch only +2.1, underdog has two-way threat, style clash: 3PT offense vs elite 3PT D *`**

## Experiment 9: Evaluation Metrics Upgrade

**Add after all other experiments are complete**:

- AUC-ROC for binary win/loss predictions
- Average Precision for upset detection (upsets are ~28% of games)
- Leave-One-Year-Out (LOYO) cross-validation: train on all years except Y, test on Y, repeat for each year 2008-2025. This gives honest accuracy (current 73.7% is likely ~1-2% inflated).

**Files**: [src/weight_optimizer.py](src/weight_optimizer.py) `_evaluate_weights()`, add `_evaluate_weights_loyo()`.