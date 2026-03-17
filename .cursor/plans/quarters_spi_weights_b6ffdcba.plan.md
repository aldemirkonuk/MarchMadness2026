---
name: Quarters SPI Weights
overview: Add proxy Q1/Q3 quarter scoring params, fix SPI proxy, and build a 3-stage weight optimization pipeline (grid search, sensitivity analysis, Bayesian optimization) with a comprehensive output report.
todos:
  - id: add-quarter-proxies
    content: Add q1_start_strength and q3_adj_strength proxy params to models, data_loader, weights
    status: completed
  - id: fix-spi
    content: Fix SPI to use roster_rank for ALL teams, not just top-20
    status: completed
  - id: build-grid-search
    content: "Build Stage 1: random weight search against historical tournaments (10k samples)"
    status: completed
  - id: build-sensitivity
    content: "Build Stage 2: sensitivity analysis from best grid search result"
    status: completed
  - id: build-bayesian
    content: "Build Stage 3: Bayesian optimization with scipy/optuna"
    status: completed
  - id: build-output-report
    content: Build comprehensive output report comparing current vs optimized weights
    status: completed
  - id: test-and-run
    content: Run optimizer, verify results, print report
    status: completed
isProject: false
---

# Quarter Scoring, SPI Fix, and Weight Optimization

## 1. Add Q1/Q3 Quarter Scoring Proxies (keep existing quadrant records)

We have no play-by-play data, so we derive proxies from existing metrics:

- `**q1_start_strength**` = how strong the team comes out of the gate. Proxy: `0.5 * AdjO_normalized + 0.3 * pace_normalized + 0.2 * three_pri_raw`. Fast-starting, high-offense teams score more in Q1. Captures "starts strong." If its way below avg. *huge indicator likely lose (if 80-20 -> 45-55 win/lose chance, verify this)
- `**q3_adj_strength`** = halftime adjustment ability. Proxy: `0.4 * coaching_factor + 0.3 * experience + 0.2 * consistency + 0.1 * clutch_factor`. Experienced coaches + veteran rosters adjust better after halftime. Captures "responds to adjustments."

Both get ~1% weight each. These live alongside the real `q1_record` and `q34_loss_rate` quadrant params (which stay).

Files: [src/models.py](src/models.py) (add attrs), [src/data_loader.py](src/data_loader.py) (compute), [src/weights.py](src/weights.py) (add weights)

## 2. Fix SPI (Star Power Index)

Currently SPI = `AdjEM / 20` for most teams (only top-20 roster_rank teams get the better formula). This makes it mostly redundant with AdjEM.

Fix: For ALL teams, use `SPI = (68 - roster_rank) / 68 * AdjEM / 15.0`, blended with public championship pick sentiment. This uses EvanMiya's roster_rank (which reflects actual player talent) for every team, not just top-20.

File: [src/data_loader.py](src/data_loader.py) lines ~396-412

## 3. Weight Optimization Pipeline (new file: `src/weight_optimizer.py`)

Three stages run sequentially, each building on the last:

### Stage 1: Grid Search / Random Search

- Generate 10,000 random weight vectors (each sums to 1.0)
- For each, run predictions on historical tournaments (2010-2025)
- Score = % of games correctly predicted (higher seed wins when model says >50%)
- Keep top 50 weight sets

### Stage 2: Sensitivity Analysis

- From the best weight set found in Stage 1, perturb each of the 38-40 params one at a time (+/-50%)
- Measure how much prediction accuracy changes per param
- Output: ranked list of "most impactful" to "least impactful" weights
- Identify params where weight barely matters (candidates for removal)

### Stage 3: Bayesian Optimization

- Use `scipy.optimize.minimize` or `optuna` (if available) 
- Start from the best grid search result
- Objective: maximize historical prediction accuracy
- Run 500-1000 trials with smart sampling
- Constraint: weights sum to 1.0

### Output Report

A comprehensive printout showing:

- Best weights found vs current weights (side by side)
- Per-param sensitivity ranking
- Historical accuracy: current weights vs optimized weights
- Recommended weight changes with magnitude and direction
- Confidence intervals on the optimization

The optimizer does NOT auto-apply weights -- it recommends, and the user decides.

Files: new [src/weight_optimizer.py](src/weight_optimizer.py), called from [src/main.py](src/main.py) via a flag or separate entry point