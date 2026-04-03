# Final Four Analysis

## What Distinguishes This Model

### 1. Evidence-first scenario reasoning

The repo does not stop at a single blended probability. It carries a four-category scenario engine that explains *why* a matchup moves:

- `ROSTER_STATE`
- `MATCHUP_STYLE`
- `FORM_TRAJECTORY`
- `INTANGIBLES`

That makes late-tournament predictions auditable instead of opaque.

### 2. Surgical self-correction instead of wholesale reweighting

The strongest pattern in this codebase is not just that it predicts, but that it diagnoses its own failure modes. The Texas inflation work already proved this: rather than replacing the base model, the system added asymmetric-data guards, decay, dampening, and coherence limits to protect the root model.

### 3. Live tournament adaptation

This model has a better live-tournament spine than most bracket models:

- injury impact is quantified, not binary
- tournament momentum is separated from season momentum
- box-score trajectories are round-aware
- late-round narrative signals can be layered after the ensemble

That combination is unusual and worth preserving.

## What We Missed

### 1. Bad live rows were still allowed to influence the model

The biggest structural gap was not a basketball idea but a data-integrity issue. `tournament_box_scores.csv` contained at least one hard mismatch:

- `R32 UCLA 73, Connecticut 57, winner=Connecticut`

That kind of row can silently poison tournament momentum and profile logic if it is trusted as authoritative.

### 2. Partial box scores were acting like weak or missing truth

Many tournament rows had only partial stat coverage. Before this pass, the loader mostly treated games as usable or unusable based on a narrow subset of fields, which meant the live layer could become biased toward the few games with richer box-score fill.

### 3. Confidence was still too blunt in noisy games

The model had good raw directional logic, but it still needed a cleaner way to say:

- this edge is real, but thinly supported
- this game is close enough that volatility matters
- this late-tournament signal should compress confidence, not just flip the pick

## What We Implemented In This Pass

### 1. Live data validation and in-memory normalization

New module: `src/live_data_validation.py`

It now:

- canonicalizes team names across live files
- compares the wide box-score file to the slim round-result files
- flags winner/score contradictions
- normalizes swapped A/B orientations in memory before downstream use

### 2. Coverage-aware tournament profiles

`src/tournament_loader.py` now tracks:

- per-stat coverage
- `data_confidence`
- `comeback_confidence`

This means incomplete games can still contribute signal, but downstream code can now distinguish strong live evidence from thin live evidence.

### 3. Explicit uncertainty in the scenario engine

`src/scenario_engine.py` now exposes:

- `uncertainty`
- `confidence_post`

Uncertainty increases when:

- the base game is close
- live tournament coverage is weak
- halftime comeback evidence is thin
- coherence is low
- 3PT volatility is high

Instead of only shifting probabilities, the engine now compresses overconfident edges toward `50/50` when support is thin.

### 4. Bracket-state-aware Final Four outputs

`src/main.py` now writes:

- `data/results/current_stage_odds.csv`
- `data/live_results/f4_predictions.csv`

These artifacts focus on the teams still alive instead of forcing the user to interpret stale full-field championship tables.

### 5. Audit upgrades

`run_accuracy_audit.py` now:

- prints live-data validation issues up front
- reports reliability bins, not just hit rate

Current reliability snapshot:

- `50-60%`: realized `50.0%`, gap `-5.1%`
- `60-70%`: realized `60.0%`, gap `-6.2%`
- `70-80%`: realized `76.5%`, gap `+1.4%`
- `80-90%`: realized `84.6%`, gap `+1.6%`
- `90-100%`: realized `89.5%`, gap `-6.4%`

That last bin confirms the main remaining calibration weakness: the stack is still a little too aggressive at the top end.

## Final Four Predictions

From `data/live_results/f4_predictions.csv`:

- `Illinois` over `Connecticut`: `54.96%`
- `Michigan` over `Arizona`: `57.97%`
- `Michigan` over `Illinois` in the title game: `77.05%`

## Current-Stage Title Odds

From `data/results/current_stage_odds.csv`:

- `Michigan`: `53.48%`
- `Arizona`: `34.83%`
- `Illinois`: `8.11%`
- `Connecticut`: `3.59%`

## Best Next Improvements

### 1. Validate the new signals against held-out historical late rounds

The added comeback and uncertainty logic is directionally sound, but it still needs a more formal historical check for overfitting in small samples.

### 2. Add feature-level confidence directly into scenario signal weights

Right now confidence compresses the final edge after the scenario calculation. A stronger next step would be letting low-coverage tournament stats reduce signal weight *inside* the affected category.

### 3. Split pre-tournament odds from live conditional odds everywhere

The repo should keep both views, but label them clearly:

- pre-tournament whole-field title odds
- live conditional title odds for teams still alive

That will make downstream interpretation much cleaner.
