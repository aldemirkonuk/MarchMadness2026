---
name: Full Bracket Progression Output
overview: Extend the dashboard's `generate_full_report()` to produce a deterministic "predicted bracket" with detailed matchup-by-matchup predictions for all rounds (R32 through Championship), using the same format as the existing R64 section.
todos:
  - id: extend-dashboard
    content: Add _generate_bracket_progression() to dashboard.py and extend generate_full_report() with prob_func parameter
    status: completed
  - id: wire-probfunc
    content: Pass ensemble_prob_func through save_dashboard() and generate_full_report() in main.py
    status: completed
  - id: test-run
    content: Run simulation, verify full_report.txt contains R32 through Championship matchup details
    status: completed
isProject: false
---

# Full Bracket Progression Output

## What Changes

Add a deterministic bracket prediction to [src/dashboard.py](src/dashboard.py) that walks the bracket from R64 winners through the Championship game, computing ensemble win probabilities at each round using the existing `_get_win_prob()` function (which already handles round-specific chaos, weight amplifiers, and fatigue).

## How It Works

1. Take the 32 R64 predicted winners (from the existing `matchups` list -- the favored team in each)
2. Pair them for R32 using the bracket structure in [src/monte_carlo.py](src/monte_carlo.py) (`BRACKET_SEEDS` defines the pairing order)
3. For each R32 matchup, call `_get_win_prob(team_a, team_b, prob_func, "R32")` to get the probability
4. Pick the favorite, record it, advance to S16
5. Repeat through E8, F4, Championship

The bracket pairing logic per region:

- R64 produces 8 winners (indices 0-7)
- R32: pairs `(0,1), (2,3), (4,5), (6,7)` -> 4 winners
- S16: pairs `(0,1), (2,3)` -> 2 winners
- E8: pair `(0,1)` -> 1 region champion
- F4: East vs West, South vs Midwest (matching `simulate_tournament` at line 215-216)
- Championship: semi1 winner vs semi2 winner

## File Changes

### [src/dashboard.py](src/dashboard.py)

Add a new function `_generate_bracket_progression()` that:

- Accepts `teams`, `matchups`, `prob_func` (the same ensemble prob_func from main.py)
- Builds the deterministic bracket round-by-round
- Returns formatted text for R32, S16, E8, F4, Championship in the same style as the R64 section

Modify `generate_full_report()`:

- Add a `prob_func` parameter (optional, defaults to None)
- Call `_generate_bracket_progression()` after the R64 section
- Output sections: "ROUND OF 32 PREDICTIONS (Detailed)", "SWEET 16 PREDICTIONS", "ELITE 8 PREDICTIONS", "FINAL FOUR", "CHAMPIONSHIP GAME"

### [src/main.py](src/main.py)

Pass `ensemble_prob_func` to `save_dashboard()` and `generate_full_report()` so the bracket progression can compute later-round probabilities. Currently `save_dashboard` is called at line 250 without the prob_func -- add it.

## Output Format (same style as R64)

```
========================================================================
  ROUND OF 32 PREDICTIONS (Deterministic Bracket)
========================================================================

  === EAST ===
  ( 1) Duke                 vs ( 9) TCU
       PICK: Duke (94.2%) [LOCK]  Round: R32

  ( 5) St. John's           vs ( 4) Kansas
       PICK: Kansas (61.3%) [LEAN]  Round: R32
  ...

========================================================================
  SWEET 16 PREDICTIONS
========================================================================
  ...

========================================================================
  CHAMPIONSHIP GAME
========================================================================
  ( 1) Duke                 vs ( 1) Michigan
       PICK: Duke (56.8%)  Round: Championship
```

## Important Notes

- This is a **deterministic chalk bracket** (always picks the favorite). It complements the Monte Carlo probabilistic output, which already shows advancement percentages accounting for upsets.
- The `prob_func` is the same narrative-wrapped ensemble function used by Monte Carlo, so all layers (injury, narrative, recency, chaos) are included.
- The bracket only appears in `full_report.txt`. The terminal output already shows the probabilistic advancement tables from Monte Carlo.

