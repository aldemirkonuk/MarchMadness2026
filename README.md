# NCAA 2026 March Madness Champion Predictor

Dual-track prediction engine: weighted composite (1A) + XGBoost (1B), blended via ensemble, with injury adjustment, recency weighting, and narrative overlay.

## Baseline Performance

| Component | Accuracy | Brier | Data |
|-----------|----------|-------|------|
| **1A (composite)** | **73.6%** | 0.203 | 1,070 games (2008–2025) |
| **Ensemble (1A+1B)** | 72.2% | 0.191 | Same |
| **LOYO CV (1A)** | 73.0% | 0.208 | Leave-one-year-out |

## Architecture

- **Phase 1A**: 42-parameter weighted composite + logistic win prob + Monte Carlo (100K sims)
- **Phase 1B**: XGBoost (25 features × 3, LOYO trained on 2010–2025)
- **Ensemble**: 55% 1A / 45% 1B, confidence-weighted lambda
- **Injury model**: BPR-based penalties, multi-category leader amplifier, collapse-risk detection
- **Narrative layer**: Manual overlay (capped ±5%), personnel_loss discounted when injury model active
- **Social validation**: Diagnostic comparison vs analyst upset picks (no prob changes)

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main
```

Outputs: `data/results/full_report.txt`, `championship_odds.csv`, `matchup_predictions.csv`, `power_rankings.csv`

## Project Structure

```
src/             Core engine
  composite.py   Phase 1A
  xgboost_model.py  Phase 1B
  ensemble.py    Blend 1A+1B
  injury_model.py   BPR-based injury degradation
  narrative_layer.py  Qualitative overlay
  social_validation.py  Analyst vs model diagnostic
  monte_carlo.py 100K sims
  weights.py     CORE_WEIGHTS, config
data/            matchups, injuries, narratives, EvanMiya
notebooks/       model_diagnostics.ipynb
archive-3/       KenPom, Tournament Matchups, game logs
```
