# NCAA 2026 March Madness Champion Predictor

Dual-track prediction engine combining a hand-tuned weighted composite model (Phase 1A) with an XGBoost ML pipeline (Phase 1B), blended via ensemble, and live-updated using Bayesian inference during the tournament.

## Architecture

- **Phase 1A**: 31-parameter weighted composite + logistic win probability + Monte Carlo (50K sims)
- **Phase 1B**: XGBoost trained on 15 years of tournament data (95 features, LOYO CV)
- **Ensemble**: Blends 1A + 1B with confidence scoring and disagreement flags
- **Phase 2**: Bayesian live updating as games are played (alpha decay per round)

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main
```

## Project Structure

```
src/             Core prediction engine
  models.py      Team/Matchup data classes
  equations.py   All 58 equations (core + niche + WTH + ML + Bayesian)
  composite.py   Phase 1A: weighted composite engine
  xgboost_model.py Phase 1B: XGBoost pipeline + SHAP
  ensemble.py    Blend 1A + 1B, confidence scoring
  monte_carlo.py Simulation engine (shared)
  bayesian.py    Phase 2: live Bayesian updater
  weights.py     Parameter weights + dataset toggle flags
  wth_layer.py   "What The Hell" chaos modifiers
  niche.py       Super niche parameters
  cinderella.py  Undervalued mid-major detector
  utils.py       Normalization, helpers
data/            Team stats, matchups, venues, historical data
notebooks/       Jupyter analysis notebooks
archive/         KenPom raw data (13 files)
archive-3/       Tournament + advanced data (38 files)
```
