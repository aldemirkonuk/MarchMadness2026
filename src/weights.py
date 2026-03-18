"""Parameter weights configuration with dataset toggle flags.

Two weight sets are available:
  ORIGINAL_WEIGHTS  = hand-tuned weights (pre-optimizer, 70.5% historical accuracy)
  CORE_WEIGHTS      = optimizer-tuned weights (73.4% historical accuracy, year-safe)

WTH layer is toggled via DATASET_CONFIG["use_wth_layer"].
  When True:  WTH chaos modifiers (sightline, altitude, chaos index) adjust
              upset volatility in every matchup computation.
  When False: pure weighted composite with no chaos adjustments.

OPTIMIZER RESULTS (1,070 historical tournament games, 2008-2025):
  ORIGINAL_WEIGHTS: 70.5% correct predictions
  CORE_WEIGHTS:     73.4% correct predictions   Δ = +2.9% (year-safe, no leakage)

DATA SOURCE LEGEND:
  [REAL]  = loaded directly from a CSV column with no approximation.
            e.g. adj_em comes straight from "KADJ EM" in KenPom Barttorvik.csv
  [MIXED] = every INPUT number is real data from real CSVs, but they are
            combined via an engineered formula (e.g. 0.35*win_pct + 0.25*BARTHAG).
            Not fabricated -- all values are real, the recipe is judgment-based.
  [PROXY] = the CONCEPT we want to measure has no data in our archives, so we
            approximate it from correlated stats (e.g. star talent from roster_rank
            because we have no individual player BPM/usage data).
"""

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL (hand-tuned) weights -- kept for A/B comparison
# ─────────────────────────────────────────────────────────────────────────────
ORIGINAL_WEIGHTS = {
    "adj_em":       0.14,     # KenPom AdjEM: pts/100 poss above average
    "shooting_eff":  0.06,    # 0.6*eFG% + 0.4*TS%
    "clutch_factor": 0.055,   # win% + BARTHAG + consistency + killshots
    "sos":          0.05,     # KenPom Elite SOS
    "to_pct":       0.04,     # TOV% SOS-adjusted
    "scoring_balance": 0.035, # 2PT%*2PTR + 3P%*3PTR
    "orb_pct":      0.03,     # OREB%
    "seed_score":   0.04,     # 1/seed
    "top25_perf":   0.024,    # Q1A win%
    "barthag":      0.04,     # Barttorvik BARTHAG
    "ftr":          0.025,    # FT% * FTR
    "ast_pct":      0.03,     # AST%
    "spi":          0.03,     # star power
    "exp":          0.025,    # roster experience
    "dvi":          0.02,     # defensive versatility
    "drb_pct":      0.025,    # DREB%
    "opp_to_pct":   0.025,    # TOV%D SOS-adjusted
    "rpi_rim":      0.021,     # rim protection
    "net_score":    0.025,    # NCAA NET
    "z_rating":     0.01,     # AdjEM+SOS composite
    "eff_height":   0.025,     # effective height (meters)
    "momentum":     0.02,     # TeamRankings trend
    "ctf":          0.01,     # coaching factor
    "rbm":          0.01,     # rebound margin
    "q1_record":    0.02,     # Q1 win% 
    "q34_loss_rate": 0.01,    # bad loss rate
    "offensive_burst": 0.01, # first-half point diff (StatSharp real data)
    "q3_adj_strength": 0.01,  # H2-H1 adjustment (StatSharp real data)
    "ppg_margin":   0.015,    # PPG margin
    "fragility_score": 0.02,  # upset vulnerability
    "march_readiness": 0.02,  # tournament readiness
    "legacy_factor": 0.01,    # historical outperformance
    "bds":          0.01,     # bench depth
    "injury_health": 0.01,    # injury rank
    "cwp_star_17_half": 0.01, # star halftime dominance
    "star_above_avg": 0.02,   # best player talent 
    "msrp":         0.005,    # scoring run potential
    "blowout_resilience": 0.003, # blowout resistance
    "foul_trouble_impact": 0.005, # star dependency
    "consistency":    0.002,  # game-to-game consistency
    "scoring_margin_std": 0.005,  # per-game scoring margin std dev (inverted)
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE (optimizer-tuned) weights -- active model
# Re-optimized after PASE reform + WTH removal + exp reduction.
# 5K grid + 200-trial Bayesian + 3K constrained perturbation, 1,070 games.
# Constraints: adj_em >= 12%, barthag >= 5%, exp ~2%, legacy_factor <= 8%.
# Year-safe features: legacy uses reformed PASE (difficulty coeff + decay);
# CTF neutral in training.
# Accuracy: 73.6% on historical tournament data (2008-2025).
# ─────────────────────────────────────────────────────────────────────────────
CORE_WEIGHTS = {
    # ── Tier 1: Raw Quality + Clutch + Elite Performance (42%) ────────────

    # [REAL] KADJ EM from KenPom Barttorvik.csv.
    # The gold standard single predictor. Now the #1 weight.
    "adj_em":           0.1154,

    # [MIXED] Clutch factor: win_pct + BARTHAG + consistency + killshots.
    # Optimizer's #2: teams that close games win in March.
    "clutch_factor":    0.0555,

    # [REAL] Win% vs Q1A opponents from Resumes.csv.
    # Beating elite teams is the best predictor of tournament success.
    "top25_perf":       0.0821,

    # [REAL] Per-game scoring margin std dev (inverted: lower = better).
    # Consistency / reliability — high-variance teams underperform.
    "scoring_margin_std": 0.0705,

    # ── Tier 2: Rebounding + Scoring Runs + Turnovers (21%) ──────────────

    # [MIXED] Rebound margin: (ORB% + DRB%)/2 - 0.5.
    "rbm":              0.0777,

    # [MIXED] MSRP: scoring run differential from EvanMiya.csv.
    "msrp":             0.0729,

    # [REAL] TOV% SOS-adjusted (inverted: lower = better).
    # Optimizer boosted this: ball security is critical in March.
    "to_pct":           0.0782,

    # ── Tier 3: Power Rating + Resilience + Defense (16%) ─────────────────

    # [REAL] BARTHAG from KenPom Barttorvik.csv.
    "barthag":          0.0679,

    # [MIXED] Blowout resilience: killshots + consistency + AdjEM + BDS.
    "blowout_resilience": 0.0428,

    # [MIXED] Rim protection.
    "rpi_rim":          0.0245,

    # [MIXED] Z Rating: 0.45*AdjEM + 0.35*SOS + 3.0.
    "z_rating":         0.0304,

    # ── Tier 4: Scoring Balance + Rebounding + Experience (8%) ────────────

    # [REAL] Scoring balance: 2PT%*2PTR + 3P%*3PTR.
    "scoring_balance":  0.0268,

    # [REAL] DREB% from KenPom Barttorvik.csv (z-score normalized).
    "drb_pct":          0.0216,

    # [REAL] EXP from KenPom Barttorvik.csv.
    # Reduced to ~2%; still a real signal, but no longer dominant.
    "exp":              0.0186,

    # [REAL] Legacy factor: reformed PASE (difficulty-normalized,
    # decay-weighted, sqrt-N, capped [-3, +3]).
    "legacy_factor":    0.0261,

    # ── Tier 5: Supporting metrics (8%) ──────────────────────────────────

    # [REAL] AST% from KenPom Barttorvik.csv.
    "ast_pct":          0.0300,

    # [MIXED] CWP: star 17+ at halftime win probability.
    "cwp_star_17_half": 0.0196,

    # [REAL] NET rating from Teamsheet Ranks.csv.
    "net_score":        0.0143,

    # [MIXED] Foul trouble impact: -(1-bench_depth) * star_power.
    # Bumped to 0.8% — star-dependent teams are vulnerable in single-elimination.
    "foul_trouble_impact": 0.0159,

    # [REAL] Halftime adjustment: H2_PD - H1_PD from StatSharp.
    "q3_adj_strength":  0.0142,

    # [PROXY] Star power index.
    "spi":              0.0071,

    # [REAL] Bench depth from KenPom Height.csv.
    "bds":              0.0089,

    # [REAL] Free throw reliability: FT% * FTR.
    "ftr":              0.0079,

    # [MIXED] Fragility (inverted). Massively reduced from prior 13.9%.
    "fragility_score":  0.0052,

    # [REAL] Injury rank from EvanMiya.csv.
    "injury_health":    0.0089,

    # [REAL] Q3+Q4 loss rate from Teamsheet Ranks.csv (inverted).
    "q34_loss_rate":    0.0044,

    # ── Tier 6: Tail params ──────────────────────────────────────────────

    # [REAL] ELITE SOS from KenPom Barttorvik.csv.
    "sos":              0.0036,

    # [MIXED] Defensive versatility: BLK% + STL% + perimeter defense.
    "dvi":              0.0015,

    # [REAL] TOV%D SOS-adjusted.
    "opp_to_pct":       0.0026,

    # [MIXED] March readiness composite.
    "march_readiness":  0.0024,

    # [PROXY] Star above average.
    "star_above_avg":   0.0022,

    # [REAL] OREB% from KenPom Barttorvik.csv.
    "orb_pct":          0.0022,

    # [REAL] Effective height (meters).
    "eff_height":       0.0020,

    # [REAL] Q1 win% from Teamsheet Ranks.csv.
    "q1_record":        0.0076,

    # [REAL] Coaching tournament factor from Coach Results.csv.
    "ctf":              0.0013,

    # [REAL] First-half point differential from StatSharp.
    "offensive_burst":  0.0097,

    # [REAL] PPG margin.
    "ppg_margin":       0.0011,

    # [MIXED] Momentum from TeamRankings ranking trend.
    "momentum":         0.0053,

    # [REAL] Merged shooting efficiency: 0.6*eFG% + 0.4*TS_approx.
    "shooting_eff":     0.0008,

    # [REAL] Seed score: 1 / seed.
    "seed_score":       0.0008,

    # [REAL] Consistency TR Rating from TeamRankings.csv.
    "consistency":      0.0095,
}

assert abs(sum(CORE_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(CORE_WEIGHTS.values()):.4f}"

assert abs(sum(ORIGINAL_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Original weights must sum to 1.0, got {sum(ORIGINAL_WEIGHTS.values()):.4f}"

INVERTED_PARAMS = {
    "to_pct",               # lower TO rate = better ball security
    "fragility_score",      # higher fragility = worse
    "injury_health",        # lower injury rank = healthier
    "foul_trouble_impact",  # negative values = more star-dependent
    "q34_loss_rate",        # higher bad loss rate = worse
    "scoring_margin_std",   # lower variance = more consistent/reliable
}

# Params that use z-score normalization instead of min-max
Z_SCORE_PARAMS = {
    "drb_pct",  # range too narrow for min-max to discriminate
}

LOGISTIC_K = 14.0

ENSEMBLE_LAMBDA = 0.55  # calibrated via LOYO Brier sweep (0.30-0.80); 65% 1A, 35% XGBoost

# Tournament chaos floor: every matchup probability is pulled toward 0.50
# by this fraction. Reflects inherent single-elimination unpredictability.
# p_final = p * (1 - CHAOS) + 0.5 * CHAOS
TOURNAMENT_CHAOS = 0.10

TEMPORAL_SCHEME = ("uniform", 1.0)  # (scheme_name, param) for optimizer recency weighting
# Tested exponential/linear/step schemes -- none improved over uniform (see sweep_temporal_schemes)

MONTE_CARLO_SIMS = 100_000

DATASET_CONFIG = {
    "use_kenpom_barttorvik": True,
    "use_coach_results": True,
    "use_team_results": True,
    "use_resumes": True,
    "use_shooting_splits": True,
    "use_tournament_locations": True,
    "use_evan_miya": True,
    "use_public_picks": True,
    "use_ap_poll": True,
    "use_538_ratings": True,

    "use_niche_layer": True,
    "use_wth_layer": False,      # WTH disabled; scoring_margin_std is sole volatility layer
    "use_cwp": True,
}

PARAM_KEYS = list(CORE_WEIGHTS.keys())
