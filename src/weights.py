"""Parameter weights configuration with dataset toggle flags.

Two weight sets are available:
  ORIGINAL_WEIGHTS  = hand-tuned weights (pre-optimizer, 70.5% historical accuracy)
  CORE_WEIGHTS      = optimizer-tuned weights (73.7% historical accuracy, leak-free)

WTH layer is toggled via DATASET_CONFIG["use_wth_layer"].
  When True:  WTH chaos modifiers (sightline, altitude, chaos index) adjust
              upset volatility in every matchup computation.
  When False: pure weighted composite with no chaos adjustments.

OPTIMIZER RESULTS (1,070 historical tournament games, 2008-2025, per-year normalization):
  ORIGINAL_WEIGHTS: 70.5% correct predictions
  CORE_WEIGHTS:     73.7% correct predictions   Δ = +3.2% (leak-free, per-year norm)

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
# Leak-free optimization with per-year normalization (no cross-year leakage).
# Multi-phase: greedy hill-climb + pairwise transfer + 3K random search.
# Per-year norm verified: each tournament year's field scaled independently.
# All weights >= 0.001 (user-required floor).
# Accuracy: 73.7% on 1,070 historical games (2008-2025), per-year norm.
# ─────────────────────────────────────────────────────────────────────────────
CORE_WEIGHTS = {
    # ── Tier 1: Raw Quality + Clutch + Elite Performance (42%) ────────────

    # ── Tier 1: Raw Quality + Power Rating (30%) ─────────────────────────

    # [REAL] KADJ EM from KenPom Barttorvik.csv.
    # The gold standard single predictor — #1 weight.
    "adj_em":           0.1050,

    # [REAL] BARTHAG from KenPom Barttorvik.csv.
    # Optimizer elevated this to #2: win probability proxy is a strong signal.
    "barthag":          0.0957,

    # [MIXED] Rebound margin: (ORB% + DRB%)/2 - 0.5.
    # Rebounding dominance is a top-3 March predictor.
    "rbm":              0.0882,

    # [REAL] Win% vs Q1A opponents from Resumes.csv.
    # Beating elite teams is the best predictor of tournament success.
    "top25_perf":       0.0746,

    # ── Tier 2: Scoring Runs + Turnovers + Consistency (19%) ────────────

    # [MIXED] MSRP: scoring run differential from EvanMiya.csv.
    "msrp":             0.0729,

    # [REAL] Per-game scoring margin std dev (inverted: lower = better).
    # Consistency / reliability — high-variance teams underperform.
    "scoring_margin_std": 0.0640,

    # [REAL] TOV% SOS-adjusted (inverted: lower = better).
    # Ball security is critical in March.
    "to_pct":           0.0533,

    # [MIXED] Blowout resilience: killshots + consistency + AdjEM + BDS.
    "blowout_resilience": 0.0533,

    # ── Tier 3: Clutch + Z-Rating + Assists (12%) ───────────────────────

    # [MIXED] Clutch factor: win_pct + BARTHAG + consistency + killshots.
    # Teams that close games win in March.
    "clutch_factor":    0.0504,

    # [MIXED] Z Rating: 0.45*AdjEM + 0.35*SOS + 3.0.
    "z_rating":         0.0369,

    # [REAL] AST% from KenPom Barttorvik.csv.
    "ast_pct":          0.0364,

    # ── Tier 4: Halftime adj + Defense + Rebounds + Scoring (11%) ───────

    # [REAL] Halftime adjustment: H2_PD - H1_PD from StatSharp.
    "q3_adj_strength":  0.0307,

    # [MIXED] Rim protection.
    "rpi_rim":          0.0273,

    # [REAL] DREB% from KenPom Barttorvik.csv (z-score normalized).
    "drb_pct":          0.0245,

    # [REAL] Scoring balance: 2PT%*2PTR + 3P%*3PTR.
    "scoring_balance":  0.0244,

    # ── Tier 5: Legacy + Foul + Depth + Experience (8%) ─────────────────

    # [REAL] Legacy factor: reformed PASE (difficulty-normalized,
    # decay-weighted, sqrt-N, capped [-3, +3]).
    "legacy_factor":    0.0158,

    # [MIXED] Foul trouble impact: -(1-bench_depth) * star_power.
    # Star-dependent teams are vulnerable in single-elimination.
    "foul_trouble_impact": 0.0145,

    # [PROXY] Star power index.
    "spi":              0.0129,

    # [REAL] EXP from KenPom Barttorvik.csv.
    # Reduced to ~1.3%; still a real signal, but no longer dominant.
    "exp":              0.0127,

    # [MIXED] CWP: star 17+ at halftime win probability.
    "cwp_star_17_half": 0.0102,

    # ── Tier 6: Supporting metrics (6%) ─────────────────────────────────

    # [MIXED] Momentum from TeamRankings ranking trend.
    "momentum":         0.0096,

    # [REAL] Bench depth from KenPom Height.csv.
    "bds":              0.0094,

    # [REAL] First-half point differential from StatSharp.
    "offensive_burst":  0.0089,

    # [REAL] Consistency TR Rating from TeamRankings.csv.
    "consistency":      0.0087,

    # [REAL] Injury rank from EvanMiya.csv.
    "injury_health":    0.0080,

    # [REAL] Free throw reliability: FT% * FTR.
    "ftr":              0.0072,

    # [REAL] NET rating from Teamsheet Ranks.csv.
    "net_score":        0.0070,

    # [REAL] Q1 win% from Teamsheet Ranks.csv.
    "q1_record":        0.0069,

    # ── Tier 7: Tail params (3%) ────────────────────────────────────────

    # [MIXED] Fragility (inverted).
    "fragility_score":  0.0047,

    # [REAL] Q3+Q4 loss rate from Teamsheet Ranks.csv (inverted).
    "q34_loss_rate":    0.0040,

    # [REAL] ELITE SOS from KenPom Barttorvik.csv.
    "sos":              0.0033,

    # [MIXED] March readiness composite.
    "march_readiness":  0.0032,

    # [MIXED] Defensive versatility: BLK% + STL% + perimeter defense.
    "dvi":              0.0028,

    # [REAL] TOV%D SOS-adjusted.
    "opp_to_pct":       0.0023,

    # [PROXY] Star above average.
    "star_above_avg":   0.0020,

    # [REAL] OREB% from KenPom Barttorvik.csv.
    "orb_pct":          0.0020,

    # [REAL] Effective height (meters).
    "eff_height":       0.0018,

    # [REAL] Merged shooting efficiency: 0.6*eFG% + 0.4*TS_approx.
    "shooting_eff":     0.0013,

    # [REAL] Coaching tournament factor from Coach Results.csv.
    "ctf":              0.0012,

    # [REAL] PPG margin.
    "ppg_margin":       0.0010,

    # [REAL] Seed score: 1 / seed.
    "seed_score":       0.0010,
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
