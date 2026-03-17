"""Parameter weights configuration with dataset toggle flags.

Two weight sets are available:
  ORIGINAL_WEIGHTS  = hand-tuned weights (pre-optimizer, 70.5% historical accuracy)
  CORE_WEIGHTS      = optimizer-tuned weights (73.8% historical accuracy)

WTH layer is toggled via DATASET_CONFIG["use_wth_layer"].
  When True:  WTH chaos modifiers (sightline, altitude, chaos index) adjust
              upset volatility in every matchup computation.
  When False: pure weighted composite with no chaos adjustments.

OPTIMIZER RESULTS (1,070 historical tournament games, 2008-2025):
  ORIGINAL_WEIGHTS: 70.5% correct predictions
  CORE_WEIGHTS:     73.8% correct predictions   Δ = +3.4%

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
    "top50_perf":   0.04,     # Q1A win%
    "barthag":      0.04,     # Barttorvik BARTHAG
    "ftr":          0.025,    # FT% * FTR
    "ast_pct":      0.02,     # AST%
    "spi":          0.03,     # star power
    "exp":          0.025,    # roster experience
    "dvi":          0.02,     # defensive versatility
    "drb_pct":      0.025,    # DREB%
    "opp_to_pct":   0.025,    # TOV%D SOS-adjusted
    "rpi_rim":      0.02,     # rim protection
    "net_score":    0.025,    # NCAA NET
    "z_rating":     0.01,     # AdjEM+SOS composite
    "eff_height":   0.02,     # effective height (meters)
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
# Optimized via 10K grid search + 500-trial Bayesian (Optuna), 1,070 games.
# Accuracy: 73.8% → 78.7% on historical tournament data (2008-2025).
# Floor of 0.001 applied; no param removed.
# ─────────────────────────────────────────────────────────────────────────────
CORE_WEIGHTS = {
    # ── Tier 1: Consistency + Coaching + Game Flow (27%) ─────────────────
    #
    # The optimizer's biggest finding: consistency and coaching matter FAR
    # more than raw talent metrics in single-elimination. Teams that
    # perform reliably game-to-game and have experienced tournament coaches
    # outperform their statistical profiles.

    # [REAL] Consistency TR Rating from TeamRankings.csv.
    # Measures game-to-game consistency (low variance in performance).
    # Volatile teams with high ceilings but low floors are upset-prone
    # in single-elimination. Consistent teams are more reliable.
    # Example: A team rated 14.2/15 consistency rarely lays an egg.
    # Why #1: the optimizer found this is the strongest separator of
    # tourney winners vs losers once you control for basic quality metrics.
    "consistency":      0.1569,

    # [REAL] Coaching tournament factor from Coach Results.csv.
    # Formula: (tourney_wins + 2) / (tourney_games + 4) -- Bayesian smoothed
    # so coaches with small samples regress toward 0.5.
    # Example: Coach with 45 wins in 70 games → CTF = 47/74 = 0.635.
    #          First-time tournament coach → CTF = 2/4 = 0.500.
    # Why #2: once fixed proxies gave the optimizer real coaching data,
    # it discovered coaches with tournament pedigree consistently outperform.
    "ctf":              0.1463,

    # ── Tier 2: Offensive Identity + Legacy + Seeding (27%) ──────────────

    # [REAL] First-half point differential per game from StatSharp.
    # Measures how strong a team starts games -- the avg scoring margin
    # in the first half across the entire season.
    # Example: Duke H1_PD = +10.3 (dominates first halves),
    #          Lehigh H1_PD = +0.4 (barely competitive early).
    # Source: StatSharp 2026 half-scoring data (HalfScoring.csv).
    "offensive_burst":  0.0773,

    # [REAL] Legacy factor: PASE (Performance Against Seed Expectation)
    # from Team Results.csv. Measures how much a program historically
    # outperforms its seed. Winsorized to [-3, +5] to prevent outliers
    # (e.g. UConn +13.5) from anchoring the entire normalization scale.
    # Example: Michigan State PASE = +2.1 (consistently outperforms seed
    #          under Tom Izzo). Virginia PASE = -1.8 (underperforms).
    "legacy_factor":    0.0727,

    # [REAL] Seed score: 1 / seed. The selection committee's final judgment.
    # Encodes injuries, late-season form, eye test, and resume evaluation.
    # Example: 1-seed → 1.000. 8-seed → 0.125. 16-seed → 0.0625.
    "seed_score":       0.0671,

    # [REAL] Halftime adjustment: H2_PD - H1_PD from StatSharp.
    # Positive = team improves in second half relative to first half.
    # Measures coaching adjustments after halftime.
    # Example: Gonzaga q3_adj = +0.3 (consistent both halves),
    #          Long Island q3_adj = +3.2 (major 2nd-half improver).
    # Source: StatSharp 2026 half-scoring data (HalfScoring.csv).
    "q3_adj_strength":  0.0536,

    # ── Tier 3: Resilience + Quality + Shooting (15%) ────────────────────

    # [MIXED] Blowout resilience: 0.30*killshots_norm + 0.30*consistency_norm
    # + 0.20*(AdjEM/40) + 0.20*BDS. All inputs from real CSVs.
    # Can the team avoid or survive blowout losses?
    "blowout_resilience": 0.0390,

    # [REAL] BARTHAG from KenPom Barttorvik.csv.
    # Barttorvik's probability of beating an average Division I team.
    # Range 0.0-1.0. A 1-seed typically has BARTHAG > 0.95.
    "barthag":          0.0377,

    # [REAL] Merged shooting efficiency: 0.6*eFG% + 0.4*TS_approx.
    # Both eFG% and TS% come from KenPom Barttorvik.csv. They were r=0.966
    # correlated, so merging into one param avoids double-counting.
    # eFG% = (FGM + 0.5*3PM) / FGA -- credits 3-pointers for extra value.
    # TS% = PTS / (2 * (FGA + 0.44*FTA)) -- includes free throw drawing.
    # Example: Duke eFG%=56.2%, TS%=59.1% → shooting_eff = 0.574.
    "shooting_eff":     0.0369,

    # [MIXED] Rebound margin: (ORB% + DRB%)/2 - 0.5.
    # How much a team outrebounds opponents on both ends. Both ORB% and DRB%
    # come directly from KenPom Barttorvik.csv -- the blend is a simple average.
    # Example: A team with ORB%=35% and DRB%=75% → RBM = (0.35+0.75)/2 - 0.5 = 0.05.
    "rbm":              0.0357,

    # ── Tier 4: Experience + Defense + Schedule (13%) ────────────────────

    # [REAL] EXP from KenPom Barttorvik.csv (or KenPom Height.csv override).
    # Minutes-weighted average years of college experience across the roster.
    # Example: Gonzaga EXP=2.8 (veteran roster with transfers and upperclassmen)
    #          vs a freshman-heavy team at EXP=1.2.
    "exp":              0.0267,

    # [MIXED] Defensive versatility: 0.3*BLK%_norm + 0.3*STL%_norm
    # + 0.4*(1 - Opp3P%_norm). All from KenPom/Misc Stats CSVs.
    # Multi-dimensional defense: blocks + steals + perimeter defense.
    "dvi":              0.0234,

    # [REAL] ELITE SOS from KenPom Barttorvik.csv.
    # Strength of schedule: KenPom's metric of average opponent quality.
    # Example: Alabama SOS=40.7 (SEC gauntlet) vs Lehigh SOS=0.5 (Patriot).
    "sos":              0.0220,

    # [MIXED] Fragility: 0.35*(1-margin_norm) + 0.25*(1-consistency_norm)
    # + 0.20*(1-win_pct) + 0.20*(chaos_index*10). All inputs real.
    # Higher fragility = more upset-prone (inverted in scoring).
    "fragility_score":  0.0218,

    # [REAL] Free throw reliability: FT% * FTR (free throw rate).
    # Both from KenPom Barttorvik.csv. Getting to the line AND converting.
    # Critical in close tournament games where fouls increase.
    "ftr":              0.0210,

    # [REAL] KADJ EM from KenPom Barttorvik.csv.
    # Adjusted efficiency margin: points scored - points allowed per 100
    # possessions, adjusted for opponent strength. The gold standard metric.
    # Still the best SINGLE predictor (r~0.90 with tournament wins), but
    # the optimizer lowered it because SOS, BARTHAG, and other params
    # already carry much of AdjEM's signal.
    # Example: Duke AdjEM = +38.9 (elite). Lehigh AdjEM = -12.3.
    "adj_em":           0.0208,

    # ── Tier 5: Turnovers + Trends + Scoring Runs (6%) ──────────────────

    # [REAL] TOV% from KenPom Barttorvik.csv, SOS-adjusted post-load.
    # Turnover rate: turnovers / (FGA + 0.475*FTA + TO). Lower = better.
    # SOS-adjusted: divided by (team_SOS / league_avg_SOS).
    # Example: Purdue TO% = 8.4% (elite ball security) vs Norfolk St = 19.8%.
    "to_pct":           0.0185,

    # [MIXED] Momentum from TeamRankings LAST/HI/LO ranking trend.
    # All inputs are real: TR publishes LAST rank, season-HI rank,
    # season-LO rank. The formula computes (LO - LAST) / (LO - HI):
    # 1.0 when peaking (LAST == HI), 0.0 when slumping (LAST == LO).
    # Example: A team peaking at their season-best rank → momentum = 0.9.
    "momentum":         0.0177,

    # [MIXED] MSRP: killshots_per_game - killshots_conceded from EvanMiya.csv.
    # Scoring run differential. Real data, but not the actual "mid-season
    # resume predictor" metric. Positive = team goes on runs more than
    # opponents go on runs against them.
    "msrp":             0.0139,

    # ── Tier 6: Rebounding + Balance + Context (5%) ─────────────────────

    # [REAL] DREB% from KenPom Barttorvik.csv (z-score normalized).
    # Defensive rebounds / (defensive rebounds + opponent offensive rebounds).
    # Range is very narrow (65%-77%) so z-score normalization amplifies
    # the real but small differences between teams.
    "drb_pct":          0.0115,

    # [REAL] Scoring balance: 2PT%*2PTR + 3P%*3PTR.
    # From KenPom Barttorvik.csv columns 2PT%, 2PTR, 3PT%, 3PTR.
    # Rewards teams efficient from BOTH inside and outside the arc.
    "scoring_balance":  0.0098,

    # [REAL] TOV%D from KenPom Barttorvik.csv, SOS-adjusted post-load.
    # Opponent turnover rate: how many turnovers your defense forces.
    # Example: Houston forces TOs at 22% against a brutal schedule → elite.
    "opp_to_pct":       0.0084,

    # [MIXED] March readiness: 0.20*win% + 0.15*(win%+0.1) + 0.15*(1-3PTR)
    # + 0.10*win% + 0.10*DRB% + 0.10*(1-Opp3P%) + 0.10*neutral + 0.10*CWP.
    # Every input is real data from KenPom/TeamRankings.
    "march_readiness":  0.0084,

    # ── Tier 7: Variance + Depth + Niche (4%) ───────────────────────────

    # [REAL] Per-game scoring margin standard deviation from game logs.
    # Computed as std(team_score - opponent_score) across all regular-season
    # games in archive-3/game-logs/*.csv. INVERTED: lower std = more reliable.
    # Example: Purdue margin_std = 9.2 (rock solid) vs Troy margin_std = 19.1.
    "scoring_margin_std": 0.0063,

    # [MIXED] Z Rating: 0.45*AdjEM + 0.35*SOS + 3.0.
    # Cross-checks efficiency against schedule difficulty in one number.
    "z_rating":         0.0063,

    # [REAL] Bench depth from KenPom Height.csv.
    # Percentage of minutes played by non-starters. Deep benches survive
    # foul trouble, injuries, and the fatigue of a 6-game tournament run.
    "bds":              0.0058,

    # [MIXED] Rim protection: (BLK%_normalized + (1-OppRimFG%)_normalized) / 2.
    # BLK% from KenPom Barttorvik.csv. Rim FG% from Shooting Splits.csv.
    "rpi_rim":          0.0050,

    # [REAL] Win% vs Q1A opponents from Teamsheet Ranks.csv + Resumes.csv.
    # Q1A = games against teams ranked 1-15 in NET on their home court,
    # or 1-25 on a neutral court. The toughest possible opponents.
    "top50_perf":       0.0048,

    # [REAL] OREB% from KenPom Barttorvik.csv.
    # Offensive rebounds / (offensive rebounds + opponent defensive rebounds).
    # Second-chance points win tight tournament games.
    "orb_pct":          0.0043,

    # [REAL] NET rating from Teamsheet Ranks.csv, normalized: (68-rank)/68.
    # The NCAA selection committee's own holistic ranking metric.
    "net_score":        0.0041,

    # [MIXED] Foul trouble impact: -(1 - bench_depth) * (SPI / max(SPI, 0.01)).
    # Measures star dependency: if your star fouls out, how much do you drop?
    "foul_trouble_impact": 0.0036,

    # ── Tier 8: Tail params (floor = 0.001) ─────────────────────────────

    # [REAL] AST% from KenPom Barttorvik.csv.
    # Assisted field goals / total field goals. Measures ball movement.
    "ast_pct":          0.0019,

    # [REAL] Q3+Q4 loss rate from Teamsheet Ranks.csv (inverted).
    "q34_loss_rate":    0.0018,

    # [REAL] Q1 win% from Teamsheet Ranks.csv.
    "q1_record":        0.0013,

    # [MIXED] CWP: star 17+ at halftime → win probability.
    "cwp_star_17_half": 0.0013,

    # [REAL] PPG margin = PPPO*pace - PPPD*pace from KenPom Barttorvik.csv.
    "ppg_margin":       0.0012,

    # [MIXED] Clutch factor: 0.35*win_pct + 0.25*BARTHAG + 0.20*consistency
    # + 0.20*killshots_norm. All four inputs are real data.
    "clutch_factor":    0.0011,

    # [REAL] Effective height from KenPom Barttorvik.csv "EFF HGT" column.
    # Minutes-weighted average player height, converted to meters.
    "eff_height":       0.0011,

    # [PROXY] Star above average: (68 - roster_rank) / 68.
    "star_above_avg":   0.0010,

    # [REAL] Injury rank from EvanMiya.csv. Lower rank = healthier roster.
    "injury_health":    0.0010,

    # [PROXY] Star power index: (68-roster_rank)/68 * AdjEM/15 + champ_pick*0.5.
    "spi":              0.0010,
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

LOGISTIC_K = 6.0

ENSEMBLE_LAMBDA = 0.5

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
    "use_wth_layer": True,       # toggle WTH chaos modifiers on/off
    "use_cwp": True,
}

PARAM_KEYS = list(CORE_WEIGHTS.keys())
