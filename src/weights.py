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
# ─────────────────────────────────────────────────────────────────────────────
CORE_WEIGHTS = {
    # ── Tier 1: SOS + Turnovers + NET + PPG (49%) ────────────────────────
    #
    # These four params dominate because they each capture a DIFFERENT
    # dimension of quality that the tournament rewards: ball security (to_pct),
    # schedule difficulty (sos), committee evaluation (net_score), and raw
    # scoring power (ppg_margin). Together they cover 49% of the model.

    # [REAL] TOV% from KenPom Barttorvik.csv, SOS-adjusted post-load.
    # Turnover rate: turnovers / (FGA + 0.475*FTA + TO). Lower = better.
    # SOS-adjusted: divided by (team_SOS / league_avg_SOS) so teams with
    # low TOs against tough schedules score higher than cupcake-schedule TOs.
    # Example: Purdue TO% = 8.4% (elite ball security) vs Norfolk St = 19.8%.
    #          After SOS-adjustment, Purdue's advantage is even larger because
    #          they protect the ball against Big Ten defenses.
    # Why so high: optimizer found TO% has the WIDEST 1-seed vs 16-seed
    # separation of any param after normalization (+0.755 gap).
    "to_pct":       0.1414,

    # [REAL] ELITE SOS from KenPom Barttorvik.csv.
    # Strength of schedule: KenPom's metric of average opponent quality.
    # Sensitivity analysis ranked this #1 -- perturbing SOS weight changes
    # prediction accuracy more than any other single param.
    # Example: Alabama SOS=40.7 (SEC gauntlet) vs Lehigh SOS=0.5 (Patriot).
    #          A 25-win team in the SEC is fundamentally different from a
    #          25-win team in a one-bid conference. SOS captures this.
    # Why so high: the NCAA tournament is ALL cross-conference matchups.
    # Teams that have been battle-tested against elite opponents handle the
    # pressure and talent level jump far better.
    "sos":          0.1246,

    # [REAL] NET rating from Teamsheet Ranks.csv, normalized: (68-rank)/68.
    # The NCAA selection committee's own holistic ranking metric.
    # NET combines: game results, strength of schedule, game location,
    # scoring margin (capped at 10), and net offensive/defensive efficiency.
    # Example: Duke NET=#1 → score=0.985. Prairie View NET=#68 → score=0.0.
    # Why so high: NET encodes information the committee uses for seeding
    # decisions, including factors we can't directly measure (eye test,
    # injury timing, key wins/losses recency).
    "net_score":    0.1258,

    # [REAL] PPG margin = PPPO*pace - PPPD*pace from KenPom Barttorvik.csv.
    # Points per game scored minus points per game allowed.
    # Uses PPPO (points per possession offense) * pace rather than raw PPG,
    # so it accounts for tempo differences between teams.
    # Example: Duke margin=+18.8 (dominant). Arkansas margin=+9.5 (solid).
    #          LIU Brooklyn margin=-7.2 (outscored regularly).
    "ppg_margin":   0.1008,

    # ── Tier 2: Boards + Experience + Stars + Q1 Start (24%) ─────────────
    #
    # Physical dominance (rebounding), roster maturity, star dependency,
    # elite-opponent performance, and explosive offensive capability.

    # [MIXED] Rebound margin: (ORB% + DRB%)/2 - 0.5.
    # How much a team outrebounds opponents on both ends. Both ORB% and DRB%
    # come directly from KenPom Barttorvik.csv -- the blend is a simple average.
    # Example: A team with ORB%=35% and DRB%=75% → RBM = (0.35+0.75)/2 - 0.5 = 0.05.
    #          Positive = outrebounding, negative = getting outrebounded.
    # Why high: the optimizer found that composite rebounding dominance is a
    # stronger predictor than ORB% or DRB% individually.
    "rbm":          0.0663,

    # [PROXY] Q1 start strength: 0.5*AdjO_norm + 0.3*pace_norm + 0.2*three_pri.
    # [REAL] First-half point differential per game from StatSharp.
    # Measures how strong a team starts games -- the avg scoring margin
    # in the first half across the entire season.
    # Example: Duke H1_PD = +10.3 (dominates first halves),
    #          Lehigh H1_PD = +0.4 (barely competitive early).
    # Source: StatSharp 2026 half-scoring data (HalfScoring.csv).
    "offensive_burst": 0.0681,

    # [REAL] EXP from KenPom Barttorvik.csv (or KenPom Height.csv override).
    # Minutes-weighted average years of college experience across the roster.
    # Example: Gonzaga EXP=2.8 (veteran roster with transfers and upperclassmen)
    #          vs a freshman-heavy team at EXP=1.2.
    # Why: experienced rosters historically overperform in the pressure cooker
    # of single-elimination tournament basketball. They've been there before.
    "exp":          0.0535,

    # [MIXED] Foul trouble impact: -(1 - bench_depth) * (SPI / max(SPI, 0.01)).
    # Measures star dependency: if your star fouls out, how much do you drop?
    # BDS (bench depth) and SPI (star power) are both real data inputs.
    # The formula is engineered to penalize teams that are both star-dependent
    # AND have shallow benches. Negative = more vulnerable.
    # Example: A team where one player dominates usage AND the bench is weak
    #          → high foul_trouble_impact → inverted to penalize in scoring.
    "foul_trouble_impact": 0.0437,

    # [REAL] Win% vs Q1A opponents from Teamsheet Ranks.csv + Resumes.csv.
    # Q1A = games against teams ranked 1-15 in NET on their home court,
    # or 1-25 on a neutral court. The toughest possible opponents.
    # Example: A team that goes 8-2 in Q1A games = 0.80 top50_perf.
    #          A mid-major that goes 0-3 in Q1A = 0.00.
    "top50_perf":   0.0326,

    # ── Tier 3: Defense + Quadrants + Shooting (13%) ─────────────────────

    # [REAL] TOV%D from KenPom Barttorvik.csv, SOS-adjusted post-load.
    # Opponent turnover rate: how many turnovers your defense forces.
    # SOS-adjusted: multiplied by (team_SOS / league_avg_SOS) so forcing TOs
    # against good teams counts more than forcing TOs against bad teams.
    # Example: Houston forces TOs at 22% against a brutal schedule → elite.
    "opp_to_pct":   0.0230,

    # [REAL] Q1 win% from Teamsheet Ranks.csv.
    # Win percentage in Quadrant 1 games (vs NET 1-30 at home, 1-50 neutral,
    # 1-75 away). Directly from the NCAA's quadrant system.
    # Example: Michigan goes 10-2 in Q1 → q1_record = 0.833.
    "q1_record":    0.0229,

    # [REAL] Merged shooting efficiency: 0.6*eFG% + 0.4*TS_approx.
    # Both eFG% and TS% come from KenPom Barttorvik.csv. They were r=0.966
    # correlated, so merging into one param avoids double-counting.
    # eFG% = (FGM + 0.5*3PM) / FGA -- credits 3-pointers for extra value.
    # TS% = PTS / (2 * (FGA + 0.44*FTA)) -- includes free throw drawing.
    # Example: Duke eFG%=56.2%, TS%=59.1% → shooting_eff = 0.574.
    "shooting_eff": 0.0208,

    # [REAL] Seed score: 1 / seed. The selection committee's final judgment.
    # Encodes injuries, late-season form, eye test, and resume evaluation.
    # Example: 1-seed → 1.000. 8-seed → 0.125. 16-seed → 0.0625.
    "seed_score":   0.0176,

    # [MIXED] Z Rating: 0.45*AdjEM + 0.35*SOS + 3.0.
    # Tries to load real Z Rating from "Z Rating Teams.csv" first.
    # Falls back to the linear approximation if no 2026 data exists.
    # Cross-checks efficiency against schedule difficulty in one number.
    "z_rating":     0.0160,

    # [PROXY] Star above average: (68 - roster_rank) / 68.
    # We have ZERO individual player stats in our archives (all CSVs are
    # team-level). EvanMiya's roster_rank is the closest proxy -- it ranks
    # overall team talent recruiting, not individual player performance.
    # To get real player BPM/usage/PPG we'd need to scrape player pages.
    # Example: Duke roster_rank=#1 → star_above_avg = 0.985.
    #          A 16-seed with roster_rank=#65 → 0.044.
    "star_above_avg": 0.0127,

    # [REAL] Consistency TR Rating from TeamRankings.csv.
    # Measures game-to-game consistency (low variance in performance).
    # Volatile teams with high ceilings but low floors are upset-prone
    # in single-elimination. Consistent teams are more reliable.
    # Example: A team rated 14.2/15 consistency rarely lays an egg.
    "consistency":  0.0096,

    # [REAL] AST% from KenPom Barttorvik.csv.
    # Assisted field goals / total field goals. Measures ball movement.
    # Teams with high AST% generate open shots through passing rather
    # than relying on individual creation. Translates to tournament play.
    "ast_pct":      0.0115,

    # ── Tier 4: Efficiency + Trend + Context (6%) ────────────────────────

    # [REAL] KADJ EM from KenPom Barttorvik.csv.
    # Adjusted efficiency margin: points scored - points allowed per 100
    # possessions, adjusted for opponent strength. The gold standard metric.
    # Still the best SINGLE predictor (r~0.90 with tournament wins), but
    # the optimizer lowered it because SOS, NET, and PPG_MARGIN already
    # carry much of AdjEM's signal. With 14% on AdjEM PLUS ~20% on
    # AdjEM-correlated params, the old model gave AdjEM ~35% of total
    # influence. The optimizer redistributes that more efficiently.
    # Example: Duke AdjEM = +38.9 (elite). Lehigh AdjEM = -12.3.
    "adj_em":       0.0098,

    # [REAL] Injury rank from EvanMiya.csv. Lower rank = healthier roster.
    # Inverted in scoring so rank #1 (healthiest) gets the best score.
    # Example: A fully healthy team ranked #3 → injury_health = 3.0
    #          → after inversion, gets a high normalized score.
    "injury_health": 0.0089,

    # [MIXED] Momentum from TeamRankings LAST/HI/LO ranking trend.
    # All inputs are real: TR publishes LAST rank, season-HI rank,
    # season-LO rank. The formula computes (LO - LAST) / (LO - HI):
    # 1.0 when peaking (LAST == HI), 0.0 when slumping (LAST == LO).
    # Blended with win% and Q1A rate. Conditional: lower seeds with
    # hard SOS get a different blend (more weight on Q1A rate).
    # Example: A team peaking at their season-best rank → momentum ≈ 0.9.
    "momentum":     0.0087,

    # [MIXED] March readiness: 0.20*win% + 0.15*(win%+0.1) + 0.15*(1-3PTR)
    # + 0.10*win% + 0.10*DRB% + 0.10*(1-Opp3P%) + 0.10*neutral + 0.10*CWP.
    # Every input is real data from KenPom/TeamRankings. The blend formula
    # weights attributes that historically predict March success:
    # not 3PT-dependent, good defense, good on neutral courts.
    "march_readiness": 0.0082,

    # [REAL] Coaching tournament factor from Coach Results.csv.
    # Formula: (tourney_wins + 2) / (tourney_games + 4) -- Bayesian smoothed
    # so coaches with small samples regress toward 0.5.
    # Example: Coach with 45 wins in 70 games → CTF = 47/74 = 0.635.
    #          First-time tournament coach → CTF = 2/4 = 0.500.
    "ctf":          0.0081,

    # [REAL] DREB% from KenPom Barttorvik.csv (z-score normalized).
    # Defensive rebounds / (defensive rebounds + opponent offensive rebounds).
    # Range is very narrow (65%-77%) so z-score normalization amplifies
    # the real but small differences between teams.
    # Example: A team at 75% DRB% is in the 95th percentile after z-score.
    "drb_pct":      0.0073,

    # [REAL] Bench depth from KenPom Height.csv.
    # Percentage of minutes played by non-starters. Deep benches survive
    # foul trouble, injuries, and the fatigue of a 6-game tournament run.
    # Example: A team where bench plays 30% of minutes → BDS = 0.30.
    "bds":          0.0062,

    # [REAL] OREB% from KenPom Barttorvik.csv.
    # Offensive rebounds / (offensive rebounds + opponent defensive rebounds).
    # Second-chance points win tight tournament games.
    # Example: A dominant offensive rebounding team at 35% ORB%.
    "orb_pct":      0.0061,

    # ── Tier 5: Clutch + Defense + Height (4%) ───────────────────────────

    # [MIXED] Clutch factor: 0.35*win_pct + 0.25*BARTHAG + 0.20*consistency
    # + 0.20*killshots_norm. All four inputs are real data:
    # - win_pct from KenPom W/L columns
    # - BARTHAG from KenPom Barttorvik.csv
    # - consistency_rating from TeamRankings.csv
    # - killshots from EvanMiya.csv (scoring runs per game)
    # The 0.35/0.25/0.20/0.20 blend is engineering judgment.
    # Example: A team with high win%, high BARTHAG, consistent performance,
    #          and good scoring run differential → clutch ≈ 0.85.
    "clutch_factor": 0.0058,

    # [MIXED] Rim protection: (BLK%_normalized + (1-OppRimFG%)_normalized) / 2.
    # BLK% from KenPom Barttorvik.csv. Rim FG% from Shooting Splits.csv.
    # Interior defense that alters shots even when not blocking them.
    "rpi_rim":      0.0054,

    # [REAL] BARTHAG from KenPom Barttorvik.csv.
    # Barttorvik's probability of beating an average Division I team.
    # Range 0.0-1.0. A 1-seed typically has BARTHAG > 0.95.
    "barthag":      0.0049,

    # [REAL] Effective height from KenPom Barttorvik.csv "EFF HGT" column.
    # Minutes-weighted average player height, converted to meters.
    # Taller teams control the paint, rebound better, and contest shots.
    # Example: Duke EFF HGT = 80.2 inches → 2.04 meters.
    #          A guard-heavy team at 76 inches → 1.93 meters.
    "eff_height":   0.0042,

    # [MIXED] Defensive versatility: 0.3*BLK%_norm + 0.3*STL%_norm
    # + 0.4*(1 - Opp3P%_norm). All from KenPom/Misc Stats CSVs.
    # Multi-dimensional defense: blocks + steals + perimeter defense.
    "dvi":          0.0031,

    # [MIXED] Blowout resilience: 0.30*killshots_norm + 0.30*consistency_norm
    # + 0.20*(AdjEM/40) + 0.20*BDS. All inputs from real CSVs.
    # Can the team avoid or survive blowout losses?
    "blowout_resilience": 0.0028,

    # ── Tier 6: Niche + CWP + Legacy (3%) ────────────────────────────────

    # [MIXED] CWP: star 17+ at halftime → win probability.
    # Formula: 0.40*BARTHAG + 0.30*(68-roster_rank)/68 + 0.15*Q1A_rate
    # + 0.15*win_pct. We have NO actual halftime scoring data -- this
    # approximates "when your star dominates, how often do you win?"
    "cwp_star_17_half": 0.0023,

    # [REAL] Legacy factor: PASE (Performance Against Seed Expectation)
    # from Team Results.csv. Measures how much a program historically
    # outperforms its seed. Winsorized to [-3, +5] to prevent outliers
    # (e.g. UConn +13.5) from anchoring the entire normalization scale.
    # Example: Michigan State PASE = +2.1 (consistently outperforms seed
    #          under Tom Izzo). Virginia PASE = -1.8 (underperforms).
    "legacy_factor": 0.0022,

    # [PROXY] Star power index: (68-roster_rank)/68 * AdjEM/15 + champ_pick*0.5.
    # NOT real BPM * Usage / League_Average as originally intended.
    # We have no individual player BPM or usage data in our archives.
    # roster_rank (from EvanMiya) ranks team talent, not individual stars.
    # champ_pick (from Public Picks) adds public sentiment.
    "spi":          0.0021,

    # [MIXED] Fragility: 0.35*(1-margin_norm) + 0.25*(1-consistency_norm)
    # + 0.20*(1-win_pct) + 0.20*(chaos_index*10). All inputs real.
    # Higher fragility = more upset-prone (inverted in scoring).
    "fragility_score": 0.0019,

    # [REAL] Scoring balance: 2PT%*2PTR + 3P%*3PTR.
    # From KenPom Barttorvik.csv columns 2PT%, 2PTR, 3PT%, 3PTR.
    # Rewards teams efficient from BOTH inside and outside the arc.
    # A team that only makes 3s can go cold; a team that only scores
    # inside can be packed in. Balance = harder to game-plan against.
    "scoring_balance": 0.0014,

    # [REAL] Q3+Q4 loss rate from Teamsheet Ranks.csv (inverted).
    # (Q3_losses + Q4_losses) / (Q3+Q4 total games). Bad losses.
    # Teams that drop games against weak opponents are unreliable.
    # Example: A 2-seed that lost to a 200+ NET team → high q34_loss_rate.
    "q34_loss_rate": 0.0014,

    # [REAL] Free throw reliability: FT% * FTR (free throw rate).
    # Both from KenPom Barttorvik.csv. Getting to the line AND converting.
    # Critical in close tournament games where fouls increase.
    "ftr":          0.0012,

    # [REAL] Halftime adjustment: H2_PD - H1_PD from StatSharp.
    # Positive = team improves in second half relative to first half.
    # Measures coaching adjustments after halftime.
    # Example: Gonzaga q3_adj = +0.3 (consistent both halves),
    #          Long Island q3_adj = +3.2 (major 2nd-half improver).
    # Source: StatSharp 2026 half-scoring data (HalfScoring.csv).
    "q3_adj_strength": 0.0011,

    # [MIXED] MSRP: killshots_per_game - killshots_conceded from EvanMiya.csv.
    # Scoring run differential. Real data, but not the actual "mid-season
    # resume predictor" metric. Positive = team goes on runs more than
    # opponents go on runs against them.
    "msrp":         0.001,

    # [REAL] Per-game scoring margin standard deviation from game logs.
    # Computed as std(team_score - opponent_score) across all regular-season
    # games in archive-3/game-logs/*.csv. INVERTED: lower std = more reliable.
    # High-variance teams can blow anyone out but also lose to anyone.
    # In single-elimination, consistency beats ceiling. A team with margin_std
    # of 8 wins by similar margins each game; one with 18 swings wildly.
    # Example: Purdue margin_std = 9.2 (rock solid) vs Troy margin_std = 19.1.
    "scoring_margin_std": 0.0050,
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
