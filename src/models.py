"""Data classes for teams, matchups, and simulation results."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Team:
    name: str
    seed: int
    region: str
    conference: str = ""

    # Tier 1 core parameters
    adj_em: float = 0.0
    adj_o: float = 0.0
    adj_d: float = 0.0
    shooting_eff: float = 0.0   # merged: 0.6*eFG% + 0.4*TS%
    efg_pct: float = 0.0       # raw eFG% (used in shooting_eff)
    ts_pct: float = 0.0        # raw TS% (used in shooting_eff)
    sos: float = 0.0
    to_pct: float = 0.0        # SOS-adjusted turnover rate
    clutch_factor: float = 0.0
    scoring_balance: float = 0.0  # 2PT%*2PTR + 3P%*3PTR (paint+perimeter)
    three_pri: float = 0.0       # kept as raw stat, not weighted
    orb_pct: float = 0.0
    seed_score: float = 0.0
    top25_perf: float = 0.0

    # Tier 2 strong parameters
    ftr: float = 0.0
    ast_pct: float = 0.0
    spi: float = 0.0
    exp: float = 0.0
    dvi: float = 0.0
    drb_pct: float = 0.0
    opp_to_pct: float = 0.0    # SOS-adjusted opponent TO rate
    rpi_rim: float = 0.0
    eff_height: float = 0.0    # minutes-weighted effective height in meters

    # Tier 3 secondary
    momentum: float = 0.0
    ctf: float = 0.0
    rbm: float = 0.0
    q1_record: float = 0.0     # real Q1 win% from Teamsheet Ranks
    q34_loss_rate: float = 0.0 # bad loss rate (Q3+Q4 losses / games, inverted)
    offensive_burst: float = 0.0  # first-half point differential (real from StatSharp)
    q3_adj_strength: float = 0.0  # second-half adjustment: H2_PD - H1_PD (real from StatSharp)
    legacy_factor: float = 0.0
    bds: float = 0.0
    pace: float = 0.0

    # Niche
    msrp: float = 0.0
    blowout_resilience: float = 0.0
    foul_trouble_impact: float = 0.0

    # CWP composites
    fragility_score: float = 0.0
    march_readiness: float = 0.0

    # Computed from new data (used as weight keys)
    net_score: float = 0.0
    ppg_margin: float = 0.0
    injury_health: float = 0.0
    star_above_avg: float = 0.0
    z_rating: float = 0.0
    cwp_star_17_half: float = 0.5
    consistency: float = 0.0
    scoring_margin_std: float = 0.0  # std dev of per-game scoring margin (game logs)

    # WTH modifiers (not scored, used as adjustments)
    chaos_index: float = 0.0
    three_pt_std: float = 0.0

    # Phase 8: Recency-weighted metrics
    form_trend: float = 0.5           # normalized slope of recent performance [0,1]
    recency_rating_norm: float = 0.5  # normalized recency-weighted margin [0,1]

    # Computed scores
    team_strength: float = 0.0
    normalized_params: dict = field(default_factory=dict)

    # Record
    wins: int = 0
    losses: int = 0

    # Additional raw stats for derivation
    blk_pct: float = 0.0
    stl_rate: float = 0.0
    opp_3p_pct: float = 0.0
    ft_pct: float = 0.0
    fta_fga: float = 0.0
    three_pa_fga: float = 0.0
    three_p_pct: float = 0.0
    opp_fg_pct_rim: float = 0.0
    two_pt_pct: float = 0.0   # 2PT field goal %
    two_pt_rate: float = 0.0  # 2PT attempt rate (% of shots that are 2s)
    proximity: float = 0.0    # kept as raw stat, not weighted

    # For Cinderella detection
    adj_em_rank: int = 0
    is_cinderella: bool = False

    # Killshots (EvanMiya proxy for scoring runs)
    killshots_per_game: float = 0.0
    killshots_conceded: float = 0.0

    # NEW: NET rating, injury impact, PPG, best player stats
    net_rating: int = 0
    injury_rank: int = 0
    roster_rank: int = 0
    ppg: float = 0.0
    opp_ppg: float = 0.0
    barthag: float = 0.0
    best_player_above_avg_pts: float = 0.0
    best_player_above_avg_reb: float = 0.0
    best_player_above_avg_ast: float = 0.0
    star_player_win_pct: float = 0.0

    # TeamRankings momentum data
    tr_rating: float = 0.0
    tr_last_rank: int = 0
    tr_hi_rank: int = 0
    tr_lo_rank: int = 0
    sos_last_rank: int = 0
    consistency_rating: float = 0.0
    neutral_rating: float = 0.0

    # CWP: star player 17+ at halftime -> win%
    cwp_star_17_half_win_pct: float = 0.5

    # P&R proxy metrics (from EvanMiya player BPR)
    big_man_offense: float = 0.0
    rim_defense_bpr: float = 0.0

    # Tournament box-score derived (upstream blended)
    tourney_efg: float = 0.0
    tourney_orb_pct: float = 0.0
    tourney_paint_pct: float = 0.0
    tourney_ast_rate: float = 0.0
    tourney_tov_pct: float = 0.0
    tourney_bench_pct: float = 0.0
    tourney_games: int = 0
    traj_fg_pct: float = 0.0
    traj_ast: float = 0.0
    traj_tov: float = 0.0
    traj_paint: float = 0.0
    traj_bench: float = 0.0
    n_accelerating: int = 0


@dataclass
class Matchup:
    team_a: Team
    team_b: Team
    round_name: str = "R64"
    region: str = ""

    win_prob_a_1a: float = 0.5
    win_prob_a_1b: float = 0.5
    win_prob_a_ensemble: float = 0.5
    volatility: float = 0.0
    confidence: float = 0.0
    h2h_season_edge: float = 0.0

    pros_a: list = field(default_factory=list)
    cons_a: list = field(default_factory=list)
    pros_b: list = field(default_factory=list)
    cons_b: list = field(default_factory=list)


@dataclass
class SimulationResult:
    n_simulations: int = 0
    champion_counts: dict = field(default_factory=dict)
    final_four_counts: dict = field(default_factory=dict)
    elite_eight_counts: dict = field(default_factory=dict)
    sweet_sixteen_counts: dict = field(default_factory=dict)
    round_of_32_counts: dict = field(default_factory=dict)

    def championship_odds(self) -> dict:
        return {
            team: count / self.n_simulations
            for team, count in sorted(
                self.champion_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def advancement_odds(self, round_name: str) -> dict:
        counts = {
            "Final Four": self.final_four_counts,
            "Elite Eight": self.elite_eight_counts,
            "Sweet Sixteen": self.sweet_sixteen_counts,
            "Round of 32": self.round_of_32_counts,
            "Champion": self.champion_counts,
        }.get(round_name, {})
        return {
            team: count / self.n_simulations
            for team, count in sorted(
                counts.items(), key=lambda x: x[1], reverse=True
            )
        }
