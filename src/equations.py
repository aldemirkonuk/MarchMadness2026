"""All 58 equations: core metrics, niche, WTH modifiers, ML, and Bayesian.

Each function takes raw stat inputs and returns the computed metric.
"""

import numpy as np
from typing import Optional


# ── Equation 1: True Shooting Percentage ──────────────────────────────────────
def true_shooting_pct(pts: float, fga: float, fta: float) -> float:
    denom = 2 * (fga + 0.44 * fta)
    return pts / denom if denom > 0 else 0.0


# ── Equation 2: Adjusted Offensive Efficiency ─────────────────────────────────
def adj_offensive_eff(pts: float, poss: float,
                      avg_def_eff: float, opp_def_eff: float) -> float:
    if poss == 0 or opp_def_eff == 0:
        return 0.0
    return (pts / poss) * 100 * (avg_def_eff / opp_def_eff)


# ── Equation 3: Adjusted Defensive Efficiency ────────────────────────────────
def adj_defensive_eff(pts_allowed: float, poss: float,
                      avg_off_eff: float, opp_off_eff: float) -> float:
    if poss == 0 or opp_off_eff == 0:
        return 0.0
    return (pts_allowed / poss) * 100 * (avg_off_eff / opp_off_eff)


# ── Equation 4: Possessions Estimate ─────────────────────────────────────────
def possessions(fga: float, orb: float, to: float, fta: float) -> float:
    return fga - orb + to + 0.475 * fta


# ── Equation 5: Offensive Rebound Rate ───────────────────────────────────────
def off_rebound_rate(orb: float, opp_drb: float) -> float:
    total = orb + opp_drb
    return orb / total if total > 0 else 0.0


# ── Equation 6: Clutch Factor (revised) ──────────────────────────────────────
def clutch_factor(close_wins: float, close_games: float,
                  clutch_net_rating: float,
                  clutch_net_min: float = -15.0,
                  clutch_net_max: float = 15.0) -> float:
    close_win_pct = close_wins / close_games if close_games > 0 else 0.5
    normed_net = (clutch_net_rating - clutch_net_min) / (clutch_net_max - clutch_net_min)
    normed_net = np.clip(normed_net, 0.0, 1.0)
    return 0.6 * close_win_pct + 0.4 * normed_net


# ── Equation 7: Three-Point Reliability Index ────────────────────────────────
def three_point_reliability(three_p_pct: float, three_pa_fga: float) -> float:
    return three_p_pct * three_pa_fga


# ── Equation 8: Assist Rate ──────────────────────────────────────────────────
def assist_rate(ast: float, fgm: float) -> float:
    return ast / fgm if fgm > 0 else 0.0


# ── Equation 9: Turnover Rate ────────────────────────────────────────────────
def turnover_rate(to: float, fga: float, fta: float) -> float:
    denom = fga + 0.475 * fta + to
    return to / denom if denom > 0 else 0.0


# ── Equation 10: Free Throw Reliability ──────────────────────────────────────
def free_throw_reliability(ft_pct: float, fta_fga: float) -> float:
    return ft_pct * fta_fga


# ── Equation 11: Star Power Index (revised) ──────────────────────────────────
def star_power_index(player_bpm: float, player_usage: float,
                     league_avg_bpm: float = 0.0) -> float:
    if league_avg_bpm == 0:
        league_avg_bpm = 1.0
    return (player_bpm * player_usage) / max(abs(league_avg_bpm), 0.01)


# ── Equation 12: Experience Factor ───────────────────────────────────────────
def experience_factor(player_minutes: list, player_years: list) -> float:
    total_min = sum(player_minutes)
    if total_min == 0:
        return 2.0
    return sum(m * y for m, y in zip(player_minutes, player_years)) / total_min


# ── Equation 13: Strength of Schedule ────────────────────────────────────────
def strength_of_schedule(opponent_adj_ems: list) -> float:
    return np.mean(opponent_adj_ems) if opponent_adj_ems else 0.0


# ── Equation 14: Coaching Factor (Bayesian smoothed) ─────────────────────────
def coaching_factor(tourney_wins: int, tourney_games: int) -> float:
    return (tourney_wins + 2) / (tourney_games + 4)


# ── Equation 15: Momentum ────────────────────────────────────────────────────
def momentum(last10_wins: int, conf_tourney_score: float) -> float:
    return 0.6 * (last10_wins / 10.0) + 0.4 * conf_tourney_score


# ── Equation 16: Legacy Factor (revised) ─────────────────────────────────────
def legacy_factor(actual_rounds: list, expected_rounds: list) -> float:
    if not actual_rounds:
        return 0.0
    diffs = [a - e for a, e in zip(actual_rounds, expected_rounds)]
    return np.mean(diffs)


# ── Equation 17: Proximity Advantage ─────────────────────────────────────────
def proximity_advantage(distance_miles: float) -> float:
    return 1.0 / (1.0 + distance_miles / 500.0)


# ── Equation 18: Defensive Versatility Index ─────────────────────────────────
def defensive_versatility(blk_norm: float, stl_norm: float,
                          opp_3p_defense_norm: float) -> float:
    return 0.3 * blk_norm + 0.3 * stl_norm + 0.4 * opp_3p_defense_norm


# ── Equation 19: Bench Depth Score ───────────────────────────────────────────
def bench_depth_score(bench_pts: float, total_pts: float) -> float:
    return bench_pts / total_pts if total_pts > 0 else 0.0


# ── Equation 20: Effective Field Goal Percentage ─────────────────────────────
def effective_fg_pct(fgm: float, three_pm: float, fga: float) -> float:
    return (fgm + 0.5 * three_pm) / fga if fga > 0 else 0.0


# ── Equation 21: Defensive Rebound Rate ──────────────────────────────────────
def def_rebound_rate(drb: float, opp_orb: float) -> float:
    total = drb + opp_orb
    return drb / total if total > 0 else 0.0


# ── Equation 22: Seed Score ──────────────────────────────────────────────────
def seed_score(seed: int) -> float:
    return 1.0 / seed if seed > 0 else 0.0


# ── Equation 23: Top-50 Performance ──────────────────────────────────────────
def top25_performance(wins_vs_top50: int, games_vs_top50: int) -> float:
    return wins_vs_top50 / games_vs_top50 if games_vs_top50 > 0 else 0.0


# ── Equation 24: Turnover Forcing Rate (defensive) ──────────────────────────
def opp_turnover_rate(opp_turnovers: float, opp_possessions: float) -> float:
    return opp_turnovers / opp_possessions if opp_possessions > 0 else 0.0


# ── Equation 25: Rim Protection Index ────────────────────────────────────────
def rim_protection_index(blk_pct_norm: float,
                         opp_fg_at_rim_inverted_norm: float) -> float:
    return blk_pct_norm + opp_fg_at_rim_inverted_norm


# ── Equation 26: 1st Half Scoring Dominance ──────────────────────────────────
def q1_scoring(team_1h_avg: float, opp_1h_avg: float) -> float:
    return team_1h_avg - opp_1h_avg


# ── Equation 27: 2nd Half / 3rd Quarter Surge ────────────────────────────────
def q3_scoring(team_2h_avg: float, opp_2h_avg: float) -> float:
    return team_2h_avg - opp_2h_avg


# ── Equation 28: Three-Point Variance (Chaos Index) ─────────────────────────
def chaos_index(three_pa_fga: float, three_p_std: float) -> float:
    return three_pa_fga * three_p_std


# ── Equation 29: Sightline Penalty ──────────────────────────────────────────
def sightline_penalty(hist_3p_drop_large_arena: float) -> float:
    return hist_3p_drop_large_arena


def three_pri_adjusted(three_pri: float, sightline_pen: float) -> float:
    return three_pri * (1.0 - sightline_pen)


# ── Equation 30: Altitude Impact ────────────────────────────────────────────
def altitude_impact(arena_elevation_ft: float, three_pa_fga: float) -> float:
    return (arena_elevation_ft / 5280.0) * three_pa_fga


# ── Equation 31: Referee Impact ─────────────────────────────────────────────
def ref_impact(team_foul_rate: float, ref_avg_fouls: float) -> float:
    return team_foul_rate * ref_avg_fouls


# ── Equation 32: Sentiment Score ────────────────────────────────────────────
def sentiment_score(positive: float, negative: float) -> float:
    total = positive + negative
    if total == 0:
        return 0.0
    return (positive - negative) / total


# ── Equation 33: Jersey Color Aggression ────────────────────────────────────
def jersey_color_aggression(foul_rate_dark: float,
                            foul_rate_light: float) -> float:
    return foul_rate_dark - foul_rate_light


def foul_projection_adjusted(base_foul_rate: float, jca_modifier: float,
                              is_wearing_dark: bool) -> float:
    return base_foul_rate * (1.0 + jca_modifier * int(is_wearing_dark))


# ── Equation 34: Max Scoring Run Potential ───────────────────────────────────
def max_scoring_run(avg_max_run: float, avg_run_allowed: float) -> float:
    return avg_max_run - avg_run_allowed


# ── Equation 35: Blowout Resilience ─────────────────────────────────────────
def blowout_resilience(comebacks_from_15: int,
                       games_trailing_15: int) -> float:
    return comebacks_from_15 / games_trailing_15 if games_trailing_15 > 0 else 0.5


# ── Equation 36: Foul Trouble Impact ────────────────────────────────────────
def foul_trouble_impact(pm_star_foul_trouble: float,
                        min_star_foul_trouble: float,
                        pm_star_on_court: float,
                        min_star_on_court: float) -> float:
    rate_foul = pm_star_foul_trouble / min_star_foul_trouble if min_star_foul_trouble > 0 else 0.0
    rate_normal = pm_star_on_court / min_star_on_court if min_star_on_court > 0 else 0.0
    return rate_foul - rate_normal


# ── Equation 37: Conditional Win Probabilities ──────────────────────────────
def conditional_win_prob(wins_when_condition: int,
                         games_when_condition: int) -> float:
    return wins_when_condition / games_when_condition if games_when_condition > 0 else 0.5


def fragility_score(overall_win_pct: float, cwp_values: list) -> float:
    adverse = [overall_win_pct - cwp for cwp in cwp_values if cwp < overall_win_pct]
    return np.mean(adverse) if adverse else 0.0


def versatility_score(cwp_values: list) -> float:
    return sum(1 for cwp in cwp_values if cwp > 0.5) / len(cwp_values) if cwp_values else 0.0


def march_readiness(cwp_trailing_half: float, cwp_leading_half: float,
                    cwp_cold_3: float, cwp_star_cold_half: float,
                    cwp_outrebounded: float, cwp_opp_hot: float) -> float:
    return (0.25 * cwp_trailing_half +
            0.20 * cwp_leading_half +
            0.20 * cwp_cold_3 +
            0.15 * cwp_star_cold_half +
            0.10 * cwp_outrebounded +
            0.10 * cwp_opp_hot)


# ── Equation 38: Team Strength Composite (Phase 1A) ─────────────────────────
def team_strength_composite(normalized_params: dict,
                            weights: dict) -> float:
    score = 0.0
    for param, weight in weights.items():
        score += weight * normalized_params.get(param, 0.0)
    return score


# ── Equation 39: Composite Score Differential ────────────────────────────────
def composite_score_differential(norm_a: dict, norm_b: dict,
                                 weights: dict) -> float:
    z = 0.0
    for param, weight in weights.items():
        z += weight * (norm_a.get(param, 0.0) - norm_b.get(param, 0.0))
    return z


# ── Equation 40: Win Probability (Logistic) ─────────────────────────────────
def win_probability_logistic(z: float, k: float = 6.0) -> float:
    return 1.0 / (1.0 + np.exp(-k * z))


# ── Equation 41: Upset Volatility Modifier ──────────────────────────────────
def upset_volatility(p_base: float, chaos_a: float, chaos_b: float,
                     pace_diff_norm: float = 0.0,
                     wth_adjustment: float = 0.0) -> float:
    v = (0.15 * (chaos_a + chaos_b) / 2.0 +
         0.05 * pace_diff_norm +
         wth_adjustment)
    v = np.clip(v, 0.0, 0.4)
    return p_base * (1.0 - v) + 0.5 * v


# ── Equation 42: Single Game Simulation ──────────────────────────────────────
def simulate_game(p_a_wins: float, rng: np.random.Generator = None) -> bool:
    if rng is None:
        rng = np.random.default_rng()
    return rng.random() < p_a_wins


# ── Equations 43-45: Monte Carlo outputs (in monte_carlo.py) ────────────────
# Championship probability, round advancement, Cinderella detection
# are computed in the simulation loop, not as standalone equations.


# ── Equation 46-49: XGBoost pipeline (in xgboost_model.py) ─────────────────
def build_ml_features(team_a_params: dict, team_b_params: dict,
                      param_keys: list) -> list:
    features = []
    for key in param_keys:
        a_val = team_a_params.get(key, 0.0)
        b_val = team_b_params.get(key, 0.0)
        features.extend([a_val, b_val, a_val - b_val])
    return features


# ── Equation 50: SHAP (in xgboost_model.py) ─────────────────────────────────


# ── Equation 51: Ensemble Win Probability ────────────────────────────────────
def ensemble_probability(p_1a: float, p_1b: float,
                         lam: float = 0.5) -> float:
    return lam * p_1a + (1.0 - lam) * p_1b


# ── Equation 52: Confidence Score ────────────────────────────────────────────
def confidence_score(p_1a: float, p_1b: float) -> float:
    return 1.0 - abs(p_1a - p_1b)


# ── Equations 53: Ensemble Monte Carlo (in monte_carlo.py) ──────────────────


# ── Equation 54: Bayesian Prior ──────────────────────────────────────────────
def bayesian_prior(prior_1a: float, prior_1b: float,
                   lam: float = 0.5) -> float:
    return lam * prior_1a + (1.0 - lam) * prior_1b


# ── Equation 55: Likelihood from Observed Game ──────────────────────────────
def game_likelihood(observed_margin: float, predicted_margin: float,
                    sigma: float = 11.0) -> float:
    from scipy.stats import norm
    return norm.pdf(observed_margin, loc=predicted_margin, scale=sigma)


# ── Equation 56: Posterior Update ────────────────────────────────────────────
def posterior_update(prior_strength: float, observed_performance: float,
                     alpha: float = 0.85) -> float:
    return alpha * prior_strength + (1.0 - alpha) * observed_performance


ROUND_ALPHA = {
    "R64": 0.85,
    "R32": 0.75,
    "S16": 0.65,
    "E8": 0.55,
    "F4": 0.45,
}


# ── Equation 57: XGBoost Re-training (in xgboost_model.py) ─────────────────


# ── Equation 58: Re-simulation (in bayesian.py) ─────────────────────────────


# ── Cinderella Detection ────────────────────────────────────────────────────
def is_cinderella(adj_em_rank: int, seed: int, exp: float,
                  clutch: float, median_clutch: float) -> bool:
    return (adj_em_rank <= 20 and
            seed >= 6 and
            exp >= 2.0 and
            clutch > median_clutch)


# ── Rebounding Margin ──────────────────────────────────────────────────────
def rebounding_margin(team_total_reb: float, opp_total_reb: float,
                      games: int) -> float:
    return (team_total_reb - opp_total_reb) / games if games > 0 else 0.0
