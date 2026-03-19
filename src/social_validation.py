"""Social Media Verification Layer — post-model comparison with analyst upset picks.

This module runs AFTER the model produces predictions. It does NOT modify
model probabilities. It compares model outputs to publicly circulating
analyst/social media upset narratives and flags three signal types:

  CONFIRMED   — Model independently agrees with the analyst upset pick.
                Both data and narrative point the same direction. High confidence.
  CONTRARIAN  — Model disagrees with the analyst pick. Either the model found
                something the crowd missed, or vice versa. Worth investigating.
  BLIND SPOT  — Model shows the favored team winning comfortably, but analysts
                cite a specific factor (injury, style mismatch, etc.) that the
                model may not fully capture. Flag for manual review.

The real value: "Our model independently flagged the same upsets the analysts
are calling — here's why the data agrees." OR: "Here's where we see something
the crowd doesn't."
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.utils import canonical_name
from src.injury_model import TeamInjuryProfile, _get_round_penalty

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SocialSignal:
    """One analyst/social media upset prediction."""
    upset_team: str          # the underdog being picked
    favored_team: str        # the favorite being picked against
    favored_seed: int
    upset_seed: int
    source: str              # "Analyst consensus", "Twitter/X", etc.
    narrative: str           # the story being told
    key_factor: str          # primary driver: star_injury, style_mismatch, etc.


@dataclass
class VerificationResult:
    """Result of comparing one social signal to model output."""
    signal: SocialSignal
    model_upset_prob: float          # model's probability for the upset
    model_favored_prob: float        # model's probability for the favorite
    tier: str                        # CONFIRMED, CONTRARIAN, BLIND_SPOT
    model_injury_penalty: float      # AdjEM injury penalty on favored team
    model_factors: List[str]         # model-side reasons supporting/refuting

    @property
    def spread_shift(self) -> str:
        """Human-readable description of how close the model thinks it is."""
        if self.model_upset_prob >= 0.50:
            return f"Model picks the upset ({self.model_upset_prob:.0%})"
        elif self.model_upset_prob >= 0.40:
            return f"Near coin-flip ({self.model_upset_prob:.0%})"
        elif self.model_upset_prob >= 0.30:
            return f"Competitive underdog ({self.model_upset_prob:.0%})"
        else:
            return f"Long shot ({self.model_upset_prob:.0%})"


def load_social_signals() -> List[SocialSignal]:
    """Load social signals from CSV."""
    path = os.path.join(BASE_DIR, "data", "social_signals.csv")
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    signals = []
    for _, row in df.iterrows():
        signals.append(SocialSignal(
            upset_team=str(row["upset_team"]).strip(),
            favored_team=str(row["favored_team"]).strip(),
            favored_seed=int(row.get("favored_seed", 0)),
            upset_seed=int(row.get("upset_seed", 0)),
            source=str(row.get("source", "Unknown")).strip(),
            narrative=str(row.get("narrative", "")).strip(),
            key_factor=str(row.get("key_factor", "")).strip(),
        ))
    return signals


def _find_matchup_prob(signal: SocialSignal,
                       matchup_results: Dict[str, float]) -> Tuple[float, float]:
    """Find the model probability for this matchup.

    matchup_results should be a dict mapping
      "TeamA vs TeamB" -> probability TeamA wins
    or similar structure from the model output.
    """
    upset_cn = canonical_name(signal.upset_team)
    fav_cn = canonical_name(signal.favored_team)

    for key, prob in matchup_results.items():
        key_cn = key.lower().replace(".", "")
        if fav_cn.lower().replace(".", "") in key_cn:
            if upset_cn.lower().replace(".", "") in key_cn:
                # Found the matchup — figure out which direction
                parts = key.split(" vs ")
                if len(parts) == 2:
                    first = canonical_name(parts[0].strip())
                    if first.lower().replace(".", "") == fav_cn.lower().replace(".", ""):
                        return (1.0 - prob, prob)  # upset prob, favored prob
                    else:
                        return (prob, 1.0 - prob)
                # Fallback: assume prob is for the first team listed
                return (1.0 - prob, prob)

    return (-1.0, -1.0)  # not found


def verify_signals(signals: List[SocialSignal],
                   matchup_results: Dict[str, float],
                   injury_profiles: Optional[Dict[str, TeamInjuryProfile]] = None,
                   teams_dict: Optional[Dict[str, object]] = None,
                   round_name: str = "R64") -> List[VerificationResult]:
    """Compare social signals to model predictions.

    Parameters
    ----------
    signals : list of SocialSignal
    matchup_results : dict mapping "Team A vs Team B" -> prob Team A wins
    injury_profiles : optional injury profile dict for checking penalties
    teams_dict : optional {canonical_name: Team} for checking model params
    round_name : which round to check (default R64)
    """
    results = []

    for signal in signals:
        upset_prob, fav_prob = _find_matchup_prob(signal, matchup_results)

        # Check injury penalty on the favored team
        inj_penalty = 0.0
        if injury_profiles:
            fav_cn = canonical_name(signal.favored_team)
            inj_penalty = _get_round_penalty(fav_cn, injury_profiles, round_name)

        # Determine model factors that support the upset narrative
        factors = []

        if inj_penalty > 0.05:
            adj_em_loss = inj_penalty * 5.5
            factors.append(f"Injury penalty: -{adj_em_loss:.1f} AdjEM on {signal.favored_team}")

            # Check for collapse risk
            if injury_profiles:
                fav_cn = canonical_name(signal.favored_team)
                profile = injury_profiles.get(fav_cn) or injury_profiles.get(signal.favored_team)
                if profile:
                    for ip in profile.injured_players:
                        if ip.collapse_risk:
                            factors.append(
                                f"COLLAPSE RISK: {ip.player} leads {ip.categories_led} "
                                f"categories (x{ip.multi_cat_amplifier:.2f})"
                            )
                        elif ip.is_star_carrier:
                            factors.append(
                                f"Star carrier out: {ip.player} "
                                f"(BPR share {ip.bpr_share:.0%})"
                            )

        # Key factor matching
        if signal.key_factor == "star_injury" and inj_penalty > 0.05:
            factors.append(f"Social narrative matches: star injury confirmed in model")
        elif signal.key_factor == "coaching_history" and teams_dict:
            fav_cn = canonical_name(signal.favored_team)
            team = teams_dict.get(fav_cn)
            if team and hasattr(team, 'ctf') and team.ctf < 0.5:
                factors.append(f"Coaching factor: {signal.favored_team} CTF={team.ctf:.2f} (below avg)")
        elif signal.key_factor == "defensive_weakness" and teams_dict:
            fav_cn = canonical_name(signal.favored_team)
            team = teams_dict.get(fav_cn)
            if team and hasattr(team, 'kadj_d'):
                factors.append(f"Defense: {signal.favored_team} AdjD={team.kadj_d:.1f}")

        # Classify into tiers
        if upset_prob < 0:
            tier = "NOT_IN_BRACKET"
        elif upset_prob >= 0.45:
            tier = "CONFIRMED"
            factors.insert(0, f"Model gives upset {upset_prob:.0%} — near or above coin-flip")
        elif upset_prob >= 0.30:
            if len(factors) >= 2:
                tier = "CONFIRMED"
                factors.insert(0, f"Model sees competitive upset ({upset_prob:.0%}) with multiple supporting factors")
            else:
                tier = "CONTRARIAN"
                factors.insert(0, f"Model gives upset only {upset_prob:.0%} despite social buzz")
        else:
            if len(factors) >= 1 and inj_penalty > 0.1:
                tier = "BLIND_SPOT"
                factors.insert(0, f"Model has upset at {upset_prob:.0%} but injury data warrants review")
            else:
                tier = "CONTRARIAN"
                factors.insert(0, f"Model disagrees: upset at {upset_prob:.0%}")

        results.append(VerificationResult(
            signal=signal,
            model_upset_prob=upset_prob,
            model_favored_prob=fav_prob,
            tier=tier,
            model_injury_penalty=inj_penalty * 5.5,  # in AdjEM units
            model_factors=factors,
        ))

    return results


def social_verification_report(results: List[VerificationResult]) -> str:
    """Generate the social media verification report."""
    lines = [
        "=" * 70,
        "  SOCIAL MEDIA VERIFICATION LAYER",
        "  Model vs. Analyst Upset Predictions",
        "=" * 70,
        "",
    ]

    tier_order = ["CONFIRMED", "BLIND_SPOT", "CONTRARIAN", "NOT_IN_BRACKET"]
    tier_labels = {
        "CONFIRMED": "CONFIRMED — Model agrees with analyst upset pick",
        "BLIND_SPOT": "BLIND SPOT — Model may underweight a key factor",
        "CONTRARIAN": "CONTRARIAN — Model disagrees with analyst pick",
        "NOT_IN_BRACKET": "NOT IN BRACKET — Matchup not found in model",
    }

    for tier in tier_order:
        tier_results = [r for r in results if r.tier == tier]
        if not tier_results:
            continue

        lines.append(f"  --- {tier_labels[tier]} ---")
        lines.append("")

        for r in tier_results:
            s = r.signal
            lines.append(f"  ({s.upset_seed}) {s.upset_team} over ({s.favored_seed}) {s.favored_team}")
            lines.append(f"  {r.spread_shift}")

            if r.model_injury_penalty > 0.1:
                lines.append(f"  Injury penalty on {s.favored_team}: -{r.model_injury_penalty:.1f} AdjEM")

            lines.append(f"  Analyst narrative: {s.narrative}")
            lines.append(f"  Key factor: {s.key_factor}")

            if r.model_factors:
                lines.append(f"  Model factors:")
                for f in r.model_factors:
                    lines.append(f"    • {f}")

            lines.append("")

    # Summary
    confirmed = len([r for r in results if r.tier == "CONFIRMED"])
    blind = len([r for r in results if r.tier == "BLIND_SPOT"])
    contrarian = len([r for r in results if r.tier == "CONTRARIAN"])

    lines.append("-" * 70)
    lines.append(f"  SUMMARY: {confirmed} confirmed | {blind} blind spots | {contrarian} contrarian")
    lines.append("-" * 70)

    if confirmed > 0:
        lines.append("")
        lines.append("  TAKEAWAY: Model independently validates analyst upset picks where noted.")
        lines.append("  Convergence between data model and expert narrative = high-confidence signal.")

    if blind > 0:
        lines.append("")
        lines.append("  REVIEW: Blind spots deserve manual investigation — the model may not")
        lines.append("  fully capture the factor the analysts are citing.")

    return "\n".join(lines)
