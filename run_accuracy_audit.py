"""Comprehensive tournament accuracy audit.

Loads all predictions and actual results, computes accuracy by round,
confidence tier, and Brier score. Classifies misses as signal vs noise.
"""

import csv
import os
import math

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(BASE, "data", "live_results")
RESULTS = os.path.join(BASE, "data", "results")


def _load_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _canonical(name):
    return name.strip().replace("UConn", "Connecticut").replace("UMBC", "Howard")


def main():
    print("=" * 80)
    print("  TOURNAMENT ACCURACY AUDIT — ALL 60 GAMES")
    print("=" * 80)

    # ── Load actual results ──────────────────────────────────────────────
    actual = {}

    for fname, rnd in [
        ("day1_thursday.csv", "R64"), ("day2_friday.csv", "R64"),
        ("r32_saturday.csv", "R32"), ("r32_sunday.csv", "R32"),
        ("s16_thursday.csv", "S16"), ("s16_friday.csv", "S16"),
        ("e8_saturday.csv", "E8"), ("e8_sunday.csv", "E8"),
    ]:
        for row in _load_csv(os.path.join(LIVE, fname)):
            a = _canonical(row.get("team_a", ""))
            b = _canonical(row.get("team_b", ""))
            winner = _canonical(row.get("winner", ""))
            score_a = int(row.get("score_a", 0) or 0)
            score_b = int(row.get("score_b", 0) or 0)
            margin = abs(score_a - score_b)
            key = tuple(sorted([a, b]))
            actual[key] = {"winner": winner, "round": rnd, "team_a": a, "team_b": b,
                           "margin": margin, "score_a": score_a, "score_b": score_b}

    # ── Load predictions ─────────────────────────────────────────────────
    predictions = {}

    # R64 predictions
    for row in _load_csv(os.path.join(RESULTS, "matchup_predictions.csv")):
        a = _canonical(row.get("team_a", ""))
        b = _canonical(row.get("team_b", ""))
        prob_a = float(row.get("prob_a_ensemble", row.get("prob_a_1a", 0.5)))
        pred_winner = a if prob_a > 0.5 else b
        conf = max(prob_a, 1 - prob_a)
        key = tuple(sorted([a, b]))
        predictions[key] = {"pred_winner": pred_winner, "prob": conf, "round": "R64",
                            "prob_a": prob_a, "team_a": a, "team_b": b}

    # R32 predictions
    for row in _load_csv(os.path.join(LIVE, "r32_full_predictions.csv")):
        a = _canonical(row.get("team_a", ""))
        b = _canonical(row.get("team_b", ""))
        prob_a = float(row.get("final_prob_a", 0.5))
        pred_winner = a if prob_a > 0.5 else b
        conf = max(prob_a, 1 - prob_a)
        key = tuple(sorted([a, b]))
        predictions[key] = {"pred_winner": pred_winner, "prob": conf, "round": "R32",
                            "prob_a": prob_a, "team_a": a, "team_b": b}

    # S16 predictions
    for row in _load_csv(os.path.join(LIVE, "s16_predictions.csv")):
        a = _canonical(row.get("team_a", ""))
        b = _canonical(row.get("team_b", ""))
        prob_a = float(row.get("final_prob_a", 0.5))
        pred_winner = a if prob_a > 0.5 else b
        conf = max(prob_a, 1 - prob_a)
        key = tuple(sorted([a, b]))
        predictions[key] = {"pred_winner": pred_winner, "prob": conf, "round": "S16",
                            "prob_a": prob_a, "team_a": a, "team_b": b}

    # E8 predictions
    for row in _load_csv(os.path.join(LIVE, "e8_predictions.csv")):
        a = _canonical(row.get("team_a", ""))
        b = _canonical(row.get("team_b", ""))
        prob_a = float(row.get("scenario_prob_a", row.get("ensemble_prob_a", 0.5)))
        pred_winner = row.get("pick", a if prob_a > 0.5 else b).strip()
        pred_winner = _canonical(pred_winner)
        conf = max(prob_a, 1 - prob_a)
        key = tuple(sorted([a, b]))
        predictions[key] = {"pred_winner": pred_winner, "prob": conf, "round": "E8",
                            "prob_a": prob_a, "team_a": a, "team_b": b}

    # ── Match predictions to results ─────────────────────────────────────
    round_stats = {}
    tier_stats = {"LOCK": [0, 0], "LEAN": [0, 0], "TOSS-UP": [0, 0]}
    misses = []
    brier_sum = 0.0
    brier_n = 0
    total_correct = 0
    total_games = 0

    for key, act in sorted(actual.items(), key=lambda x: {"R64": 0, "R32": 1, "S16": 2, "E8": 3}.get(x[1]["round"], 9)):
        rnd = act["round"]
        pred = predictions.get(key)

        if not pred:
            continue

        correct = (pred["pred_winner"] == act["winner"])
        total_correct += int(correct)
        total_games += 1

        if rnd not in round_stats:
            round_stats[rnd] = [0, 0]
        round_stats[rnd][0] += int(correct)
        round_stats[rnd][1] += 1

        tier = "LOCK" if pred["prob"] > 0.80 else "LEAN" if pred["prob"] > 0.60 else "TOSS-UP"
        tier_stats[tier][0] += int(correct)
        tier_stats[tier][1] += 1

        prob_winner = pred["prob"] if correct else (1 - pred["prob"])
        brier_sum += (1 - prob_winner) ** 2
        brier_n += 1

        if not correct:
            margin = act["margin"]
            noise_label = "NOISE" if margin <= 5 else "SIGNAL"
            misses.append({
                "round": rnd, "actual_winner": act["winner"],
                "pred_winner": pred["pred_winner"], "prob": pred["prob"],
                "margin": margin, "label": noise_label,
                "teams": f"{act['team_a']} vs {act['team_b']}",
            })

    # ── Print results ────────────────────────────────────────────────────
    print(f"\n  OVERALL: {total_correct}/{total_games} = {total_correct/max(1,total_games):.1%}")
    print(f"  Brier Score: {brier_sum/max(1,brier_n):.4f} (lower = better)\n")

    print("  BY ROUND:")
    for rnd in ["R64", "R32", "S16", "E8"]:
        if rnd in round_stats:
            c, t = round_stats[rnd]
            print(f"    {rnd:6s}: {c}/{t} = {c/max(1,t):.1%}")

    print("\n  BY CONFIDENCE TIER:")
    for tier in ["LOCK", "LEAN", "TOSS-UP"]:
        c, t = tier_stats[tier]
        if t > 0:
            print(f"    {tier:8s}: {c}/{t} = {c/max(1,t):.1%}")

    print(f"\n  MISSES ({len(misses)} total):")
    print(f"  {'Round':<6} {'Matchup':<35} {'Predicted':<18} {'Actual':<18} {'Prob':>6} {'Margin':>6} {'Type'}")
    print(f"  {'-'*100}")
    for m in misses:
        print(f"  {m['round']:<6} {m['teams']:<35} {m['pred_winner']:<18} {m['actual_winner']:<18} {m['prob']:>5.1%} {m['margin']:>5}  {m['label']}")

    noise_count = sum(1 for m in misses if m["label"] == "NOISE")
    signal_count = sum(1 for m in misses if m["label"] == "SIGNAL")
    print(f"\n  Classification: {noise_count} NOISE (margin <= 5pts), {signal_count} SIGNAL (margin > 5pts)")
    print(f"  If NOISE misses are 'acceptable variance': adjusted accuracy = {(total_correct + noise_count)}/{total_games} = {(total_correct + noise_count)/max(1,total_games):.1%}")

    # ── Championship odds vs actual F4 ───────────────────────────────────
    print(f"\n{'='*80}")
    print("  CHAMPIONSHIP ODDS vs ACTUAL FINAL FOUR")
    print(f"{'='*80}")
    champ_odds = {}
    for row in _load_csv(os.path.join(RESULTS, "championship_odds.csv")):
        team = row.get("team", "").strip()
        pct = float(row.get("championship_pct", 0))
        champ_odds[team] = pct

    actual_f4 = ["Arizona", "Illinois", "Connecticut", "Michigan"]
    for team in actual_f4:
        odds = champ_odds.get(team, 0)
        print(f"  {team:<20} Pre-tourney champ odds: {odds:.2f}%")

    print(f"\n  Combined pre-tourney probability of this exact F4: "
          f"{champ_odds.get('Arizona',0)*champ_odds.get('Illinois',0)*champ_odds.get('Connecticut',0)*champ_odds.get('Michigan',0)/1e6:.6f}%")

    eliminated_favorites = [
        ("Duke", champ_odds.get("Duke", 0), "Lost E8 to UConn on buzzer-beater"),
        ("Florida", champ_odds.get("Florida", 0), "Lost R32 to Iowa on last-second 3"),
        ("Gonzaga", champ_odds.get("Gonzaga", 0), "Lost R32 to 11-seed Texas"),
        ("Michigan State", champ_odds.get("Michigan State", 0), "Lost S16 to UConn"),
        ("Iowa State", champ_odds.get("Iowa State", 0), "Lost S16 to Tennessee (J.Jefferson OUT)"),
    ]
    print(f"\n  ELIMINATED FAVORITES:")
    for team, odds, reason in eliminated_favorites:
        print(f"    {team:<20} {odds:.2f}% -> {reason}")


if __name__ == "__main__":
    main()
