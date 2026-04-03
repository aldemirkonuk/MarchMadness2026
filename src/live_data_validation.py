"""Validation and normalization helpers for live tournament data."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


SLIM_RESULT_FILES = (
    ("day1_thursday.csv", "R64"),
    ("day2_friday.csv", "R64"),
    ("r32_saturday.csv", "R32"),
    ("r32_sunday.csv", "R32"),
    ("s16_thursday.csv", "S16"),
    ("s16_friday.csv", "S16"),
    ("e8_saturday.csv", "E8"),
    ("e8_sunday.csv", "E8"),
)

TEAM_CANONICAL_MAP = {
    "UConn": "Connecticut",
    "UCONN": "Connecticut",
    "UMBC": "Howard",
}


@dataclass
class ValidationIssue:
    severity: str
    source: str
    round_name: str
    teams: str
    message: str


@dataclass
class SlimResult:
    round_name: str
    team_a: str
    team_b: str
    score_a: int
    score_b: int
    winner: str


def canonical_team_name(name: str) -> str:
    name = (name or "").strip()
    return TEAM_CANONICAL_MAP.get(name, name)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _sorted_key(round_name: str, team_a: str, team_b: str) -> Tuple[str, str, str]:
    a = canonical_team_name(team_a)
    b = canonical_team_name(team_b)
    return (round_name.strip(),) + tuple(sorted((a, b)))


def load_slim_results(base_dir: str = None) -> Dict[Tuple[str, str, str], SlimResult]:
    """Load slim result files into a normalized matchup lookup."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    live_dir = os.path.join(base_dir, "data", "live_results")
    lookup: Dict[Tuple[str, str, str], SlimResult] = {}

    for filename, round_name in SLIM_RESULT_FILES:
        path = os.path.join(live_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r") as handle:
            for row in csv.DictReader(handle):
                team_a = canonical_team_name(row.get("team_a", ""))
                team_b = canonical_team_name(row.get("team_b", ""))
                if not team_a or not team_b:
                    continue
                lookup[_sorted_key(round_name, team_a, team_b)] = SlimResult(
                    round_name=round_name,
                    team_a=team_a,
                    team_b=team_b,
                    score_a=_safe_int(row.get("score_a")),
                    score_b=_safe_int(row.get("score_b")),
                    winner=canonical_team_name(row.get("winner", "")),
                )
    return lookup


def _swap_ab_columns(row: Dict[str, str]) -> Dict[str, str]:
    swapped = dict(row)
    for key in list(row.keys()):
        if not key.endswith("_a"):
            continue
        partner = key[:-2] + "_b"
        if partner not in row:
            continue
        swapped[key] = row.get(partner, "")
        swapped[partner] = row.get(key, "")
    return swapped


def _winner_matches_scores(team_a: str, score_a: int, team_b: str, score_b: int, winner: str) -> bool:
    if score_a == score_b:
        return winner in ("", team_a, team_b)
    expected = team_a if score_a > score_b else team_b
    return winner == expected


def _normalize_box_row(
    row: Dict[str, str],
    slim_lookup: Dict[Tuple[str, str, str], SlimResult],
) -> Tuple[Dict[str, str], List[ValidationIssue]]:
    """Normalize one wide box-score row against slim canonical files."""
    issues: List[ValidationIssue] = []
    normalized = dict(row)

    round_name = (row.get("round", "") or "").strip()
    team_a = canonical_team_name(row.get("team_a", ""))
    team_b = canonical_team_name(row.get("team_b", ""))
    score_a = _safe_int(row.get("score_a"))
    score_b = _safe_int(row.get("score_b"))
    winner = canonical_team_name(row.get("winner", ""))
    teams = f"{team_a} vs {team_b}"

    normalized["team_a"] = team_a
    normalized["team_b"] = team_b
    normalized["winner"] = winner

    if not team_a or not team_b or (score_a == 0 and score_b == 0):
        return normalized, issues

    if not _winner_matches_scores(team_a, score_a, team_b, score_b, winner):
        issues.append(ValidationIssue(
            severity="error",
            source="tournament_box_scores.csv",
            round_name=round_name,
            teams=teams,
            message=f"Winner/score mismatch: {team_a} {score_a}, {team_b} {score_b}, winner={winner}",
        ))

    slim = slim_lookup.get(_sorted_key(round_name, team_a, team_b))
    if not slim:
        return normalized, issues

    if slim.winner and winner and slim.winner != winner:
        issues.append(ValidationIssue(
            severity="error",
            source="tournament_box_scores.csv",
            round_name=round_name,
            teams=teams,
            message=f"Wide/slim winner disagreement: wide={winner}, slim={slim.winner}",
        ))

    wide_scores = sorted((score_a, score_b))
    slim_scores = sorted((slim.score_a, slim.score_b))
    if wide_scores != slim_scores:
        issues.append(ValidationIssue(
            severity="warning",
            source="tournament_box_scores.csv",
            round_name=round_name,
            teams=teams,
            message=f"Wide/slim score disagreement: wide={score_a}-{score_b}, slim={slim.score_a}-{slim.score_b}",
        ))
        return normalized, issues

    same_orientation = team_a == slim.team_a and team_b == slim.team_b
    swapped_orientation = team_a == slim.team_b and team_b == slim.team_a

    if swapped_orientation:
        swapped = _swap_ab_columns(normalized)
        swapped["team_a"] = slim.team_a
        swapped["team_b"] = slim.team_b
        swapped["score_a"] = str(slim.score_a)
        swapped["score_b"] = str(slim.score_b)
        swapped["winner"] = slim.winner
        issues.append(ValidationIssue(
            severity="warning",
            source="tournament_box_scores.csv",
            round_name=round_name,
            teams=f"{slim.team_a} vs {slim.team_b}",
            message="Swapped A/B orientation in memory to match slim result file",
        ))
        return swapped, issues

    if same_orientation:
        normalized["score_a"] = str(slim.score_a)
        normalized["score_b"] = str(slim.score_b)
        normalized["winner"] = slim.winner

    return normalized, issues


def load_validated_tournament_rows(base_dir: str = None) -> Tuple[List[Dict[str, str]], List[ValidationIssue]]:
    """Load tournament box-score rows with in-memory normalization."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    path = os.path.join(base_dir, "data", "live_results", "tournament_box_scores.csv")
    if not os.path.isfile(path):
        return [], []

    slim_lookup = load_slim_results(base_dir)
    rows: List[Dict[str, str]] = []
    issues: List[ValidationIssue] = []
    seen = set()

    with open(path, "r") as handle:
        for row in csv.DictReader(handle):
            normalized, row_issues = _normalize_box_row(row, slim_lookup)
            issues.extend(row_issues)
            round_name = normalized.get("round", "").strip()
            team_a = normalized.get("team_a", "").strip()
            team_b = normalized.get("team_b", "").strip()
            if team_a and team_b:
                key = _sorted_key(round_name, team_a, team_b)
                if key in seen:
                    issues.append(ValidationIssue(
                        severity="warning",
                        source="tournament_box_scores.csv",
                        round_name=round_name,
                        teams=f"{team_a} vs {team_b}",
                        message="Duplicate matchup row after normalization",
                    ))
                else:
                    seen.add(key)
            rows.append(normalized)

    return rows, issues


def validation_report(issues: List[ValidationIssue]) -> str:
    if not issues:
        return "Live data validation: no issues detected."

    n_errors = sum(1 for issue in issues if issue.severity == "error")
    n_warnings = sum(1 for issue in issues if issue.severity == "warning")
    lines = [
        "=" * 80,
        "  LIVE DATA VALIDATION",
        "=" * 80,
        f"  {n_errors} error(s), {n_warnings} warning(s)",
    ]
    for issue in issues[:12]:
        lines.append(
            f"  [{issue.severity.upper()}] {issue.round_name} {issue.teams}: {issue.message}"
        )
    if len(issues) > 12:
        lines.append(f"  ... {len(issues) - 12} more issue(s)")
    return "\n".join(lines)
