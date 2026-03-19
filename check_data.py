"""Preflight data check — run this before the main pipeline to catch missing/empty files.

Usage:
    python check_data.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(BASE_DIR, "archive")
ARCHIVE3 = os.path.join(BASE_DIR, "archive-3")
DATA_DIR  = os.path.join(BASE_DIR, "data")

# ── Required files ────────────────────────────────────────────────────────────
# Each entry: (directory_key, filename, description, required)
REQUIRED = [
    # archive-3 — primary model inputs
    (ARCHIVE3, "KenPom Barttorvik.csv",          "KenPom + Barttorvik merged stats (MAIN)", True),
    (ARCHIVE3, "KenPom Preseason.csv",            "KenPom preseason ratings",                True),
    (ARCHIVE3, "EvanMiya.csv",                    "EvanMiya player ratings",                 True),
    (ARCHIVE3, "Tournament Matchups.csv",          "Historical tournament matchups",           True),
    (ARCHIVE3, "Tournament Locations.csv",         "Tournament location data",                True),
    (ARCHIVE3, "Coach Results.csv",               "Coach tournament performance",             True),
    (ARCHIVE3, "Resumes.csv",                     "Team resume / SOS data",                  True),
    (ARCHIVE3, "Shooting Splits.csv",             "Shot location splits",                    True),
    (ARCHIVE3, "TeamRankings.csv",                "TeamRankings metrics",                    True),
    (ARCHIVE3, "Teamsheet Ranks.csv",             "Teamsheet rankings",                      True),
    (ARCHIVE3, "Z Rating Teams.csv",              "Z-rating team data",                      True),
    (ARCHIVE3, "Z Rating Cumulative.csv",         "Z-rating cumulative",                     True),
    (ARCHIVE3, "538 Ratings.csv",                 "538 team ratings",                        True),
    (ARCHIVE3, "Heat Check Ratings.csv",          "Heat Check ratings",                      True),
    (ARCHIVE3, "Heat Check Tournament Index.csv", "Heat Check tournament index",             True),
    (ARCHIVE3, "RPPF Ratings.csv",               "RPPF power ratings",                      True),
    (ARCHIVE3, "AP Poll Data.csv",               "AP Poll data",                            True),
    (ARCHIVE3, "Public Picks.csv",               "Public bracket pick percentages",          True),
    (ARCHIVE3, "Conference Results.csv",          "Conference tournament results",            True),
    (ARCHIVE3, "Barttorvik Away-Neutral.csv",     "Barttorvik away/neutral splits",          True),
    (ARCHIVE3, "HalfScoring.csv",                "Half-time scoring data",                   True),

    # archive — supplemental (KenPom Height, Coaches, etc.)
    (ARCHIVE, "REF _ Current NCAAM Coaches (2026).csv", "Coach roster", True),

    # data — matchups + picks
    (DATA_DIR, "matchups.csv", "2026 bracket matchup pairs", True),
]

# ── Game logs (recency engine) ─────────────────────────────────────────────────
GAME_LOG_DIR = os.path.join(ARCHIVE3, "game-logs")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  NCAA 2026 DATA PREFLIGHT CHECK")
print("=" * 70)

ok = 0
warn = 0
fail = 0
issues = []

# Check each required file
for dirpath, fname, desc, required in REQUIRED:
    path = os.path.join(dirpath, fname)
    if not os.path.exists(path):
        status = "MISSING"
        fail += 1
        issues.append(f"MISSING  — {fname}  ({desc})")
        print(f"  ❌ MISSING  — {fname}")
    else:
        size = os.path.getsize(path)
        if size == 0:
            status = "EMPTY"
            fail += 1
            issues.append(f"EMPTY    — {fname}  ({size} bytes)  [{path}]")
            print(f"  ❌ EMPTY    — {fname}  ({desc})")
        elif size < 200:
            status = "TINY"
            warn += 1
            issues.append(f"TINY     — {fname}  ({size} bytes, may be LFS pointer)")
            print(f"  ⚠️  TINY    — {fname}  ({size} bytes — may be Git LFS pointer)")
        else:
            ok += 1
            kb = size / 1024
            print(f"  ✅ OK      — {fname}  ({kb:.0f} KB)")

# Check game-logs directory
print()
if not os.path.isdir(GAME_LOG_DIR):
    print(f"  ❌ MISSING  — archive-3/game-logs/ (recency engine will be skipped)")
    fail += 1
    issues.append("MISSING  — archive-3/game-logs/ directory")
else:
    logs = [f for f in os.listdir(GAME_LOG_DIR) if f.endswith(".csv")]
    empty_logs = [f for f in logs if os.path.getsize(os.path.join(GAME_LOG_DIR, f)) == 0]
    if empty_logs:
        print(f"  ⚠️  GAME-LOGS — {len(logs)} total, {len(empty_logs)} empty ({', '.join(empty_logs[:3])}...)")
        warn += len(empty_logs)
    else:
        print(f"  ✅ GAME-LOGS — {len(logs)} files OK  (recency engine ready)")
        ok += 1

print()
print("=" * 70)
print(f"  RESULT:  {ok} OK  |  {warn} warnings  |  {fail} failures")
print("=" * 70)

if issues:
    print("\n  ISSUES TO FIX:")
    for iss in issues:
        print(f"    • {iss}")
    print()

    if any("EMPTY" in i or "MISSING" in i for i in issues):
        print("  ACTION REQUIRED:")
        print("  ─────────────────")
        print("  Empty/missing files cannot be auto-generated by the pipeline.")
        print("  For each affected file:")
        print("    1. Re-export the data from KenPom, Barttorvik, or your data source.")
        print("    2. Save it to the path shown above.")
        print("    3. Re-run this check before running the main pipeline.")
        print()
        print("  If files show as TINY (<200 bytes), you may have a Git LFS issue.")
        print("  Run:  git lfs pull")
        print("  Then re-run:  python check_data.py")
        print()

if fail > 0:
    sys.exit(1)
else:
    print("\n  ✅  All required files present. Safe to run:  python -m src.main\n")
