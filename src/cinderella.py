"""Undervalued mid-major detector (Cinderella identifier).

Flags teams meeting ALL criteria:
- AdjEM rank in top 20 nationally
- Tournament seed >= 6
- Average roster experience >= 2.0 years
- Clutch Factor above tournament median
"""

import numpy as np
from typing import List
from src.models import Team


def detect_cinderellas(teams: List[Team]) -> List[Team]:
    """Identify and flag Cinderella candidates. Returns flagged teams."""
    clutch_values = [t.clutch_factor for t in teams]
    median_clutch = np.median(clutch_values)

    cinderellas = []
    for t in teams:
        t.is_cinderella = (
            t.adj_em_rank <= 20 and
            t.seed >= 6 and
            t.exp >= 2.0 and
            t.clutch_factor > median_clutch
        )
        if t.is_cinderella:
            cinderellas.append(t)

    return cinderellas


def cinderella_report(teams: List[Team]) -> str:
    """Generate a human-readable Cinderella report."""
    cinderellas = detect_cinderellas(teams)

    if not cinderellas:
        # Relax criteria: top 25 AdjEM, seed >= 5, exp >= 1.5
        for t in teams:
            if (t.adj_em_rank <= 25 and t.seed >= 5 and t.exp >= 1.5):
                t.is_cinderella = True
                cinderellas.append(t)

    lines = [
        "=" * 60,
        "  CINDERELLA / UPSET CANDIDATES",
        "=" * 60,
    ]

    if not cinderellas:
        lines.append("  No strong Cinderella candidates detected.")
        lines.append("  (All top-20 AdjEM teams are seeded appropriately)")
    else:
        for t in cinderellas:
            lines.append(
                f"  {'*'*3} {t.name} ({t.seed}-seed, {t.region}) {'*'*3}"
            )
            lines.append(
                f"      AdjEM Rank: {t.adj_em_rank} | "
                f"Exp: {t.exp:.1f} | "
                f"Clutch: {t.clutch_factor:.3f}"
            )
            lines.append(
                f"      Why: Top-{t.adj_em_rank} team seeded as {t.seed}. "
                f"{'Experienced roster.' if t.exp >= 2.5 else ''}"
            )
            lines.append("")

    # Also flag "near misses" -- teams just outside Cinderella criteria
    lines.append("\n  Upset Watch (strong teams with beatable matchups):")
    near_misses = [
        t for t in teams
        if t.adj_em_rank <= 30 and t.seed >= 5 and not t.is_cinderella
    ]
    for t in sorted(near_misses, key=lambda x: x.adj_em_rank)[:5]:
        lines.append(
            f"    {t.name} ({t.seed}-seed) -- AdjEM rank {t.adj_em_rank}"
        )

    return "\n".join(lines)
