"""Analyst-rating pressure signal.

Two distinct signals come out of the FMP analyst data:

  1. **Upgrade-after-major-run** — when a stock has run hard (>= 25% in
     60 days) and analysts are still upgrading, that's classic late-cycle
     "chase" behaviour. Maps to the operator's exhaustion-rule item:
     "analyst upgrades come after a major run". Trips the 7th
     momentum-exhaustion signal.
  2. **Downgrade acceleration** — three or more downgrades in 30 days
     is meaningful supply pressure. Feeds into the thesis-break detector
     when paired with theme weakness.

Optional v2 (not implemented): consensus price-target trend — if the
consensus has *fallen* in the last month while spot is up, that's
also a "chase" pattern.

Returns ``GradePressure`` with all signals exposed so the maint loop
can choose what to use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tunables.
RUN_PCT_60D                 = 0.25     # 25% gain over 60 days = "major run"
RECENT_UPGRADE_DAYS         = 7        # upgrade within last 7 days
DOWNGRADE_STREAK_DAYS       = 30
DOWNGRADE_STREAK_COUNT      = 3        # 3 downgrades in 30d = acceleration


@dataclass
class GradePressure:
    upgrades_30d: int = 0
    downgrades_30d: int = 0
    upgrades_recent: list[dict[str, Any]] = field(default_factory=list)
    downgrades_recent: list[dict[str, Any]] = field(default_factory=list)
    upgrade_after_run: bool = False           # 7th exhaustion signal
    downgrade_acceleration: bool = False
    pct_60d: Optional[float] = None
    rationale: str = ""


def _parse_grade_date(s: Any) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


_BUY_TIER = {"buy", "strong buy", "outperform", "overweight", "positive",
             "accumulate", "add", "long-term buy"}
_HOLD_TIER = {"hold", "neutral", "market perform", "equal-weight",
              "in-line", "sector perform"}
_SELL_TIER = {"sell", "strong sell", "underperform", "underweight",
              "negative", "reduce"}


def _tier(grade: Optional[str]) -> str:
    g = (grade or "").lower().strip()
    if g in _BUY_TIER:
        return "buy"
    if g in _HOLD_TIER:
        return "hold"
    if g in _SELL_TIER:
        return "sell"
    return "unknown"


def _direction(prev: Optional[str], new: Optional[str]) -> str:
    """Return 'upgrade' / 'downgrade' / 'side' based on tier transitions.

    New coverage initiations are scored as upgrades when the new grade
    is buy-tier (analyst chose to be positive), downgrades for sell-tier,
    and 'side' for hold. Lateral moves within a tier are 'side'.
    """
    p, n = _tier(prev), _tier(new)
    severity = {"sell": -1, "hold": 0, "buy": 1, "unknown": 0}
    if severity[n] > severity[p]:
        return "upgrade"
    if severity[n] < severity[p]:
        return "downgrade"
    return "side"


async def evaluate_analyst_pressure(
    *,
    symbol: str,
    grade_changes: list[dict[str, Any]],
    pct_60d: Optional[float] = None,
) -> GradePressure:
    """Score upgrades/downgrades over the last 30 days; classify
    upgrade-after-run + downgrade-acceleration."""
    today = date.today()
    cutoff_30d = today - timedelta(days=30)
    cutoff_recent = today - timedelta(days=RECENT_UPGRADE_DAYS)

    upgrades_30d: list[dict[str, Any]] = []
    downgrades_30d: list[dict[str, Any]] = []
    upgrades_recent: list[dict[str, Any]] = []

    for r in grade_changes:
        d = _parse_grade_date(r.get("date") or r.get("publishedDate"))
        if d is None or d < cutoff_30d:
            continue
        prev_grade = r.get("previousGrade")
        new_grade = r.get("newGrade") or r.get("gradeChange")
        direction = _direction(prev_grade, new_grade)
        row = {
            "date": d.isoformat(),
            "firm": r.get("gradingCompany") or r.get("publisher"),
            "previous": prev_grade, "new": new_grade,
        }
        if direction == "upgrade":
            upgrades_30d.append(row)
            if d >= cutoff_recent:
                upgrades_recent.append(row)
        elif direction == "downgrade":
            downgrades_30d.append(row)

    upgrade_after_run = (
        bool(upgrades_recent)
        and pct_60d is not None and pct_60d >= RUN_PCT_60D
    )
    downgrade_acceleration = (
        len(downgrades_30d) >= DOWNGRADE_STREAK_COUNT
    )

    parts = []
    if upgrade_after_run:
        firms = ", ".join(u.get("firm") or "?" for u in upgrades_recent[:3])
        parts.append(
            f"upgrade after run: {pct_60d*100:.0f}% / 60d, "
            f"{len(upgrades_recent)} upgrade(s) in 7d ({firms})"
        )
    if downgrade_acceleration:
        firms = ", ".join(d.get("firm") or "?" for d in downgrades_30d[:3])
        parts.append(
            f"downgrade acceleration: {len(downgrades_30d)} in 30d ({firms})"
        )

    return GradePressure(
        upgrades_30d=len(upgrades_30d),
        downgrades_30d=len(downgrades_30d),
        upgrades_recent=upgrades_recent,
        downgrades_recent=downgrades_30d,
        upgrade_after_run=upgrade_after_run,
        downgrade_acceleration=downgrade_acceleration,
        pct_60d=pct_60d,
        rationale=" · ".join(parts) if parts else "no notable analyst pressure",
    )
