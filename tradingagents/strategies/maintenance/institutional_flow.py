"""13F institutional ownership flow.

Quarterly snapshot of how institutions are positioned in a name. Three
signals flow out of the FMP /institutional-ownership endpoint:

  1. **Accumulation** — investorsHoldingChange and ownershipPercentChange
     both positive, meaningful magnitude. Smart money quietly building.
     Bull-thesis enhancer.
  2. **Distribution** — investorsHoldingChange and ownershipPercentChange
     both negative; reducedPositionsChange dominant. Smart money
     trimming. Adds weight to the exit-pressure score.
  3. **Crowded** — ownershipPercent already > CROWDED_OWNERSHIP_PCT
     (default 80%). Limited marginal buyer left; subsequent rallies
     more vulnerable to selling pressure.

13F data lags ~45 days from quarter-end (filing deadline), so the
latest-quarter signal is a *trailing* read. The 4-hour FMP cache is
plenty — quarterly data doesn't move intra-day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tunables.
ACCUMULATION_MIN_INVESTORS_PCT  = 0.02      # +2% growth in 13F holders
ACCUMULATION_MIN_OWNERSHIP_PCT  = 0.5       # +0.5pp ownership share
DISTRIBUTION_MIN_INVESTORS_PCT  = -0.02     # -2% holders
DISTRIBUTION_MIN_OWNERSHIP_PCT  = -0.5      # -0.5pp ownership
CROWDED_OWNERSHIP_PCT           = 80.0      # >80% institutionally held = crowded


@dataclass
class InstitutionalFlow:
    period: Optional[str] = None              # 'YYYY-MM-DD' quarter end
    investors_holding: int = 0
    investors_change: int = 0
    investors_change_pct: Optional[float] = None
    ownership_pct: Optional[float] = None
    ownership_pct_change: Optional[float] = None
    new_positions: int = 0
    closed_positions: int = 0
    increased_positions: int = 0
    reduced_positions: int = 0
    flow_label: str = "neutral"               # accumulating | distributing | crowded | neutral
    crowded: bool = False
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _latest_completed_quarter(today: Optional[date] = None) -> tuple[int, int]:
    """Return (year, quarter) for the latest 13F quarter that's likely
    filed already. 13F deadline is 45 days after quarter end, so we
    require the quarter to have ended at least 50 days ago."""
    today = today or date.today()
    # Calendar quarter ending dates
    candidates = [
        date(today.year, 12, 31), date(today.year, 9, 30),
        date(today.year, 6, 30),  date(today.year, 3, 31),
        date(today.year - 1, 12, 31),
    ]
    for cand in candidates:
        if (today - cand).days >= 50:
            q = (cand.month - 1) // 3 + 1
            return (cand.year, q)
    return (today.year - 1, 4)


def evaluate_institutional_flow(
    *,
    symbol: str,
    summary: dict[str, Any],
) -> InstitutionalFlow:
    """Classify the latest 13F snapshot into accumulating / distributing /
    crowded / neutral. ``summary`` is the FMP response shape."""
    if not summary:
        return InstitutionalFlow(rationale=f"{symbol}: no 13F data")

    investors = int(summary.get("investorsHolding") or 0)
    investors_change = int(summary.get("investorsHoldingChange") or 0)
    last_investors = int(summary.get("lastInvestorsHolding") or 0)
    investors_change_pct = (
        investors_change / last_investors if last_investors > 0 else None
    )
    ownership_pct = summary.get("ownershipPercent")
    ownership_pct_change = summary.get("ownershipPercentChange")
    new_positions = int(summary.get("newPositions") or 0)
    closed_positions = int(summary.get("closedPositions") or 0)
    increased = int(summary.get("increasedPositions") or 0)
    reduced = int(summary.get("reducedPositions") or 0)

    crowded = (
        ownership_pct is not None and float(ownership_pct) >= CROWDED_OWNERSHIP_PCT
    )

    flow_label = "neutral"
    rationale_parts = []
    if (investors_change_pct is not None and ownership_pct_change is not None
            and investors_change_pct >= ACCUMULATION_MIN_INVESTORS_PCT
            and ownership_pct_change >= ACCUMULATION_MIN_OWNERSHIP_PCT):
        flow_label = "accumulating"
        rationale_parts.append(
            f"{symbol}: {investors_change:+,} institutions Q/Q "
            f"({investors_change_pct*100:+.1f}%), "
            f"ownership {ownership_pct_change:+.2f}pp"
        )
    elif (investors_change_pct is not None and ownership_pct_change is not None
            and investors_change_pct <= DISTRIBUTION_MIN_INVESTORS_PCT
            and ownership_pct_change <= DISTRIBUTION_MIN_OWNERSHIP_PCT):
        flow_label = "distributing"
        rationale_parts.append(
            f"{symbol}: {investors_change:+,} institutions Q/Q "
            f"({investors_change_pct*100:+.1f}%), "
            f"ownership {ownership_pct_change:+.2f}pp"
        )
    if crowded:
        flow_label = "crowded" if flow_label == "neutral" else f"{flow_label}_crowded"
        rationale_parts.append(
            f"crowded ownership {ownership_pct:.1f}% "
            f"(threshold {CROWDED_OWNERSHIP_PCT:.0f}%)"
        )
    if not rationale_parts:
        rationale_parts.append(
            f"{symbol}: holders {investors_change:+,} ({investors:,}), "
            f"ownership {ownership_pct or 0:.1f}%"
        )

    return InstitutionalFlow(
        period=str(summary.get("date") or ""),
        investors_holding=investors,
        investors_change=investors_change,
        investors_change_pct=investors_change_pct,
        ownership_pct=float(ownership_pct) if ownership_pct is not None else None,
        ownership_pct_change=float(ownership_pct_change) if ownership_pct_change is not None else None,
        new_positions=new_positions,
        closed_positions=closed_positions,
        increased_positions=increased,
        reduced_positions=reduced,
        flow_label=flow_label,
        crowded=crowded,
        rationale=" · ".join(rationale_parts),
        detail={
            "accumulation_thresholds": {
                "investors_pct": ACCUMULATION_MIN_INVESTORS_PCT,
                "ownership_pct": ACCUMULATION_MIN_OWNERSHIP_PCT,
            },
            "distribution_thresholds": {
                "investors_pct": DISTRIBUTION_MIN_INVESTORS_PCT,
                "ownership_pct": DISTRIBUTION_MIN_OWNERSHIP_PCT,
            },
        },
    )
