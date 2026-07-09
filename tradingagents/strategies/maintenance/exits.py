"""Exit logic for stocks and PMCC positions.

Stocks:
  * Avoid signal — latest run scored this ticker as Avoid → close
  * ATR trailing stop — close if (entry_avg - current) > 2.5 × ATR_30d

PMCC:
  * LEAP delta < 0.65 — thesis broken
  * Operator manual close (via /api/admin/positions/exit)

This module returns ``ExitDecision``; the maintenance loop applies the
auto-gate, then submits via walking-limit (combo for PMCC, single-leg
for stocks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


ATR_STOP_MULTIPLE = 2.5


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str = ""
    exit_kind: Literal["avoid_signal", "atr_stop", "thesis_break", "operator", "none"] = "none"
    detail: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Stock exits
# ---------------------------------------------------------------------------


async def maybe_exit_stock(
    *,
    symbol: str,
    current_price: float,
    avg_price: float,
    latest_decision: Optional[str],     # "Buy" / "Hold" / "Avoid" / None
    atr_30d: Optional[float],
) -> ExitDecision:
    """Decide if a stock position should be closed."""
    # Avoid signal from latest theme run
    if latest_decision == "Avoid":
        return ExitDecision(
            should_exit=True,
            reason=f"latest scorecard says Avoid for {symbol}",
            exit_kind="avoid_signal",
        )

    # ATR-based trailing stop. Only fires when underwater enough that
    # the stop math is meaningful — protects against random intraday drift.
    if atr_30d and atr_30d > 0 and avg_price > 0:
        loss = avg_price - current_price
        threshold = ATR_STOP_MULTIPLE * atr_30d
        if loss > threshold:
            return ExitDecision(
                should_exit=True,
                reason=f"ATR stop hit: down ${loss:.2f} > {ATR_STOP_MULTIPLE}× ATR_30d (${threshold:.2f})",
                exit_kind="atr_stop",
                detail={"loss": loss, "atr": atr_30d, "threshold": threshold},
            )

    return ExitDecision(should_exit=False, reason="hold")


# ---------------------------------------------------------------------------
# PMCC close
# ---------------------------------------------------------------------------


async def maybe_close_pmcc(
    *,
    symbol: str,
    leap_delta: Optional[float],
    latest_decision: Optional[str],
    leap_dte: Optional[int],
) -> ExitDecision:
    """Close a PMCC if the thesis broke or the latest run says Avoid."""
    if latest_decision == "Avoid":
        return ExitDecision(
            should_exit=True,
            reason=f"latest scorecard says Avoid",
            exit_kind="avoid_signal",
        )

    if leap_delta is not None and abs(leap_delta) < 0.65:
        # Delta decay is a PRICE-drawdown proxy (a deep-ITM LEAP only loses this
        # much delta when the underlying has fallen materially). On a high-beta
        # book this can be a normal down move, not a thesis break — so this exit
        # is FLAG-ONLY (operator confirms); it deliberately does NOT auto-dump.
        return ExitDecision(
            should_exit=True,
            reason=(f"LEAP delta {abs(leap_delta):.2f} < 0.65 — deep-ITM cushion "
                    f"eroded; review (price-driven, confirm before closing)"),
            exit_kind="delta_decay",
        )

    # If LEAP is in last 90 days and we haven't rolled forward (operator
    # didn't accept), close to avoid the time-decay cliff. This is a STRUCTURAL
    # (calendar) reason, not a drawdown, so it may auto-execute.
    if leap_dte is not None and leap_dte < 90:
        return ExitDecision(
            should_exit=True,
            reason=f"LEAP DTE {leap_dte} < 90 — exit before time-decay cliff",
            exit_kind="dte_cliff",
        )

    return ExitDecision(should_exit=False, reason="hold")
