"""IV percentile + IV-vs-realized premium signal.

Maps to the operator's exhaustion-rule item: "Short term call IV is
extremely elevated" — classic late-cycle euphoria pattern. When OTM
call sellers are pricing in a 60% move on a stock that has only
realized 25% historically, retail euphoria is paying up for upside
exposure and the rally is often topping.

Two complementary measures:

  1. **IV percentile** — today's front-month ATM call IV vs the
     trailing N-day distribution from ``iv_snapshots``. Trips at
     percentile >= 90 (top decile). Requires N >= 30 days of capture
     to be statistically meaningful; before that, returns None for
     the percentile field (signal still fires off measure #2).

  2. **IV-vs-realized premium** — current ATM call IV divided by
     the underlying's 30-day annualized realized volatility. Trips
     when the ratio is >= 2.0 (IV pricing in 2x realized). Zero
     cold-start — works from day one of capture.

The "tripped" boolean is OR of the two: either condition counts as
the 8th exhaustion signal. Both are recorded in the audit row for
visibility.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Tunables.
PERCENTILE_LOOKBACK_DAYS    = 252               # 1 trading year if available
PERCENTILE_MIN_DAYS         = 30                # require at least this many to compute
PERCENTILE_TRIGGER          = 0.90              # top decile = elevated
IV_VS_REALIZED_TRIGGER      = 2.0               # IV >= 2x realized = rich
TRADING_DAYS_PER_YEAR       = 252


@dataclass
class IvSignal:
    iv_today: Optional[float] = None
    realized_30d_ann: Optional[float] = None
    iv_vs_realized: Optional[float] = None
    iv_percentile: Optional[float] = None
    days_of_history: int = 0
    tripped: bool = False
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def compute_realized_vol_annualized(closes: list[float]) -> Optional[float]:
    """30-day annualized realized vol from a close series (close-to-close).

    Caller passes >= 21 daily closes (~30 calendar days). Returns None
    when there's not enough data.
    """
    if not closes or len(closes) < 21:
        return None
    rets = []
    for i in range(1, len(closes)):
        if closes[i - 1] <= 0 or closes[i] <= 0:
            continue
        rets.append(math.log(closes[i] / closes[i - 1]))
    if len(rets) < 20:
        return None
    # Use the last 21 returns
    rets = rets[-21:]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    sigma = math.sqrt(var)
    return sigma * math.sqrt(TRADING_DAYS_PER_YEAR)


async def evaluate_iv_signal(
    session: AsyncSession,
    *,
    symbol: str,
    iv_today: Optional[float],
    closes: Optional[list[float]] = None,
) -> IvSignal:
    """Compute the IV signal for one symbol.

    ``iv_today`` is the front-month ATM call IV (decimal, e.g. 0.45 for 45%).
    ``closes`` is a daily close series for the underlying — used to compute
    realized vol. Both are best-effort; missing data sets the related
    sub-signal to None and the function falls back to the other.
    """
    from api.app.db import IvSnapshot

    out = IvSignal(iv_today=iv_today)

    # Realized vol
    if closes:
        out.realized_30d_ann = compute_realized_vol_annualized(closes)
    if iv_today is not None and out.realized_30d_ann and out.realized_30d_ann > 0:
        out.iv_vs_realized = round(iv_today / out.realized_30d_ann, 2)

    # Percentile lookup over the trailing window
    cutoff = date.today() - timedelta(days=PERCENTILE_LOOKBACK_DAYS)
    rows = (
        await session.execute(
            select(IvSnapshot.atm_call_iv)
            .where(IvSnapshot.symbol == symbol)
            .where(IvSnapshot.date >= cutoff)
        )
    ).all()
    history = [float(r[0]) for r in rows if r[0] is not None]
    out.days_of_history = len(history)

    if iv_today is not None and len(history) >= PERCENTILE_MIN_DAYS:
        below = sum(1 for h in history if h <= iv_today)
        out.iv_percentile = round(below / len(history), 3)

    # Trip decision
    parts: list[str] = []
    if out.iv_percentile is not None and out.iv_percentile >= PERCENTILE_TRIGGER:
        parts.append(
            f"IV percentile {out.iv_percentile*100:.0f}% (>= {PERCENTILE_TRIGGER*100:.0f}% over {len(history)}d)"
        )
    if out.iv_vs_realized is not None and out.iv_vs_realized >= IV_VS_REALIZED_TRIGGER:
        parts.append(
            f"IV/realized {out.iv_vs_realized:.2f}x (IV {iv_today*100:.0f}% vs realized {out.realized_30d_ann*100:.0f}%)"
        )
    out.tripped = bool(parts)

    if parts:
        out.rationale = " · ".join(parts)
    elif iv_today is None:
        out.rationale = f"{symbol}: IV not available"
    elif out.iv_percentile is None:
        ratio_str = f"{out.iv_vs_realized:.2f}x" if out.iv_vs_realized else "n/a"
        out.rationale = (
            f"{symbol}: IV {iv_today*100:.0f}%, IV/realized={ratio_str}, "
            f"{len(history)}/{PERCENTILE_MIN_DAYS}d history (percentile not yet available)"
        )
    else:
        out.rationale = (
            f"{symbol}: IV {iv_today*100:.0f}% at {out.iv_percentile*100:.0f}th percentile "
            f"(below {PERCENTILE_TRIGGER*100:.0f}% trigger)"
        )

    out.detail = {
        "percentile_trigger": PERCENTILE_TRIGGER,
        "iv_vs_realized_trigger": IV_VS_REALIZED_TRIGGER,
        "lookback_days": PERCENTILE_LOOKBACK_DAYS,
        "min_days_for_percentile": PERCENTILE_MIN_DAYS,
    }
    return out


async def capture_iv_snapshot(
    session: AsyncSession,
    *,
    symbol: str, ibkr: Any, snap_date: Optional[date] = None,
) -> bool:
    """Capture today's front-month ATM call IV for ``symbol`` into iv_snapshots.

    Idempotent: skips if a snapshot for this (symbol, date) already exists.
    Returns True if a row was written, False if skipped or fetch failed.
    """
    from api.app.db import IvSnapshot

    when = snap_date or date.today()

    # Idempotency: skip if already captured today
    existing = (
        await session.execute(
            select(IvSnapshot.id)
            .where(IvSnapshot.symbol == symbol)
            .where(IvSnapshot.date == datetime(when.year, when.month, when.day, tzinfo=timezone.utc))
        )
    ).first()
    if existing:
        return False

    try:
        result = await ibkr.get_atm_call_iv(symbol=symbol)
    except Exception as e:
        logger.warning("IV capture: fetch failed for %s: %s", symbol, e)
        return False
    iv = result.get("iv")
    if iv is None or iv <= 0:
        return False

    row = IvSnapshot(
        symbol=symbol,
        date=datetime(when.year, when.month, when.day, tzinfo=timezone.utc),
        atm_call_iv=float(iv),
        dte_used=result.get("dte"),
        strike_used=result.get("strike"),
        spot_at_capture=result.get("spot"),
    )
    session.add(row)
    return True
