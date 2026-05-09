"""Earnings calendar lookups via FMP.

FMP exposes ``/stable/earnings-calendar?symbol={ticker}`` returning a list
of past + future earnings dates. We cache results for 4 hours since the
calendar doesn't shift intra-day.

Used by the maintenance loop to:
  * Buy back short calls 2 sessions before a print (vol crush asymmetry)
  * Re-sell the day after, capturing fresh post-earnings IV
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


_CACHE: dict[str, tuple[float, Optional[date]]] = {}
_CACHE_TTL_SEC = 4 * 60 * 60     # 4h — earnings dates don't move intra-day


async def get_earnings_dates_for(symbols) -> dict:
    """Batch lookup: {symbol: date} for the next future earnings of each
    symbol in the input set. Symbols with no scheduled earnings are
    omitted from the result.

    Used by the SEC-filings watcher to classify 8-K severity (filings
    near earnings dates are likely earnings-related, not material events).
    """
    out: dict = {}
    for sym in symbols:
        try:
            d = await _next_earnings_cached(sym.upper() if isinstance(sym, str) else sym)
        except Exception:
            continue
        if d is not None:
            out[sym.upper() if isinstance(sym, str) else sym] = d
    return out


async def _next_earnings_cached(sym: str) -> Optional[date]:
    """Cached single-symbol lookup with the same TTL as days_to_earnings."""
    now_ts = time.time()
    cached = _CACHE.get(sym)
    if cached and now_ts - cached[0] < _CACHE_TTL_SEC:
        return cached[1]
    next_dt = await _fetch_next_earnings(sym)
    _CACHE[sym] = (now_ts, next_dt)
    return next_dt


async def days_to_earnings(symbol: str) -> Optional[int]:
    """Return days until the next scheduled earnings, or None if unavailable."""
    sym = symbol.upper()
    now_ts = time.time()
    cached = _CACHE.get(sym)
    if cached and now_ts - cached[0] < _CACHE_TTL_SEC:
        next_dt = cached[1]
    else:
        next_dt = await _fetch_next_earnings(sym)
        _CACHE[sym] = (now_ts, next_dt)
    if next_dt is None:
        return None
    return (next_dt - date.today()).days


async def earnings_hedge_due(symbol: str, *, hedge_window_sessions: int = 2) -> bool:
    """True if next earnings is within the hedge window (default 2 sessions)."""
    days = await days_to_earnings(symbol)
    return days is not None and 0 <= days <= hedge_window_sessions


async def _fetch_next_earnings(symbol: str) -> Optional[date]:
    """Pull the soonest future earnings date from FMP. None if unavailable."""
    try:
        from tradingagents.dataflows.providers.fmp import FmpProvider
    except Exception:
        return None
    try:
        p = FmpProvider()
    except Exception as e:
        logger.warning("earnings: FMP provider init failed: %s", e)
        return None
    try:
        # FMP /stable/ earnings-calendar returns rows with `date` field.
        body = await p._http.get_json(
            "/stable/earnings-calendar",
            params={"symbol": symbol, "apikey": p._api_key},
        )
        if not isinstance(body, list):
            return None
        today = date.today()
        future = []
        for row in body:
            d = row.get("date")
            if not d:
                continue
            try:
                parsed = date.fromisoformat(str(d)[:10])
            except (TypeError, ValueError):
                continue
            if parsed >= today:
                future.append(parsed)
        return min(future) if future else None
    except Exception as e:
        logger.warning("earnings: fetch failed for %s: %s", symbol, e)
        return None
