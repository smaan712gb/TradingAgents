"""Multi-provider fallback for the most-hammered data fetches.

Wraps each data type with a chain of providers. Order is by preference:

  Stock historical bars: Polygon → IBKR → AlphaVantage → FMP → yfinance
  Sector ETF history:    Polygon → IBKR → yfinance
  Quote / spot:          IBKR → Polygon → yfinance

The fallback triggers on:
  * RateLimitError       — provider's rate limit hit, try next
  * AuthError            — missing/invalid key, try next
  * ProviderError        — generic provider failure, try next
  * Network errors       — connection / timeout

The first provider that returns valid data wins. If none succeed,
returns None (or empty list) — caller decides whether that's fatal.

This is the dispatch path the maintenance loop should use instead of
instantiating PolygonProvider directly. Was: 'Polygon-or-bust'. Now:
'try the cheap-to-call fast providers in order, give up only when all
of them are out'.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from .providers.base import AuthError, ProviderError, RateLimitError

logger = logging.getLogger(__name__)


_FALLBACK_EXCEPTIONS = (RateLimitError, AuthError, ProviderError)


async def get_stock_data_with_fallback(
    symbol: str,
    start_date: date,
    end_date: date,
    *,
    ibkr_provider: Any = None,
) -> Optional[Any]:
    """Try Polygon → IBKR → FMP → yfinance for daily bars.

    Returns a pandas DataFrame with at least 'Close' (and ideally Open / High
    / Low / Volume) columns, or None if every provider fails.

    ``ibkr_provider`` is optional. When supplied (e.g. the live IbkrProvider
    instance the maint loop already has), IBKR slots into the chain after
    Polygon. Otherwise IBKR is skipped.
    """
    # 1. Polygon
    try:
        from .providers.polygon import PolygonProvider
        df = await PolygonProvider().get_stock_data(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
    except _FALLBACK_EXCEPTIONS as e:
        logger.warning("polygon get_stock_data failed for %s: %s — falling back", symbol, e)
    except Exception as e:
        logger.warning("polygon get_stock_data unexpected error for %s: %s", symbol, e)

    # 2. IBKR (if a connected provider is supplied)
    if ibkr_provider is not None:
        try:
            df = await _ibkr_to_dataframe(ibkr_provider, symbol, start_date, end_date)
            if df is not None and len(df) > 0:
                return df
        except _FALLBACK_EXCEPTIONS as e:
            logger.warning("ibkr historical bars failed for %s: %s — falling back", symbol, e)
        except Exception as e:
            logger.warning("ibkr historical bars unexpected error for %s: %s", symbol, e)

    # 3. FMP
    try:
        from .providers.fmp import FmpProvider
        fmp = FmpProvider()
        df = await _fmp_to_dataframe(fmp, symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
    except _FALLBACK_EXCEPTIONS as e:
        logger.warning("fmp historical bars failed for %s: %s — falling back", symbol, e)
    except Exception as e:
        logger.warning("fmp historical bars unexpected error for %s: %s", symbol, e)

    # 4. yfinance — final fallback (no API key required)
    try:
        df = await _yfinance_fallback(symbol, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
    except Exception as e:
        logger.warning("yfinance fallback failed for %s: %s", symbol, e)

    logger.warning("all stock data providers exhausted for %s", symbol)
    return None


# ---------------------------------------------------------------------------
# Provider adapters — normalize responses to a polygon-compatible DataFrame
# (Open, High, Low, Close, Volume columns; date index)
# ---------------------------------------------------------------------------


async def _ibkr_to_dataframe(
    ibkr: Any, symbol: str, start_date: date, end_date: date,
) -> Optional[Any]:
    """Pull IBKR daily bars and shape them into a pandas DataFrame."""
    days = max(1, (end_date - start_date).days + 1)
    duration = f"{days} D" if days <= 365 else f"{(days // 365) + 1} Y"
    bars = await ibkr.get_historical_bars(
        symbol=symbol, duration=duration, bar_size="1 day",
        what_to_show="TRADES", use_rth=True,
    )
    if not bars:
        return None
    import pandas as pd
    rows = []
    for b in bars:
        d = getattr(b, "date", None)
        if d is None:
            continue
        rows.append({
            "Date": d,
            "Open": float(getattr(b, "open", 0) or 0),
            "High": float(getattr(b, "high", 0) or 0),
            "Low":  float(getattr(b, "low", 0) or 0),
            "Close": float(getattr(b, "close", 0) or 0),
            "Volume": float(getattr(b, "volume", 0) or 0),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("Date")
    df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
    return df


async def _fmp_to_dataframe(
    fmp: Any, symbol: str, start_date: date, end_date: date,
) -> Optional[Any]:
    """Pull FMP /stable/historical-price-eod/full and shape into a DataFrame.

    FMP returns rows in reverse chronological order; we flip + slice to the
    requested window so the shape matches Polygon's output.
    """
    body = await fmp._http.get_json(
        "/stable/historical-price-eod/full",
        params={
            "symbol": symbol,
            "from": start_date.isoformat(),
            "to":   end_date.isoformat(),
            "apikey": fmp._api_key,
        },
    )
    rows: list[dict[str, Any]] = []
    if isinstance(body, dict):
        rows = body.get("historical") or []
    elif isinstance(body, list):
        rows = body
    if not rows:
        return None
    import pandas as pd
    out = []
    for r in rows:
        d = r.get("date")
        if not d:
            continue
        out.append({
            "Date": d,
            "Open":   float(r.get("open") or 0),
            "High":   float(r.get("high") or 0),
            "Low":    float(r.get("low") or 0),
            "Close":  float(r.get("close") or 0),
            "Volume": float(r.get("volume") or 0),
        })
    if not out:
        return None
    df = pd.DataFrame(out)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df


async def _yfinance_fallback(
    symbol: str, start_date: date, end_date: date,
) -> Optional[Any]:
    """Free-tier yfinance — last resort. Fast enough for the daily bar
    use case, no API key required."""
    import asyncio
    try:
        import yfinance as yf
    except ImportError:
        return None
    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(
        None,
        lambda: yf.download(
            symbol, start=start_date, end=end_date,
            progress=False, auto_adjust=True,
        ),
    )
    if df is None or df.empty:
        return None
    # yfinance returns multi-index columns when downloading single ticker
    # — flatten to single level.
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    return df
