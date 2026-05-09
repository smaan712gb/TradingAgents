"""Polygon.io provider.

Implements `CoreStockProvider` and `OptionsChainProvider`.

Endpoints used (Stocks Advanced + Options Advanced tier):

* `/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}`
    OHLCV bars; we parameterize daily by default but expose `interval`.
* `/v3/snapshot/options/{symbol}`
    Real-time chain snapshot, includes greeks and open interest.
* `/v3/reference/options/contracts`
    Per-expiry contract listing (used to enumerate strikes).

The agents see a *summary string* — the upstream graph's analyst tools
return strings that go into the LLM context. We keep that contract for
the existing `get_stock_data` / `get_indicators` surface but also expose
typed `OptionContract` objects for the new options-flow analyst.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import pandas as pd

from .base import (
    AuthError,
    OptionContract,
    OptionRight,
    ProviderError,
)
from .cache import cached
from .http import AsyncHttpClient

logger = logging.getLogger(__name__)


_INTERVAL_MAP = {
    # tradingagents convention -> (multiplier, timespan)
    "1m":  (1, "minute"),
    "5m":  (5, "minute"),
    "15m": (15, "minute"),
    "1h":  (1, "hour"),
    "1d":  (1, "day"),
    "1w":  (1, "week"),
}


class PolygonProvider:
    name = "polygon"

    def __init__(self, api_key: Optional[str] = None) -> None:
        api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise AuthError("polygon")
        self._http = AsyncHttpClient(
            provider_name="polygon",
            base_url="https://api.polygon.io",
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # CoreStockProvider
    # ------------------------------------------------------------------

    @cached(ttl_s=60, namespace="polygon.aggs")
    async def get_stock_data(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval not in _INTERVAL_MAP:
            raise ValueError(f"unsupported interval {interval!r}; expected one of {list(_INTERVAL_MAP)}")
        multiplier, timespan = _INTERVAL_MAP[interval]
        path = (
            f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        body = await self._http.get_json(path, params={"adjusted": "true", "limit": 50000})
        results = body.get("results") or []
        if not results:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame(results).rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "ts"}
        )
        df["Date"] = pd.to_datetime(df["ts"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        return df

    # ------------------------------------------------------------------
    # OptionsChainProvider
    # ------------------------------------------------------------------

    @cached(ttl_s=300, namespace="polygon.options.chain")
    async def get_options_chain(self, symbol: str, expiry: date) -> list[OptionContract]:
        # Polygon paginates contracts; pull all pages for the expiry, then
        # snapshot to enrich with greeks / OI / quotes.
        contracts = await self._list_contracts(symbol, expiry)
        snapshot = await self.get_options_snapshot(symbol)
        snap_by_ticker = {c["details"]["ticker"]: c for c in snapshot.get("results", [])}

        out: list[OptionContract] = []
        for c in contracts:
            t = c["ticker"]
            snap = snap_by_ticker.get(t, {})
            quote = snap.get("last_quote") or {}
            trade = snap.get("last_trade") or {}
            greeks = snap.get("greeks") or {}
            day = snap.get("day") or {}
            out.append(
                OptionContract(
                    ticker=symbol,
                    expiry=expiry,
                    strike=Decimal(str(c["strike_price"])),
                    right=OptionRight.CALL if c["contract_type"] == "call" else OptionRight.PUT,
                    bid=_to_decimal(quote.get("bid")),
                    ask=_to_decimal(quote.get("ask")),
                    last=_to_decimal(trade.get("price")),
                    open_interest=int(snap.get("open_interest") or 0),
                    volume=int(day.get("volume") or 0),
                    iv=_to_float(snap.get("implied_volatility")),
                    delta=_to_float(greeks.get("delta")),
                    gamma=_to_float(greeks.get("gamma")),
                    vega=_to_float(greeks.get("vega")),
                    theta=_to_float(greeks.get("theta")),
                )
            )
        return out

    @cached(ttl_s=30, namespace="polygon.options.snapshot")
    async def get_options_snapshot(self, symbol: str) -> dict[str, Any]:
        return await self._http.get_json(
            f"/v3/snapshot/options/{symbol}", params={"limit": 250}
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    async def _list_contracts(self, symbol: str, expiry: date) -> list[dict[str, Any]]:
        params = {
            "underlying_ticker": symbol,
            "expiration_date": expiry.isoformat(),
            "limit": 1000,
        }
        results: list[dict[str, Any]] = []
        path = "/v3/reference/options/contracts"
        while True:
            body = await self._http.get_json(path, params=params)
            page = body.get("results") or []
            results.extend(page)
            cursor = body.get("next_url")
            if not cursor or len(page) == 0:
                return results
            # Polygon's next_url is a full URL; strip the host so the client
            # appends it to the configured base_url.
            path = cursor.split("polygon.io", 1)[-1]
            params = {}  # already encoded in next_url


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_decimal(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except Exception:
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Legacy adapter — wraps async into sync for the existing dataflow layer.
# ---------------------------------------------------------------------------


def get_polygon_stock_data_sync(
    symbol: str, curr_date: str, look_back_days: int, online: bool = True
) -> str:
    """Sync wrapper that mirrors `y_finance.get_YFin_data_online`'s string
    contract so it can drop into the existing `VENDOR_METHODS` table."""
    import asyncio

    end = datetime.strptime(curr_date, "%Y-%m-%d").date()
    start = end - timedelta(days=look_back_days)

    async def _run() -> pd.DataFrame:
        provider = PolygonProvider()
        try:
            return await provider.get_stock_data(symbol, start, end)
        finally:
            await provider.aclose()

    df = asyncio.run(_run())
    return df.to_csv(date_format="%Y-%m-%d")
