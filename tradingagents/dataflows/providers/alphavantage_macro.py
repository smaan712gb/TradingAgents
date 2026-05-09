"""AlphaVantage macro / global indicators provider.

Implements `MacroSignalProvider`. The upstream repo already wraps
AlphaVantage for stocks/news/fundamentals — we keep that as-is and add a
narrower wrapper for the macro endpoints the Alternative Data Agent uses:

* `function=TREASURY_YIELD` (3M / 2Y / 10Y / 30Y)
* `function=FEDERAL_FUNDS_RATE`
* `function=CPI`
* `function=INFLATION`
* `function=REAL_GDP`
* `function=RETAIL_SALES`
* `function=UNEMPLOYMENT`
* `function=NONFARM_PAYROLL`

These are slow-moving series — we cache aggressively (24h) and the agent
gets a deterministic markdown summary.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Optional

from .base import AuthError, MacroSignal, ProviderError
from .cache import cached
from .http import AsyncHttpClient

logger = logging.getLogger(__name__)


_SERIES_TO_FUNCTION: dict[str, dict[str, Any]] = {
    "TREASURY_YIELD_3M":    {"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "3month"},
    "TREASURY_YIELD_2Y":    {"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "2year"},
    "TREASURY_YIELD_10Y":   {"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "10year"},
    "TREASURY_YIELD_30Y":   {"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "30year"},
    "FED_FUNDS_RATE":       {"function": "FEDERAL_FUNDS_RATE", "interval": "monthly"},
    "CPI_YOY":              {"function": "CPI", "interval": "monthly"},
    "INFLATION":            {"function": "INFLATION"},
    "REAL_GDP":             {"function": "REAL_GDP", "interval": "annual"},
    "RETAIL_SALES":         {"function": "RETAIL_SALES"},
    "UNEMPLOYMENT":         {"function": "UNEMPLOYMENT"},
    "NONFARM_PAYROLL":      {"function": "NONFARM_PAYROLL"},
}

_UNITS: dict[str, str] = {
    "TREASURY_YIELD_3M":  "%",
    "TREASURY_YIELD_2Y":  "%",
    "TREASURY_YIELD_10Y": "%",
    "TREASURY_YIELD_30Y": "%",
    "FED_FUNDS_RATE":     "%",
    "CPI_YOY":            "index",
    "INFLATION":          "%",
    "REAL_GDP":           "$B",
    "RETAIL_SALES":       "$M",
    "UNEMPLOYMENT":       "%",
    "NONFARM_PAYROLL":    "thousands",
}


class AlphaVantageMacroProvider:
    name = "alphavantage_macro"

    def __init__(self, api_key: Optional[str] = None) -> None:
        api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            raise AuthError("alphavantage_macro")
        self._api_key = api_key
        self._http = AsyncHttpClient(
            provider_name="alphavantage_macro",
            base_url="https://www.alphavantage.co",
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    @cached(ttl_s=24 * 3600, namespace="av.macro")
    async def get_macro_signal(self, series_id: str, lookback_days: int = 365) -> list[MacroSignal]:
        spec = _SERIES_TO_FUNCTION.get(series_id)
        if not spec:
            raise ProviderError(
                "alphavantage_macro",
                f"unknown series {series_id!r}; expected one of {list(_SERIES_TO_FUNCTION)}",
            )
        params = {**spec, "apikey": self._api_key, "datatype": "json"}
        body = await self._http.get_json("/query", params=params)
        if isinstance(body, dict) and ("Error Message" in body or "Note" in body):
            # AlphaVantage uses these for both auth errors and rate limits.
            raise ProviderError("alphavantage_macro", body.get("Error Message") or body["Note"])
        rows = body.get("data") or []
        cutoff = date.today() - timedelta(days=lookback_days)
        out: list[MacroSignal] = []
        for r in rows:
            try:
                d = date.fromisoformat(r["date"])
            except (KeyError, ValueError):
                continue
            if d < cutoff:
                continue
            try:
                value = float(r["value"])
            except (KeyError, ValueError, TypeError):
                continue
            out.append(
                MacroSignal(
                    series_id=series_id,
                    observed_at=d,
                    value=value,
                    unit=_UNITS.get(series_id, ""),
                )
            )
        out.sort(key=lambda s: s.observed_at)
        return out

    async def summarize_for_agent(self, series_ids: list[str], lookback_days: int = 365) -> str:
        """Render a tidy macro context block for the Alternative Data Agent."""
        import asyncio

        results = await asyncio.gather(
            *(self.get_macro_signal(sid, lookback_days) for sid in series_ids),
            return_exceptions=True,
        )

        lines = [f"## Macro context — last {lookback_days}d"]
        for sid, res in zip(series_ids, results):
            if isinstance(res, Exception):
                lines.append(f"- **{sid}**: _unavailable ({res})_")
                continue
            if not res:
                lines.append(f"- **{sid}**: _no data_")
                continue
            latest = res[-1]
            prior = res[0]
            delta = latest.value - prior.value
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
            lines.append(
                f"- **{sid}**: {latest.value:,.2f}{latest.unit}  ({arrow} {delta:+,.2f} since {prior.observed_at})"
            )
        return "\n".join(lines)
