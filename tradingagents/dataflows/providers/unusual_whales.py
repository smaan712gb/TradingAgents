"""Unusual Whales provider.

Implements `OptionsFlowProvider`. UW exposes a large surface; we wrap the
endpoints the Thematic Scorecard cares about:

* `/api/stock/{ticker}/option-trades`     — block / sweep flow alerts
* `/api/stock/{ticker}/greek-exposure`    — gamma/delta exposure curve
* `/api/stock/{ticker}/max-pain`          — per-expiry max pain price
* `/api/stock/{ticker}/dark-pool-prints`  — off-exchange trade prints
* `/api/stock/{ticker}/atm-chain`         — at-the-money chain (greeks)

The actual paths sometimes shift between API revisions — anything you
discover that's different in your tier should be patched in this file
only; the agents don't know about UW endpoint shapes.

Rate limit: UW's documented default is 600 requests/min per key. The
shared `AsyncHttpClient` semaphore caps concurrency at 6 by default; we
also sleep on 429 with the `Retry-After` hint.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from .base import (
    AuthError,
    FlowAlert,
    GammaLevel,
    ProviderError,
)
from .cache import cached
from .http import AsyncHttpClient

logger = logging.getLogger(__name__)


class UnusualWhalesProvider:
    name = "unusual_whales"

    def __init__(self, api_key: Optional[str] = None) -> None:
        # Accept both env var names — the architecture doc uses
        # UNUSUAL_WHALES_API_KEY, the .env shipped with this project uses
        # UW_API_KEY (shorter). Either works.
        api_key = (
            api_key
            or os.getenv("UNUSUAL_WHALES_API_KEY")
            or os.getenv("UW_API_KEY")
        )
        if not api_key:
            raise AuthError("unusual_whales")
        self._http = AsyncHttpClient(
            provider_name="unusual_whales",
            base_url="https://api.unusualwhales.com",
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/plain, */*",
            },
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # OptionsFlowProvider
    # ------------------------------------------------------------------

    @cached(ttl_s=30, namespace="uw.flow")
    async def get_options_flow(
        self,
        symbol: str,
        since: datetime,
        min_premium: Decimal = Decimal("100000"),
    ) -> list[FlowAlert]:
        # Endpoint shape was `option-trades` in older UW docs but is now
        # `flow-alerts` on the live API. Both return the same alert payload.
        # The live API caps `limit` at 200, so we request the max and let
        # the agent reason about the busiest 200 alerts.
        body = await self._http.get_json(
            f"/api/stock/{symbol}/flow-alerts",
            params={
                "min_premium": str(min_premium),
                "limit": 200,
            },
        )
        rows = body.get("data") or body.get("trades") or []
        alerts: list[FlowAlert] = []
        for r in rows:
            triggered_at = _parse_dt(r.get("executed_at") or r.get("time"))
            if triggered_at is None or triggered_at < since:
                continue
            alerts.append(
                FlowAlert(
                    ticker=symbol,
                    triggered_at=triggered_at,
                    contract=r.get("option_symbol") or r.get("contract") or "",
                    side=r.get("side") or r.get("side_taker") or "",
                    premium=Decimal(str(r.get("premium") or r.get("total_premium") or 0)),
                    size=int(r.get("size") or r.get("volume") or 0),
                    open_interest=int(r.get("open_interest") or 0),
                    iv=_to_float(r.get("implied_volatility")),
                    sweep=bool(r.get("sweep") or r.get("is_sweep") or False),
                    repeat_count=int(r.get("repeat_count") or 1),
                    raw=r,
                )
            )
        return alerts

    @cached(ttl_s=300, namespace="uw.gamma")
    async def get_gamma_levels(self, symbol: str) -> list[GammaLevel]:
        body = await self._http.get_json(f"/api/stock/{symbol}/greek-exposure")
        levels = body.get("levels") or body.get("data") or []
        captured_at = datetime.now(tz=timezone.utc)
        out: list[GammaLevel] = []
        for lvl in levels:
            out.append(
                GammaLevel(
                    ticker=symbol,
                    captured_at=captured_at,
                    level_type=lvl.get("type") or lvl.get("kind") or "level",
                    price=Decimal(str(lvl.get("price") or lvl.get("strike") or 0)),
                    notional=_to_decimal(lvl.get("notional") or lvl.get("gex_dollar")),
                    notes=lvl.get("notes") or "",
                )
            )
        return out

    @cached(ttl_s=600, namespace="uw.maxpain")
    async def get_max_pain(self, symbol: str, expiry: Optional[Any] = None) -> Decimal:
        params: dict[str, Any] = {}
        if expiry is not None:
            params["expiration_date"] = expiry.isoformat() if hasattr(expiry, "isoformat") else expiry
        body = await self._http.get_json(f"/api/stock/{symbol}/max-pain", params=params)
        # API returns either {"max_pain": 1230.0} or {"data":[{"expiry":..., "max_pain":...}]}
        if isinstance(body, dict) and "max_pain" in body:
            return Decimal(str(body["max_pain"]))
        rows = body.get("data") or []
        if not rows:
            raise ProviderError("unusual_whales", f"no max-pain data for {symbol}")
        # Closest-expiry by default.
        rows.sort(key=lambda r: r.get("expiration_date") or r.get("expiry") or "")
        return Decimal(str(rows[0].get("max_pain") or rows[0].get("price") or 0))

    @cached(ttl_s=120, namespace="uw.darkpool")
    async def get_dark_pool_prints(self, symbol: str, since: datetime) -> list[dict[str, Any]]:
        body = await self._http.get_json(
            f"/api/stock/{symbol}/dark-pool-prints",
            params={"limit": 500},
        )
        rows = body.get("data") or []
        out: list[dict[str, Any]] = []
        for r in rows:
            ts = _parse_dt(r.get("executed_at") or r.get("time"))
            if ts is None or ts < since:
                continue
            r["_ts"] = ts.isoformat()
            out.append(r)
        return out

    # ------------------------------------------------------------------
    # Higher-level helpers — used by the new options-flow analyst.
    # ------------------------------------------------------------------

    async def summarize_flow_for_agent(self, symbol: str, hours: int = 24) -> str:
        """Produce a markdown summary suitable for direct injection into an
        analyst prompt. The analyst still does the reasoning; this helper
        just gives it a tidy, deterministic, deduped view."""
        since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        alerts, gamma, mp = await _gather(
            self.get_options_flow(symbol, since),
            self.get_gamma_levels(symbol),
            self.get_max_pain(symbol),
        )
        # Bucket flow by direction.
        bullish = [a for a in alerts if "ABOVE" in a.side.upper() or "C " in a.contract]
        bearish = [a for a in alerts if "BELOW" in a.side.upper() or "P " in a.contract]
        sweeps = [a for a in alerts if a.sweep]

        lines = [
            f"## Options flow — {symbol} (last {hours}h)",
            f"- Alerts: {len(alerts)}  (bullish-leaning: {len(bullish)}, bearish-leaning: {len(bearish)}, sweeps: {len(sweeps)})",
            f"- Aggregate premium: ${sum(int(a.premium) for a in alerts):,.0f}",
            f"- Max pain (nearest expiry): ${mp}",
            "",
            "### Gamma levels",
        ]
        for lvl in sorted(gamma, key=lambda g: g.price)[:8]:
            notional = f" (${int(lvl.notional):,})" if lvl.notional else ""
            lines.append(f"- {lvl.level_type:>14} @ ${lvl.price}{notional}  {lvl.notes}")
        lines.append("")
        lines.append("### Top alerts by premium")
        for a in sorted(alerts, key=lambda x: x.premium, reverse=True)[:10]:
            lines.append(
                f"- {a.triggered_at:%Y-%m-%d %H:%M} {a.contract:30s} {a.side:>14}  "
                f"size={a.size:>5} prem=${int(a.premium):>10,}  oi={a.open_interest}  sweep={a.sweep}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _gather(*coros: Any) -> Any:
    import asyncio
    return await asyncio.gather(*coros)


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        # UW timestamps are ISO-8601 with Z; tolerate both.
        s = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


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
