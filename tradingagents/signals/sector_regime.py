"""Sector-ETF regime detection.

The agents trade single stocks, but every long thesis lives or dies by
the sector's regime. SMH at $300 holding the 50-day with rising 20-day
RoC is a different world from SMH below the 200-day, vol expanding, no
buyers stepping in. This module computes that picture per theme so the
auto-gate can refuse new long entries during confirmed sector downtrends
and the agents can ground their reasoning in regime context.

Each theme maps to one or two ETFs (primary + optional secondary). The
worst-of-N regime across the mapped ETFs is the theme's regime — keeps
us from getting cute when only one of two correlated ETFs is healthy.

Data sources, in order of preference:

  * Spot + bars: IBKR (paid subscription) — real-time and authoritative.
  * Spot + bars fallback: yfinance — free and unauthenticated. Used when
    IBKR Gateway isn't reachable so the system degrades gracefully rather
    than hard-failing on the regime gate.
  * Smart-money tilt: Unusual Whales gamma exposure + flow alerts ON the
    ETF itself. Gamma sign + recent put/call premium tilt is leading;
    price-MTF crosses confirm what UW shows hours-to-days earlier. This
    is the headline upgrade — UW > price.

Classification combines price-MTF and UW context, *worst-applicable wins*:

  uptrend     — price > 50ma > 200ma, 20d momentum > 0, UW gamma supportive
  pullback    — price > 200ma but < 50ma OR UW gamma flipped to negative
  range       — flat 50ma + sideways momentum + UW neutral
  downtrend   — price < 50ma < 200ma, 20d momentum < 0, UW negative gamma

UW context can DEMOTE a price-MTF uptrend to pullback if dealer
positioning has flipped — that's how we get ahead of regime shifts
that haven't yet shown up on the daily chart.

Cache TTL is 15 min (price) and 5 min (UW) — UW updates faster.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal, Optional

logger = logging.getLogger(__name__)


Regime = Literal["uptrend", "pullback", "range", "downtrend", "unknown"]


# ---------------------------------------------------------------------------
# Theme → ETF map. Reviewed top-down so the mapping is the same wherever it's
# read. Primary first; secondary is a confirming or contrasting tape.
# ---------------------------------------------------------------------------

THEME_TO_ETFS: dict[str, list[str]] = {
    # AI / semi cluster — SMH is the cleanest regime tape, SOXX for confirm.
    "ai-memory-wall":           ["SMH", "SOXX"],
    "advanced-packaging":       ["SMH", "SOXX"],
    "custom-silicon-supply":    ["SMH", "SOXX"],
    "ai-interconnect":          ["SMH", "SOXX"],
    "ai-storage":               ["SMH", "SOXX"],
    "ai-test-metrology":        ["SMH", "SOXX"],
    "on-device-ai":             ["SMH", "QQQ"],   # mobile/edge → broader tech tape
    "private-ai-compute":       ["SMH", "QQQ"],
    "optical-networking":       ["SMH", "IYZ"],   # IYZ = telecom equipment
    # Power / infrastructure cluster — XLU for utility regime, XLI for builder regime.
    "data-center-power-wall":   ["XLU", "XLI"],
    "grid-bottleneck":          ["XLU", "XLI"],
    "ai-dc-construction":       ["XLI", "XLU"],
    "liquid-cooling":           ["SMH", "XLI"],   # AI demand drives, industrials build
    # Energy / materials.
    "nuclear-for-ai":           ["URA", "XLU"],
    "critical-materials":       ["XME", "XLB"],
}

# Broader tape ETFs read once per cycle and shown as regime context for everything.
TAPE_ETFS = ["SPY", "QQQ", "^VIX", "TLT"]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class UwContext:
    """UW-derived smart-money tilt for one ETF."""
    gamma_sign: Optional[Literal["positive", "negative", "neutral"]] = None
    gamma_flip_strike: Optional[float] = None
    flow_premium_24h_call: float = 0.0
    flow_premium_24h_put: float = 0.0
    flow_tilt: Optional[Literal["bullish", "bearish", "neutral"]] = None
    note: str = ""


@dataclass
class RegimeContext:
    theme_id: str
    etfs: list[str]
    regime: Regime
    rationale: str               # one-line plain-English summary
    spot: dict[str, float]       # ETF → last close
    vs_50ma_pct: dict[str, float]
    vs_200ma_pct: dict[str, float]
    momentum_20d_pct: dict[str, float]
    realized_vol_30d_pct: dict[str, float]
    dd_from_52wh_pct: dict[str, float]
    uw: dict[str, UwContext]     # ETF → smart-money tilt
    price_source: dict[str, str] # ETF → "ibkr" | "yfinance"
    captured_at: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d["captured_at"] = self.captured_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------


_CACHE: dict[str, tuple[float, RegimeContext]] = {}
_CACHE_TTL_SEC = 15 * 60


# ---------------------------------------------------------------------------
# yfinance helpers (sync API → run in a thread)
# ---------------------------------------------------------------------------


async def _fetch_ohlcv_polygon(symbol: str) -> Optional[list[tuple[str, float, float]]]:
    """Polygon daily aggregates for a 1-year window. Preferred read path.

    Polygon is purpose-built for historical equity reads and doesn't have
    IBKR's single-session restriction. We use it for everything bar-level
    on ETFs and stocks; IBKR remains the authoritative source for the
    *account* (positions, account summary, orders).

    Failure modes (rate-limit, error, etc.) return None so the caller
    falls back to yfinance.
    """
    try:
        from datetime import date, timedelta
        import pandas as pd
        from tradingagents.dataflows.providers.polygon import PolygonProvider  # type: ignore
        p = PolygonProvider()
        end = date.today()
        start = end - timedelta(days=400)
        df = await p.get_stock_data(symbol, start, end)
        # Defensive: PolygonProvider can return a string error message on
        # rate-limit instead of a DataFrame. Coerce any non-DataFrame to None.
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        rows: list[tuple[str, float, float]] = []
        for ts, row in df.iterrows():
            rows.append((
                ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10],
                float(row["Close"] if "Close" in df.columns else row["close"]),
                float(row["High"]  if "High"  in df.columns else row["high"]),
            ))
        return rows
    except Exception as e:
        logger.warning("Polygon historical fetch failed for %s: %s", symbol, e)
        return None


def _fetch_ohlcv_yfinance_sync(symbol: str, period: str = "1y") -> Optional[list[tuple[str, float, float]]]:
    """Last-resort fallback: yfinance daily bars. Sync — wrap via to_thread."""
    try:
        import yfinance as yf  # type: ignore
        df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        if df.empty or "Close" not in df.columns:
            return None
        rows: list[tuple[str, float, float]] = []
        for ts, row in df.iterrows():
            rows.append((ts.strftime("%Y-%m-%d"), float(row["Close"]), float(row["High"])))
        return rows
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", symbol, e)
        return None


async def _fetch_ohlcv(symbol: str) -> tuple[Optional[list[tuple[str, float, float]]], str]:
    """Polygon → yfinance. IBKR is intentionally not used here (single-session
    market-data restriction conflicts with TWS); IBKR is reserved for orders
    and live position reads via the singleton in api.app.positions.
    Returns (rows, source_label).
    """
    rows = await _fetch_ohlcv_polygon(symbol)
    if rows:
        return rows, "polygon"
    rows = await asyncio.to_thread(_fetch_ohlcv_yfinance_sync, symbol)
    return rows, ("yfinance" if rows else "none")


# ---------------------------------------------------------------------------
# UW enrichment — gamma + flow tilt per ETF
# ---------------------------------------------------------------------------


def _occ_right(contract: str) -> str:
    """Return "C" or "P" from an OCC option symbol, or "" if unparseable.

    OCC format is root + YYMMDD + C|P + 8-digit strike, e.g.

        NVDA260904C00227500
                   ^ right, 9 chars from the end

    This used to be `contract.endswith("C")`, which matches nothing: the symbol
    ends in the STRIKE, not the right. Both the call and put sums were therefore
    always exactly 0.0, for every symbol, silently — the fetch succeeded, the
    list was populated, and the totals were zero, so nothing logged an error.

    Measured on NVDA (2026-08-18, 165 live alerts): the correct parse gives
    $52,866,161 call premium against $9,710,158 put — a decisively bullish
    tape the system had been reading as no data at all.

    Three consumers were affected, all failing quietly toward "no signal":
      * flow_tilt was never once "bullish" across the whole universe, so the
        rotation detector's `bearish > bullish` test was free and
        flow_distribution tripped on 15 of 17 themes;
      * flow_imbalance was always None, so z_flow_imbalance carried zero
        variance and the quant overlay dropped it entirely;
      * the scorecard's options score sat at its 5.0 neutral default on every
        idea in the morning report.

    Kept deliberately strict: the right must be exactly where OCC puts it, and
    anything else returns "" rather than guessing, so a malformed symbol is
    excluded from both sums instead of being silently counted as a call.
    """
    c = (contract or "").strip().upper()
    # Minimum viable OCC symbol is root(>=1) + YYMMDD(6) + right(1) + strike(8).
    # Validate the date and strike too, so a fragment like "C00227500" — which
    # has the right in the correct SLOT but no ticker root — is rejected rather
    # than silently counted as a call.
    if len(c) < 16:
        return ""
    if c[-9] not in ("C", "P"):
        return ""
    if not c[-8:].isdigit() or not c[-15:-9].isdigit():
        return ""
    return c[-9]


async def _fetch_uw_context(symbol: str) -> UwContext:
    """Pull gamma exposure + recent flow tilt from UW for an ETF.

    Failure to fetch returns an empty UwContext — the regime stays in
    price-MTF-only mode. Call sites must not assume UW data is present.
    """
    try:
        from tradingagents.dataflows.providers.unusual_whales import UnusualWhalesProvider
    except Exception:
        return UwContext(note="UW provider not importable")

    try:
        uw = UnusualWhalesProvider()
    except Exception as e:
        return UwContext(note=f"UW init failed: {e}")

    ctx = UwContext()
    # Gamma exposure: positive GEX = MM long gamma (vol-suppressing), negative = vol-amplifying.
    try:
        gex_levels = await uw.get_gamma_levels(symbol)
        if gex_levels:
            # Aggregate net gamma at spot. UW returns a list of GammaLevel; we
            # take the closest-to-spot level's notional sign as the signal.
            closest = min(gex_levels, key=lambda g: abs(float(g.price)))
            sign = "positive" if (closest.notional or 0) > 0 else (
                "negative" if (closest.notional or 0) < 0 else "neutral"
            )
            ctx.gamma_sign = sign  # type: ignore[assignment]
            ctx.gamma_flip_strike = float(closest.price)
    except Exception as e:
        ctx.note = f"gamma fetch failed: {e}"

    # Flow alerts in the last 24h — net call vs put premium.
    try:
        from datetime import timedelta
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        alerts = await uw.get_options_flow(symbol=symbol, since=since)
        call_prem = sum(float(a.premium or 0) for a in alerts
                        if _occ_right(a.contract) == "C")
        put_prem = sum(float(a.premium or 0) for a in alerts
                       if _occ_right(a.contract) == "P")
        ctx.flow_premium_24h_call = round(call_prem, 0)
        ctx.flow_premium_24h_put = round(put_prem, 0)
        if call_prem > put_prem * 1.5 and call_prem > 250_000:
            ctx.flow_tilt = "bullish"
        elif put_prem > call_prem * 1.5 and put_prem > 250_000:
            ctx.flow_tilt = "bearish"
        elif call_prem + put_prem > 100_000:
            ctx.flow_tilt = "neutral"
    except Exception as e:
        if not ctx.note:
            ctx.note = f"flow fetch failed: {e}"

    return ctx


def _sma(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _slope_pct(values: list[float], n: int) -> Optional[float]:
    """Percent change over the last n values."""
    if len(values) < n + 1:
        return None
    a, b = values[-n - 1], values[-1]
    if a == 0:
        return None
    return (b / a - 1.0) * 100


def _realized_vol_30d_pct(closes: list[float]) -> Optional[float]:
    """Annualised stdev of last 30 daily log returns, in %."""
    import math
    if len(closes) < 31:
        return None
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(-30, 0)]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252) * 100


def _dd_from_52wh_pct(closes: list[float], highs: list[float]) -> Optional[float]:
    """Last close vs trailing 52-week high, in %. Negative = drawdown."""
    if not closes or not highs:
        return None
    window = min(252, len(highs))
    peak = max(highs[-window:])
    if peak == 0:
        return None
    return (closes[-1] / peak - 1.0) * 100


# ---------------------------------------------------------------------------
# Per-ETF regime classification
# ---------------------------------------------------------------------------


def classify_regime(
    closes: list[float],
    highs: Optional[list[float]] = None,
) -> tuple[Regime, str]:
    """Classify a single ETF's regime from its close series.

    Returns (regime, rationale). Highs optional — used only for drawdown.
    """
    if len(closes) < 200:
        return ("unknown", "insufficient history")

    last = closes[-1]
    sma50 = _sma(closes, 50) or 0.0
    sma200 = _sma(closes, 200) or 0.0
    mom20 = _slope_pct(closes, 20) or 0.0
    sma50_slope = _slope_pct(closes[:-0] if False else closes, 20) or 0.0  # 50ma slope proxy: 20d roc on price

    # Decision tree, worst-of wins to err on the side of safety.
    if last < sma50 < sma200 and mom20 < 0:
        return ("downtrend",
                f"below 50/200 ({last:.1f} < {sma50:.1f} < {sma200:.1f}), 20d momentum {mom20:.1f}%")
    if last > sma50 > sma200 and mom20 > 0:
        return ("uptrend",
                f"above 50/200, 20d momentum {mom20:.1f}%")
    if last > sma200 and (last < sma50 or mom20 < 0):
        return ("pullback",
                f"above 200 but {('below 50' if last < sma50 else 'momentum negative')} (mom {mom20:.1f}%)")
    return ("range",
            f"price {last:.1f} flat around 50/200 ({sma50:.1f}/{sma200:.1f}), mom {mom20:.1f}%")


def _worst_regime(regimes: list[Regime]) -> Regime:
    """Pick the worst regime across mapped ETFs.

    Order: downtrend < pullback < range < uptrend, unknown floats above range.
    """
    rank = {"downtrend": 0, "pullback": 1, "range": 2, "unknown": 3, "uptrend": 4}
    return min(regimes, key=lambda r: rank.get(r, 99))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def get_theme_regime(theme_id: str) -> RegimeContext:
    """Compute (or return cached) regime for ``theme_id``.

    Falls back to a permissive ``unknown`` regime when yfinance is unreachable
    so a network blip doesn't block all auto entries; the auto-gate treats
    ``unknown`` as no-block (informational only).
    """
    now = time.time()
    cached = _CACHE.get(theme_id)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    etfs = THEME_TO_ETFS.get(theme_id, [])
    if not etfs:
        ctx = RegimeContext(
            theme_id=theme_id, etfs=[], regime="unknown",
            rationale="no ETF mapping for theme",
            spot={}, vs_50ma_pct={}, vs_200ma_pct={},
            momentum_20d_pct={}, realized_vol_30d_pct={},
            dd_from_52wh_pct={}, uw={}, price_source={},
            captured_at=datetime.now(timezone.utc),
        )
        _CACHE[theme_id] = (now, ctx)
        return ctx

    # Pull ETF history (IBKR-first, yfinance-fallback) and UW context concurrently.
    price_results = await asyncio.gather(*(_fetch_ohlcv(etf) for etf in etfs))
    uw_results = await asyncio.gather(*(_fetch_uw_context(etf) for etf in etfs))

    spot: dict[str, float] = {}
    vs50: dict[str, float] = {}
    vs200: dict[str, float] = {}
    mom: dict[str, float] = {}
    rv: dict[str, float] = {}
    dd: dict[str, float] = {}
    uw_map: dict[str, UwContext] = {}
    src_map: dict[str, str] = {}
    per_etf_regime: list[Regime] = []
    rationales: list[str] = []

    for etf, (rows, src), uw_ctx in zip(etfs, price_results, uw_results):
        uw_map[etf] = uw_ctx
        src_map[etf] = src
        if not rows:
            per_etf_regime.append("unknown")
            rationales.append(f"{etf}: no price data")
            continue
        closes = [r[1] for r in rows]
        highs = [r[2] for r in rows]
        price_regime, price_rationale = classify_regime(closes, highs)
        # Combine price-MTF with UW tilt.
        regime, rationale_extra = _combine_with_uw(price_regime, uw_ctx)
        per_etf_regime.append(regime)
        rationales.append(f"{etf}: {price_rationale}{rationale_extra}")
        spot[etf] = round(closes[-1], 2)
        sma50 = _sma(closes, 50)
        sma200 = _sma(closes, 200)
        if sma50:
            vs50[etf] = round((closes[-1] / sma50 - 1) * 100, 2)
        if sma200:
            vs200[etf] = round((closes[-1] / sma200 - 1) * 100, 2)
        m = _slope_pct(closes, 20)
        if m is not None:
            mom[etf] = round(m, 2)
        v = _realized_vol_30d_pct(closes)
        if v is not None:
            rv[etf] = round(v, 1)
        d = _dd_from_52wh_pct(closes, highs)
        if d is not None:
            dd[etf] = round(d, 2)

    overall = _worst_regime(per_etf_regime) if per_etf_regime else "unknown"
    ctx = RegimeContext(
        theme_id=theme_id, etfs=etfs, regime=overall,
        rationale=" · ".join(rationales) or "no data",
        spot=spot, vs_50ma_pct=vs50, vs_200ma_pct=vs200,
        momentum_20d_pct=mom, realized_vol_30d_pct=rv, dd_from_52wh_pct=dd,
        uw=uw_map, price_source=src_map,
        captured_at=datetime.now(timezone.utc),
    )
    _CACHE[theme_id] = (now, ctx)
    return ctx


def _combine_with_uw(price_regime: Regime, uw: UwContext) -> tuple[Regime, str]:
    """Apply UW context as an adjustment to the price-MTF regime.

    Demote logic (UW *earlier* than price-MTF most of the time):
      uptrend  + negative gamma + bearish flow → pullback (smart money exiting)
      uptrend  + bearish flow only             → uptrend (note flow risk)
      pullback + positive gamma + bullish flow → uptrend (regime turning up)
      range    + bearish flow                  → pullback
    """
    if uw.gamma_sign is None and uw.flow_tilt is None:
        return price_regime, ""

    # Strong demotion: uptrend with both flips → pullback
    if (price_regime == "uptrend"
            and uw.gamma_sign == "negative"
            and uw.flow_tilt == "bearish"):
        return ("pullback",
                f" — UW: negative gamma + bearish flow demotes from uptrend")

    # Promotion: pullback turning into uptrend per smart money
    if (price_regime == "pullback"
            and uw.gamma_sign == "positive"
            and uw.flow_tilt == "bullish"):
        return ("uptrend",
                f" — UW: positive gamma + bullish flow promotes from pullback")

    # Range with confirming bear flow → pullback
    if price_regime == "range" and uw.flow_tilt == "bearish":
        return ("pullback", f" — UW: bearish flow tilts range to pullback")

    # Otherwise leave price regime alone but note UW context
    notes = []
    if uw.gamma_sign:
        notes.append(f"gamma {uw.gamma_sign}")
    if uw.flow_tilt:
        notes.append(f"flow {uw.flow_tilt}")
    return price_regime, f" — UW: {', '.join(notes)}" if notes else ""


def invalidate_cache(theme_id: Optional[str] = None) -> None:
    """Clear cached regimes. Useful after market close or for tests."""
    if theme_id is None:
        _CACHE.clear()
    else:
        _CACHE.pop(theme_id, None)
