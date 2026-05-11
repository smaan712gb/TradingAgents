"""Closing-bell accumulation detector.

A confirmation layer for end-of-day institutional accumulation flow.
Designed to run at ~15:50 ET (10 minutes before the close) for any
ticker that's already a Buy candidate from the theme scorecard.

The shape we're looking for is a cluster, not a single big print:

  Gate 1 — End-of-Day Accumulation Quality
    * last-30-min RVOL >= 1.5-2.0× the 20-day average for that window
    * total day RVOL   >= 1.3×
    * MOC closing print at or above day's VWAP
    * stock spent >= 60% of the session above VWAP
    * theme confirmation: >= 3 basket members showing the same shape

  Gate 2 — After-Hours Follow-Through (16:00-16:30 ET)
    * >= 2 distinct AH prints
    * cumulative AH volume meets a size floor (mid-cap default 100K sh)
    * AH max price at or above the RTH close

  Failure filters (ANY tripping = skip)
    * Gap implied > 5% (likely mean-revert)
    * Friday AH (weekend repositioning noise)
    * Earnings within 14 days
    * Sector ETF (SMH/SOXX) below 20-day MA and trending down

Three distinguishable phenomena produce visually similar volume:
  1. Real accumulation  — VWAP-buying, MOC, dark prints leaking via FINRA
  2. Short covering     — also bullish but mean-reverts faster
  3. Gamma-pinned ramp  — fades immediately on the next open

The single most discriminating filter is **AH follow-through**:
gamma-driven ramps almost always fade after hours; real accumulation
holds or extends. Gate 2 is what carries the signal-to-noise weight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Tunables — operator can override per call site
# ---------------------------------------------------------------------------

LAST_30M_RVOL_TRIGGER       = 1.5     # >= 1.5x of trailing 20d same-window avg
DAY_RVOL_TRIGGER            = 1.3     # >= 1.3x of trailing 20d daily avg
TIME_ABOVE_VWAP_MIN_PCT     = 0.60    # >= 60% of bars above VWAP
AH_PRINT_COUNT_MIN          = 2       # at least 2 AH prints
AH_SIZE_FLOOR_SHARES        = 100_000 # cumulative AH volume floor (mid-cap)
THEME_CONFIRMATION_MIN      = 3       # >=3 basket members showing same shape
GAP_FAIL_THRESHOLD          = 0.05    # > 5% implied gap = likely fade
EARNINGS_BLACKOUT_DAYS      = 14      # skip if earnings within N days
SECTOR_TREND_DAYS           = 20      # SMH/SOXX vs 20d MA check


# Sector ETFs to check for tape-wide head-fake risk
SECTOR_HEAD_FAKE_ETFS       = {"SMH", "SOXX", "XLK"}


@dataclass
class AccumulationMetrics:
    """All the intraday + AH metrics we extract per symbol."""
    symbol: str
    last_30m_volume: float = 0.0
    last_30m_avg_20d: float = 0.0
    last_30m_rvol: Optional[float] = None
    day_volume: float = 0.0
    day_avg_20d: float = 0.0
    day_rvol: Optional[float] = None
    vwap: Optional[float] = None
    moc_price: Optional[float] = None
    moc_at_or_above_vwap: bool = False
    pct_session_above_vwap: Optional[float] = None
    ah_print_count: int = 0
    ah_cumulative_volume: float = 0.0
    ah_max_price: Optional[float] = None
    ah_holds_close: bool = False
    rth_close: Optional[float] = None
    fetched_at: Optional[datetime] = None


@dataclass
class AccumulationSignal:
    """Output of evaluate_closing_accumulation per symbol."""
    symbol: str
    setup_passes: bool = False
    confidence: str = "none"            # high | medium | low | none
    gate1_passes: bool = False
    gate2_passes: bool = False
    failures: list[str] = field(default_factory=list)
    theme_confirmed: Optional[bool] = None
    entry_recommendation: str = ""
    rationale: str = ""
    metrics: Optional[AccumulationMetrics] = None
    failure_filters_tripped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IBKR data adapters — pull intraday + AH bars
# ---------------------------------------------------------------------------


async def fetch_intraday_metrics(
    *, symbol: str, ibkr: Any,
) -> Optional[AccumulationMetrics]:
    """Pull 5-min bars for today (RTH only) + 20 trailing days; compute
    VWAP, day volume, last-30-min volume, RVOLs, MOC price, time-above-VWAP.

    Uses IBKR's reqHistoricalDataAsync (already in the provider). RTH-only
    so the AH prints are excluded — those come from fetch_after_hours_metrics.
    """
    try:
        # 20 days of 5-min RTH bars: 20 * 78 = 1560 bars
        bars = await ibkr.get_historical_bars(
            symbol=symbol, duration="20 D", bar_size="5 mins",
            what_to_show="TRADES", use_rth=True,
        )
    except Exception as e:
        logger.warning("CBA: intraday fetch failed for %s: %s", symbol, e)
        return None
    if not bars:
        return None

    # Group bars by trading day
    bars_by_day: dict[date, list[Any]] = {}
    for b in bars:
        bd = getattr(b, "date", None)
        if bd is None:
            continue
        day = bd.date() if isinstance(bd, datetime) else bd
        bars_by_day.setdefault(day, []).append(b)
    if not bars_by_day:
        return None

    # The most recent day in the data IS "today's" session
    days_sorted = sorted(bars_by_day.keys())
    today_day = days_sorted[-1]
    prior_days = days_sorted[:-1][-20:]  # up to 20 most recent prior days

    today_bars = bars_by_day[today_day]
    if not today_bars:
        return None

    # VWAP, day volume, MOC price, time-above-VWAP
    cum_vol = 0.0
    cum_pv = 0.0
    above_vwap_bars = 0
    total_bars = 0
    for b in today_bars:
        v = float(getattr(b, "volume", 0) or 0)
        c = float(getattr(b, "close", 0) or 0)
        if v <= 0 or c <= 0:
            continue
        cum_vol += v
        cum_pv += c * v
        total_bars += 1
        running_vwap = cum_pv / cum_vol
        if c >= running_vwap:
            above_vwap_bars += 1
    final_vwap = (cum_pv / cum_vol) if cum_vol > 0 else None
    pct_above = (above_vwap_bars / total_bars) if total_bars > 0 else None

    # MOC ≈ the last RTH bar's close
    last_bar = today_bars[-1]
    moc_price = float(getattr(last_bar, "close", 0) or 0) or None
    moc_above = bool(moc_price and final_vwap and moc_price >= final_vwap)

    # Last 30-min volume = last 6 bars (5-min × 6 = 30 min)
    last_30m_bars = today_bars[-6:] if len(today_bars) >= 6 else today_bars
    last_30m_volume = sum(float(getattr(b, "volume", 0) or 0) for b in last_30m_bars)

    # 20-day same-window: last 30 min of each prior day
    last_30m_avg_20d = 0.0
    prior_30m_vols = []
    for day in prior_days:
        day_bars = bars_by_day.get(day) or []
        last_6 = day_bars[-6:] if len(day_bars) >= 6 else day_bars
        prior_30m_vols.append(
            sum(float(getattr(b, "volume", 0) or 0) for b in last_6)
        )
    last_30m_avg_20d = (
        sum(prior_30m_vols) / len(prior_30m_vols) if prior_30m_vols else 0.0
    )
    last_30m_rvol = (
        last_30m_volume / last_30m_avg_20d
        if last_30m_avg_20d > 0 else None
    )

    # 20-day daily volume
    prior_daily_vols = []
    for day in prior_days:
        prior_daily_vols.append(
            sum(float(getattr(b, "volume", 0) or 0) for b in bars_by_day[day])
        )
    day_avg_20d = (
        sum(prior_daily_vols) / len(prior_daily_vols) if prior_daily_vols else 0.0
    )
    day_rvol = cum_vol / day_avg_20d if day_avg_20d > 0 else None

    return AccumulationMetrics(
        symbol=symbol,
        last_30m_volume=last_30m_volume,
        last_30m_avg_20d=last_30m_avg_20d,
        last_30m_rvol=last_30m_rvol,
        day_volume=cum_vol,
        day_avg_20d=day_avg_20d,
        day_rvol=day_rvol,
        vwap=final_vwap,
        moc_price=moc_price,
        moc_at_or_above_vwap=moc_above,
        pct_session_above_vwap=pct_above,
        rth_close=moc_price,
        fetched_at=datetime.now(timezone.utc),
    )


async def fetch_after_hours_metrics(
    *, symbol: str, ibkr: Any, rth_close: Optional[float],
) -> tuple[int, float, Optional[float], bool]:
    """Pull 1-min bars for today including AH; return (count, cumvol, max,
    holds_close) for the 16:00-16:30 ET window.

    use_rth=False so we get the AH session. Filters bars to the AH window
    based on ET wall-clock time.
    """
    if rth_close is None or rth_close <= 0:
        return (0, 0.0, None, False)
    try:
        bars = await ibkr.get_historical_bars(
            symbol=symbol, duration="1 D", bar_size="1 min",
            what_to_show="TRADES", use_rth=False,
        )
    except Exception as e:
        logger.warning("CBA: AH fetch failed for %s: %s", symbol, e)
        return (0, 0.0, None, False)
    if not bars:
        return (0, 0.0, None, False)

    ah_window_start = dtime(16, 0)
    ah_window_end   = dtime(16, 30)
    count = 0
    cumvol = 0.0
    maxp: Optional[float] = None
    for b in bars:
        bd = getattr(b, "date", None)
        if bd is None:
            continue
        bd_et = bd.astimezone(ET) if isinstance(bd, datetime) and bd.tzinfo else None
        if bd_et is None:
            continue
        t = bd_et.time()
        if not (ah_window_start <= t < ah_window_end):
            continue
        v = float(getattr(b, "volume", 0) or 0)
        c = float(getattr(b, "close", 0) or 0)
        if v <= 0 or c <= 0:
            continue
        count += 1
        cumvol += v
        if maxp is None or c > maxp:
            maxp = c

    holds_close = bool(maxp and maxp >= rth_close)
    return (count, cumvol, maxp, holds_close)


# ---------------------------------------------------------------------------
# Failure filters
# ---------------------------------------------------------------------------


async def check_failure_filters(
    *, symbol: str, ibkr: Any,
    rth_close: Optional[float], ah_max_price: Optional[float],
    now_et: Optional[datetime] = None,
) -> list[str]:
    """Return a list of tripped failure-filter names. Empty = clean."""
    now_et = now_et or datetime.now(ET)
    tripped: list[str] = []

    # Friday AH noise filter
    if now_et.weekday() == 4:                # Friday
        tripped.append("friday_ah_noise")

    # Wide-gap mean-revert filter
    if rth_close and ah_max_price:
        implied_gap = (ah_max_price - rth_close) / rth_close
        if implied_gap > GAP_FAIL_THRESHOLD:
            tripped.append(f"wide_gap_{implied_gap*100:.1f}pct")

    # Earnings blackout — within 14 days = positioning noise
    try:
        from tradingagents.strategies.maintenance.earnings import days_to_earnings
        dte = await days_to_earnings(symbol)
        if dte is not None and 0 <= dte <= EARNINGS_BLACKOUT_DAYS:
            tripped.append(f"earnings_in_{dte}d")
    except Exception:
        pass

    # Sector head-fake — SMH / SOXX below 20-day MA and trending down
    try:
        head_fake = await _sector_in_downtrend(ibkr=ibkr)
        if head_fake:
            tripped.append(f"sector_downtrend_{head_fake}")
    except Exception:
        pass

    return tripped


async def _sector_in_downtrend(*, ibkr: Any) -> Optional[str]:
    """True if SMH or SOXX is below its 20-day MA and trending down.
    Returns the offending ticker name, or None."""
    from tradingagents.dataflows.fallback import get_stock_data_with_fallback

    for etf in ("SMH", "SOXX"):
        try:
            df = await get_stock_data_with_fallback(
                etf, date.today() - timedelta(days=40), date.today(),
                ibkr_provider=ibkr,
            )
            if df is None or len(df) < 21 or "Close" not in df.columns:
                continue
            closes = df["Close"].astype(float).tolist()
            ma20 = sum(closes[-20:]) / 20
            last = closes[-1]
            slope_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
            if last < ma20 and slope_5d < 0:
                return etf
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


async def evaluate_closing_accumulation(
    *, symbol: str, ibkr: Any,
    theme_basket_signals: Optional[dict[str, bool]] = None,
    last_30m_rvol_trigger: float = LAST_30M_RVOL_TRIGGER,
    day_rvol_trigger: float = DAY_RVOL_TRIGGER,
    ah_size_floor: float = AH_SIZE_FLOOR_SHARES,
) -> AccumulationSignal:
    """Run both gates + failure filters for one symbol.

    ``theme_basket_signals`` is an optional map of {symbol: gate1_passed}
    for other basket members — used for the theme-confirmation check.
    When None, theme confirmation is reported as Unknown and doesn't
    block the signal.
    """
    metrics = await fetch_intraday_metrics(symbol=symbol, ibkr=ibkr)
    if metrics is None:
        return AccumulationSignal(
            symbol=symbol, setup_passes=False, confidence="none",
            rationale="intraday data unavailable",
        )

    ah_count, ah_vol, ah_max, ah_holds = await fetch_after_hours_metrics(
        symbol=symbol, ibkr=ibkr, rth_close=metrics.rth_close,
    )
    metrics.ah_print_count = ah_count
    metrics.ah_cumulative_volume = ah_vol
    metrics.ah_max_price = ah_max
    metrics.ah_holds_close = ah_holds

    failures: list[str] = []

    # ---- Gate 1: end-of-day accumulation quality ----
    gate1_ok = True
    if metrics.last_30m_rvol is None or metrics.last_30m_rvol < last_30m_rvol_trigger:
        gate1_ok = False
        failures.append(
            f"last_30m_rvol={metrics.last_30m_rvol:.2f}" if metrics.last_30m_rvol
            else "last_30m_rvol=unavailable"
        )
    if metrics.day_rvol is None or metrics.day_rvol < day_rvol_trigger:
        gate1_ok = False
        failures.append(
            f"day_rvol={metrics.day_rvol:.2f}" if metrics.day_rvol
            else "day_rvol=unavailable"
        )
    if not metrics.moc_at_or_above_vwap:
        gate1_ok = False
        failures.append("moc_below_vwap")
    if (metrics.pct_session_above_vwap is None
            or metrics.pct_session_above_vwap < TIME_ABOVE_VWAP_MIN_PCT):
        gate1_ok = False
        failures.append(
            f"time_above_vwap={metrics.pct_session_above_vwap:.0%}"
            if metrics.pct_session_above_vwap is not None
            else "time_above_vwap=unavailable"
        )

    # ---- Gate 2: after-hours follow-through ----
    gate2_ok = True
    if metrics.ah_print_count < AH_PRINT_COUNT_MIN:
        gate2_ok = False
        failures.append(f"ah_prints={metrics.ah_print_count}")
    if metrics.ah_cumulative_volume < ah_size_floor:
        gate2_ok = False
        failures.append(
            f"ah_volume={metrics.ah_cumulative_volume:,.0f} (need {ah_size_floor:,.0f})"
        )
    if not metrics.ah_holds_close:
        gate2_ok = False
        failures.append("ah_below_close")

    # ---- Failure filters ----
    failure_filters = await check_failure_filters(
        symbol=symbol, ibkr=ibkr,
        rth_close=metrics.rth_close, ah_max_price=metrics.ah_max_price,
    )

    # ---- Theme confirmation ----
    theme_confirmed: Optional[bool] = None
    if theme_basket_signals:
        confirmed_count = sum(1 for s, ok in theme_basket_signals.items() if ok and s != symbol)
        theme_confirmed = confirmed_count >= THEME_CONFIRMATION_MIN
        if not theme_confirmed:
            failures.append(f"theme_confirmation_{confirmed_count}/{THEME_CONFIRMATION_MIN}")

    # ---- Final verdict + confidence ----
    setup_passes = (gate1_ok and gate2_ok and not failure_filters)
    if theme_confirmed is False:
        setup_passes = False

    if setup_passes and theme_confirmed:
        confidence = "high"
        entry = "Aggressive: close-buy now (gates clean + theme confirmed). Standard: opening range break tomorrow."
    elif setup_passes:
        confidence = "medium"
        entry = "Standard: opening range break tomorrow. Conservative: VWAP reclaim entry."
    elif gate1_ok and not gate2_ok:
        confidence = "low"
        entry = "Skip — closing print was strong but AH faded. Likely gamma-pinned ramp, not accumulation."
    else:
        confidence = "none"
        entry = "Skip — gates did not pass."

    rationale = (
        f"{symbol}: gate1={'PASS' if gate1_ok else 'FAIL'}, "
        f"gate2={'PASS' if gate2_ok else 'FAIL'}"
        + (f", filters_tripped={','.join(failure_filters)}" if failure_filters else "")
        + (f", theme_confirmed={theme_confirmed}" if theme_confirmed is not None else "")
    )

    return AccumulationSignal(
        symbol=symbol,
        setup_passes=setup_passes,
        confidence=confidence,
        gate1_passes=gate1_ok,
        gate2_passes=gate2_ok,
        failures=failures,
        theme_confirmed=theme_confirmed,
        entry_recommendation=entry,
        rationale=rationale,
        metrics=metrics,
        failure_filters_tripped=failure_filters,
    )


async def sweep_theme_for_accumulation(
    *, theme_id: str, symbols: list[str], ibkr: Any,
) -> list[AccumulationSignal]:
    """Run the CBA detector across an entire theme basket, with the
    theme-confirmation pass requiring multiple basket members to show
    the same shape.

    Two-pass: first compute per-symbol metrics + gate1, then second pass
    folds in theme confirmation now that we know which siblings passed.
    """
    import asyncio
    # Pass 1: per-symbol evaluation, no theme confirmation yet
    first_results = await asyncio.gather(
        *(evaluate_closing_accumulation(symbol=s, ibkr=ibkr) for s in symbols),
        return_exceptions=True,
    )

    # Build the basket gate1 map
    basket_gate1: dict[str, bool] = {}
    for s, r in zip(symbols, first_results):
        if isinstance(r, AccumulationSignal):
            basket_gate1[s] = r.gate1_passes

    # Pass 2: only re-evaluate the ones that passed gate1, to update
    # theme-confirmation status (others stay as-is)
    final: list[AccumulationSignal] = []
    for s, r in zip(symbols, first_results):
        if isinstance(r, Exception):
            logger.warning("CBA sweep: %s raised %s", s, r)
            final.append(AccumulationSignal(
                symbol=s, setup_passes=False, confidence="none",
                rationale=f"error: {r}",
            ))
            continue
        if not isinstance(r, AccumulationSignal):
            continue
        if r.gate1_passes:
            r2 = await evaluate_closing_accumulation(
                symbol=s, ibkr=ibkr, theme_basket_signals=basket_gate1,
            )
            final.append(r2)
        else:
            final.append(r)
    return final
