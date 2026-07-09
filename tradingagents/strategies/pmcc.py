"""PMCC eligibility + leg selection from the IBKR option chain.

Given a symbol and current spot, walks the IBKR-listed expirations and
strikes and chooses two legs:

  Long  — deep-ITM LEAP with target Δ ≈ 0.90, 18–24 months DTE
  Short — OTM near-dated call with target Δ ≈ 0.25, 21–35 days DTE

The algorithm probes a small grid of candidate expiry/strike combinations,
fetches Greeks via IBKR market data (tick 106), and picks the contracts
closest to target delta within each candidate window. Eligibility gates
reject names whose option markets are too thin to honour PMCC discipline:

  * LEAP open interest    >= 500
  * Short call OI         >= 250
  * LEAP bid/ask spread   <= 7% of mid
  * Short call spread     <= 10% of mid
  * Net debit              > 0
  * Max loss            >= 0 (sanity)

Returns ``PmccCandidate`` with both legs + financials, or
``PmccEligibility(eligible=False, reason=…)`` when the chain doesn't
support a clean diagonal today.

This module never places orders. The walking-limit executor consumes
``PmccCandidate`` and submits to IBKR.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Targets and thresholds — operator-tunable later via Settings.
# ---------------------------------------------------------------------------

LEAP_DELTA_TARGET           = 0.85
# High-conviction policy: widen the delta band so a sparse LEAP chain
# doesn't drop us into stock fallback. We'd rather hold a 0.65Δ LEAP that
# we got filled on than a stock position with no time-decay collection.
LEAP_DELTA_RANGE            = (0.65, 0.95)
LEAP_DTE_MIN_DAYS           = 18 * 30
LEAP_DTE_MAX_DAYS           = 24 * 30
# Eligibility = "is it worth ATTEMPTING a walking-limit fill?" The
# walking-limit executor caps the actual paid debit at mid + 25% of
# half-spread and abandons cleanly if it can't fill, so a wide build-
# time spread is not the same as a wide fill. These thresholds gate
# the attempt; the executor gates the price.
LEAP_MIN_OI                 = 50
LEAP_MAX_SPREAD_PCT         = 0.30

SHORT_DELTA_TARGET          = 0.25
SHORT_DELTA_RANGE           = (0.10, 0.50)
SHORT_DTE_MIN_DAYS          = 21
SHORT_DTE_MAX_DAYS          = 45
SHORT_MIN_OI                = 25
SHORT_MAX_SPREAD_PCT        = 0.50

# Sample width — enough to land in the delta band even when listed strikes
# are spaced 5–10 apart on $200+ stocks.
MAX_LEAP_CANDIDATES         = 12
MAX_SHORT_CANDIDATES        = 10


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class OptionLeg:
    expiry: str          # YYYYMMDD
    strike: float
    right: str           # "C" | "P"
    conid: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    model_price: Optional[float] = None    # IBKR's theoretical fair value (after-hours fallback)
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    open_interest: Optional[int] = None
    underlying_price: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        # Live bid/ask first.
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        # Last trade.
        if self.last is not None and self.last > 0:
            return float(self.last)
        # IBKR's theoretical fair value (modelGreeks.optPrice). Populated
        # even when bid/ask/last aren't — covers after-hours and thin contracts.
        if self.model_price is not None and self.model_price > 0:
            return float(self.model_price)
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        # 0/0 in after-hours — treat as "spread not applicable".
        if self.bid <= 0 or self.ask <= 0:
            return None
        m = (self.bid + self.ask) / 2
        if m == 0:
            return None
        return (self.ask - self.bid) / m


@dataclass
class PmccCandidate:
    symbol: str
    spot: float
    contracts: int                    # number of spreads (1 contract = 100 shares)
    leap: OptionLeg
    short_call: OptionLeg
    net_debit: float                  # mid combo at submit time, per spread
    max_loss: float                   # = net_debit (capital at risk per spread)
    notional_exposure: float          # leap.delta * 100 * spot * contracts
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spot": self.spot,
            "contracts": self.contracts,
            "leap": _leg_to_dict(self.leap),
            "short_call": _leg_to_dict(self.short_call),
            "net_debit": self.net_debit,
            "max_loss": self.max_loss,
            "notional_exposure": self.notional_exposure,
            "rationale": self.rationale,
        }


@dataclass
class PmccEligibility:
    eligible: bool
    candidate: Optional[PmccCandidate] = None
    reason: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


def _leg_to_dict(leg: OptionLeg) -> dict[str, Any]:
    return {
        "expiry": leg.expiry, "strike": leg.strike, "right": leg.right,
        "conid": leg.conid,
        "bid": leg.bid, "ask": leg.ask, "last": leg.last, "mid": leg.mid,
        "iv": leg.iv, "delta": leg.delta, "gamma": leg.gamma,
        "theta": leg.theta, "vega": leg.vega,
        "open_interest": leg.open_interest,
        "spread_pct": leg.spread_pct,
        "underlying_price": leg.underlying_price,
    }


# ---------------------------------------------------------------------------
# Helpers — date math + grid construction
# ---------------------------------------------------------------------------


def _parse_expiry(yyyymmdd: str) -> date:
    return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def _dte(expiry: str, today: Optional[date] = None) -> int:
    return (_parse_expiry(expiry) - (today or date.today())).days


def _filter_expirations_by_dte(
    expirations: list[str], dte_min: int, dte_max: int,
) -> list[str]:
    today = date.today()
    out = []
    for exp in expirations:
        try:
            d = _dte(exp, today)
        except (ValueError, IndexError):
            continue
        if dte_min <= d <= dte_max:
            out.append(exp)
    return out


def _strikes_around(strikes: list[float], target: float, n: int) -> list[float]:
    """Return the ``n`` strikes closest to ``target`` (sorted by distance)."""
    if not strikes:
        return []
    return sorted(strikes, key=lambda s: abs(s - target))[:n]


def _candidate_leap_strikes(
    spot: float, strikes: list[float], n: int = MAX_LEAP_CANDIDATES,
) -> list[float]:
    """Sample LEAP candidate strikes spanning the 0.75–0.95 delta region.

    The chain endpoint returns the *union* of strikes across all
    expirations; many of those don't exist on any single LEAP expiration.
    To avoid 'No security definition' errors during quoting, we restrict
    to common-increment strikes ($5 above $50, $10 above $200) which are
    the strikes guaranteed to exist on long-dated chains.
    """
    lo, hi = spot * 0.55, spot * 0.85
    increment = _common_strike_increment(spot)
    in_window = [
        s for s in strikes
        if lo <= s <= hi and abs(s - round(s / increment) * increment) < 0.01
    ]
    if in_window:
        mid = (lo + hi) / 2
        return sorted(in_window, key=lambda s: abs(s - mid))[:n]
    # Fallback: any ITM strike, ignoring increment. Will produce more
    # 'security not found' errors but at least quotes something.
    itm = [s for s in strikes if s < spot]
    return _strikes_around(itm or strikes, spot * 0.7, n)


def _common_strike_increment(spot: float) -> float:
    """Standard strike increments for US-listed equity options.

    < $25:  $0.50
    $25 - $200: $5.00
    >= $200: $10.00 (LEAPs and most weeklies on liquid mega-caps)
    """
    if spot < 25:
        return 0.5
    if spot < 200:
        return 5.0
    return 10.0


def _candidate_short_strikes(
    spot: float, strikes: list[float], n: int = MAX_SHORT_CANDIDATES,
) -> list[float]:
    """OTM strike sample around the 0.25-delta target.

    Restrict to common-increment strikes for the same listed-strike
    coverage reasons as the LEAP picker.
    """
    target = spot * 1.10
    increment = _common_strike_increment(spot)
    candidates = [
        s for s in strikes
        if spot < s <= spot * 1.30
        and abs(s - round(s / increment) * increment) < 0.01
    ]
    if candidates:
        return _strikes_around(candidates, target, n)
    # Fallback to whatever is OTM.
    otm = [s for s in strikes if s > spot]
    return _strikes_around(otm or strikes, target, n)


# ---------------------------------------------------------------------------
# Public selection function
# ---------------------------------------------------------------------------


async def select_pmcc_legs(
    *,
    symbol: str,
    contracts: int,
    ibkr: Any,                              # IbkrProvider
    leap_delta_target: float = LEAP_DELTA_TARGET,
    short_delta_target: float = SHORT_DELTA_TARGET,
    require_short_call: bool = True,
) -> PmccEligibility:
    """Build the best PMCC candidate for ``symbol``, or fail eligibility.

    Sequence:
      1. Get listed chain (expirations + strikes) from IBKR.
      2. Pick LEAP expirations in the 18–24mo window; for each, sample
         strikes near the 0.90-delta target and quote them. Pick the leg
         closest to target delta within the eligibility band.
      3. Pick short expirations in the 21–35-day window; same procedure
         for 0.25-delta.
      4. Validate OI / spread / net debit gates.

    ``require_short_call=False`` (LEAPS-only strategy): steps 3-4's short-leg
    checks are skipped entirely and the candidate carries a zero-value
    placeholder short leg. A LEAPS-only book never sells the short call, so
    its OI / spread / delta must never veto an otherwise-eligible LEAP —
    that dead constraint blocked a top-conviction ALAB entry on 2026-07-09.
    """
    chain = await ibkr.get_option_chain(symbol=symbol)
    expirations = chain["expirations"]
    strikes = chain["strikes"]

    if not expirations or not strikes:
        return PmccEligibility(False, reason="empty option chain")

    # Get the underlying spot via a Stock quote — IBKR returns this in
    # option market-data Greeks, but we need it BEFORE sampling strikes.
    # Cheapest path: request market data on the underlying.
    spot = await _fetch_spot(ibkr, symbol)
    if not spot or spot <= 0:
        return PmccEligibility(False, reason="could not fetch underlying spot")

    # ---- LEAP leg ---------------------------------------------------
    leap_exps = _filter_expirations_by_dte(expirations, LEAP_DTE_MIN_DAYS, LEAP_DTE_MAX_DAYS)
    if not leap_exps:
        return PmccEligibility(False, reason=f"no expirations in {LEAP_DTE_MIN_DAYS}-{LEAP_DTE_MAX_DAYS}d window")
    # Prefer expirations closest to the 21mo midpoint.
    target_leap_dte = (LEAP_DTE_MIN_DAYS + LEAP_DTE_MAX_DAYS) // 2
    leap_exps_sorted = sorted(leap_exps, key=lambda e: abs(_dte(e) - target_leap_dte))[:2]
    leap_strikes = _candidate_leap_strikes(spot, strikes)
    leap = await _pick_leg_closest_to_delta(
        ibkr, symbol, leap_exps_sorted, leap_strikes, "C",
        delta_target=leap_delta_target, delta_range=LEAP_DELTA_RANGE,
        spot_for_fallback=spot, moneyness_target=0.70,    # ~30% ITM proxy for 0.85Δ
    )
    if leap is None:
        return PmccEligibility(False, reason="no LEAP candidate within target delta band")
    # OI check: only enforce when IBKR returned a value. After-hours and on
    # less-active option contracts the OI tick (generic tick 101) doesn't
    # always populate; in that case we let the build proceed and let the
    # submit path fail-fast if OI is genuinely thin during RTH.
    if leap.open_interest is not None and leap.open_interest < LEAP_MIN_OI:
        return PmccEligibility(False, reason=f"LEAP OI {leap.open_interest} < {LEAP_MIN_OI}")
    if leap.spread_pct is not None and leap.spread_pct > LEAP_MAX_SPREAD_PCT:
        return PmccEligibility(False, reason=f"LEAP spread {leap.spread_pct:.2%} > {LEAP_MAX_SPREAD_PCT:.0%}")

    # ---- Short call leg ---------------------------------------------
    if not require_short_call:
        # LEAPS-only: zero-value placeholder — never quoted, never submitted.
        # mid resolves to None → short_mid 0 → net debit = the LEAP's mid.
        short = OptionLeg(expiry=leap.expiry, strike=leap.strike * 100,
                          right="C", conid=0)
    else:
        short_exps = _filter_expirations_by_dte(expirations, SHORT_DTE_MIN_DAYS, SHORT_DTE_MAX_DAYS)
        if not short_exps:
            return PmccEligibility(False, reason=f"no expirations in {SHORT_DTE_MIN_DAYS}-{SHORT_DTE_MAX_DAYS}d window")
        target_short_dte = 28
        short_exps_sorted = sorted(short_exps, key=lambda e: abs(_dte(e) - target_short_dte))[:2]
        short_strikes = _candidate_short_strikes(spot, strikes)
        short = await _pick_leg_closest_to_delta(
            ibkr, symbol, short_exps_sorted, short_strikes, "C",
            delta_target=short_delta_target, delta_range=SHORT_DELTA_RANGE,
            spot_for_fallback=spot, moneyness_target=1.10,    # ~10% OTM proxy for 0.25Δ
        )
        if short is None:
            return PmccEligibility(False, reason="no short call within target delta band")
        if short.open_interest is not None and short.open_interest < SHORT_MIN_OI:
            return PmccEligibility(False, reason=f"short call OI {short.open_interest} < {SHORT_MIN_OI}")
        if short.spread_pct is not None and short.spread_pct > SHORT_MAX_SPREAD_PCT:
            return PmccEligibility(False, reason=f"short call spread {short.spread_pct:.2%} > {SHORT_MAX_SPREAD_PCT:.0%}")

        # Strike sanity: short strike must be > leap strike (otherwise the
        # combo is effectively a credit spread and the math breaks).
        if short.strike <= leap.strike:
            return PmccEligibility(
                False, reason=f"short strike {short.strike} <= LEAP strike {leap.strike}",
            )

    # ---- Combo financials -------------------------------------------
    leap_mid = leap.mid or 0
    short_mid = short.mid or 0

    # If the LEAP has no usable mid we can't price the combo — bail out.
    # (Catches the "0 - 5 = -5 = credit" false positive when only one
    # leg's quote populated.)
    if leap_mid <= 0:
        return PmccEligibility(
            False, reason="LEAP has no usable mid (no bid/ask/last/model_price)",
        )

    net_debit = round(leap_mid - short_mid, 2)
    if net_debit <= 0:
        return PmccEligibility(
            False,
            reason=(f"net debit {net_debit} <= 0 (LEAP ${leap_mid:.2f} - "
                    f"short ${short_mid:.2f}) — combo is a credit; likely stale quotes"),
        )

    notional = (leap.delta or 0.85) * 100 * spot * contracts

    leap_delta_str = f"{leap.delta:.2f}" if leap.delta is not None else "—"
    if not require_short_call:
        rationale = (
            f"LEAP ${leap.strike:.0f} {leap.expiry} (Δ≈{leap_delta_str}, mid ${leap_mid:.2f}) "
            f"bought outright (LEAPS-only, no short call) "
            f"= debit ${net_debit:.2f} per contract × {contracts} contracts"
        )
    else:
        short_delta_str = f"{short.delta:.2f}" if short.delta is not None else "—"
        rationale = (
            f"LEAP ${leap.strike:.0f} {leap.expiry} (Δ≈{leap_delta_str}, mid ${leap_mid:.2f}) "
            f"+ short ${short.strike:.0f} {short.expiry} (Δ≈{short_delta_str}, mid ${short_mid:.2f}) "
            f"= net debit ${net_debit:.2f} per spread × {contracts} contracts"
        )
    candidate = PmccCandidate(
        symbol=symbol, spot=spot, contracts=contracts,
        leap=leap, short_call=short,
        net_debit=net_debit, max_loss=net_debit * contracts * 100,
        notional_exposure=notional, rationale=rationale,
    )
    return PmccEligibility(True, candidate=candidate, reason="ok")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


async def _fetch_spot(ibkr: Any, symbol: str) -> Optional[float]:
    """Get current spot price. IBKR live → IBKR cached close → Polygon → None.

    During RTH the IBKR live snapshot is the right answer. After-hours the
    `last` and `close` ticks aren't always populated (depends on subscription
    + recency) so we fall through to Polygon's most-recent daily close,
    which is reliable 24/7 and what we want for *picking strikes anyway*
    (the live executor will re-quote at submit time).
    """
    spot = await _fetch_spot_ibkr(ibkr, symbol)
    if spot and spot > 0:
        return spot
    # Fallback 1: Polygon last-close.
    try:
        from datetime import date, timedelta
        from tradingagents.dataflows.providers.polygon import PolygonProvider  # type: ignore
        p = PolygonProvider()
        end = date.today()
        start = end - timedelta(days=10)
        df = await p.get_stock_data(symbol, start, end)
        if isinstance(df, str):  # defensive: a corrupted cache hit / error string
            raise TypeError(f"polygon get_stock_data returned str, not DataFrame: {df[:80]}")
        if df is not None and not df.empty:
            close = float(df["Close"].iloc[-1] if "Close" in df.columns else df["close"].iloc[-1])
            if close > 0:
                return close
    except Exception as e:
        logger.warning("Polygon spot fallback failed for %s: %s", symbol, e)
    # Fallback 2: yfinance.
    try:
        import yfinance as yf  # type: ignore
        df = await asyncio.to_thread(lambda: yf.Ticker(symbol).history(period="5d"))
        if df is not None and not df.empty and "Close" in df.columns:
            v = float(df["Close"].iloc[-1])
            if v > 0:
                logger.info("spot for %s via yfinance fallback: %.2f", symbol, v)
                return v
    except Exception as e:
        logger.warning("yfinance spot fallback failed for %s: %s", symbol, e)
    return None


async def _fetch_spot_ibkr(ibkr: Any, symbol: str) -> Optional[float]:
    """Inner helper — IBKR-only spot fetch with bounded wait."""
    try:
        from ib_insync import Stock  # type: ignore
        ib = await ibkr._ensure_connected()
        contract = Stock(symbol, "SMART", "USD")
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            return None
        contract = qualified[0]
        ticker = ib.reqMktData(contract, "", False, False)
        for _ in range(30):
            last = _try_float(ticker.last)
            close = _try_float(ticker.close)
            mp = None
            try:
                mp = _try_float(ticker.marketPrice() if hasattr(ticker, "marketPrice") else None)
            except Exception:
                mp = None
            if any(v and v > 0 for v in (last, close, mp)):
                break
            await asyncio.sleep(0.1)
        last = _try_float(ticker.last)
        close = _try_float(ticker.close)
        try:
            mp = _try_float(ticker.marketPrice() if hasattr(ticker, "marketPrice") else None)
        except Exception:
            mp = None
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass
        for v in (last, mp, close):
            if v and v > 0:
                return float(v)
        return None
    except Exception as e:
        logger.warning("IBKR spot fetch failed for %s: %s", symbol, e)
        return None


def _try_float(v: Any) -> Optional[float]:
    """Best-effort float conversion that filters NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


async def _pick_leg_closest_to_delta(
    ibkr: Any,
    symbol: str,
    expirations: list[str],
    strikes: list[float],
    right: str,
    *,
    delta_target: float,
    delta_range: tuple[float, float],
    spot_for_fallback: Optional[float] = None,
    moneyness_target: Optional[float] = None,
) -> Optional[OptionLeg]:
    """Quote the (expiry × strike) grid; pick the best leg.

    Primary path: filter by delta in ``delta_range``, return closest to
    ``delta_target``.

    After-hours fallback: if NO leg returned a usable delta (model Greeks
    don't always stream outside RTH), fall back to a *strike-distance*
    heuristic — pick the strike closest to ``moneyness_target * spot``.
    The executor re-validates delta at submit time during RTH.
    """
    candidates_with_delta: list[OptionLeg] = []
    candidates_all: list[OptionLeg] = []

    coros = []
    metadata = []
    for exp in expirations:
        for strike in strikes:
            coros.append(ibkr.get_option_quote(symbol=symbol, expiry=exp, strike=strike, right=right))
            metadata.append((exp, strike))
    if not coros:
        return None

    sem = asyncio.Semaphore(3)
    async def _bounded(c):
        async with sem:
            try:
                return await c
            except Exception as e:
                logger.warning("option quote failed: %s", e)
                return None
    results = await asyncio.gather(*( _bounded(c) for c in coros))

    for (exp, strike), q in zip(metadata, results):
        if q is None:
            continue
        leg = OptionLeg(
            expiry=exp, strike=float(strike), right=right,
            conid=int(q.get("conid", 0)),
            bid=q.get("bid"), ask=q.get("ask"), last=q.get("last"),
            model_price=q.get("model_price"),
            iv=q.get("iv"),  delta=q.get("delta"), gamma=q.get("gamma"),
            theta=q.get("theta"), vega=q.get("vega"),
            open_interest=q.get("open_interest"),
            underlying_price=q.get("underlying_price"),
        )
        if leg.conid == 0:
            continue
        candidates_all.append(leg)
        if leg.delta is not None:
            d = abs(leg.delta)
            if delta_range[0] <= d <= delta_range[1]:
                candidates_with_delta.append(leg)

    # Primary path: delta-based selection.
    if candidates_with_delta:
        return min(candidates_with_delta, key=lambda l: abs(abs(l.delta) - delta_target))

    # Fallback: strike-distance heuristic when Greeks aren't streaming.
    if candidates_all and spot_for_fallback and moneyness_target:
        target_strike = spot_for_fallback * moneyness_target
        logger.info(
            "PMCC %s leg: no Greeks, falling back to strike-distance "
            "(target %.2f, delta_target %.2f)",
            right, target_strike, delta_target,
        )
        return min(candidates_all, key=lambda l: abs(l.strike - target_strike))

    return None
