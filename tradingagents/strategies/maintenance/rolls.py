"""Roll logic for PMCC positions — operator-specified mechanics.

Short-call rolls follow these rules:

  1. Defensive trigger: short call delta in 0.65-0.75 → evaluate roll.
     A short call ITM-and-rising is capping our LEAP and threatens
     assignment. We don't wait for delta=1.0; 0.70 is the action level.

  2. Time roll: DTE ≤ 7 with short still OTM (delta < 0.30) → roll to
     capture the next theta cycle without paying for the existing one.

  3. Profit roll: short price ≤ 20% of original credit → close-and-redeploy.

When rolling:
  * Roll out 2-4 weeks (14-28 DTE)
  * Roll up to a strike *above* the 1-SD expected move from current spot,
    bounded below by the recent high (technical resistance proxy)
  * Strong-momentum mode: target 0.15 delta (further OTM) instead of 0.25,
    so we don't cap the LEAP too tightly during a runup
  * Refuse the roll if the net debit > 30% of the original credit collected
    — the trade is no longer math-positive at that point

LEAP forward roll:
  * Trigger at DTE ≤ 180 (6 months — theta decay accelerates here)
  * Don't roll if LEAP delta has decayed below 0.65 — close instead
  * Force-close at DTE ≤ 90 to avoid the gamma cliff
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

from tradingagents.strategies.pmcc import (
    OptionLeg,
    _candidate_short_strikes,
    _common_strike_increment,
    _filter_expirations_by_dte,
    _pick_leg_closest_to_delta,
    SHORT_DELTA_RANGE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Operator-tuned thresholds
# ---------------------------------------------------------------------------

# Short call rolls
SHORT_DTE_ROLL_TRIGGER          = 7
SHORT_DELTA_DEFENSIVE           = 0.70    # 0.65-0.75 band; midpoint
SHORT_PROFIT_PCT_FOR_ROLL       = 0.80

# Replacement contract picking
SHORT_NEW_DTE_MIN               = 14      # 2 weeks
SHORT_NEW_DTE_MAX               = 28      # 4 weeks
SHORT_TARGET_DELTA_NORMAL       = 0.25
SHORT_TARGET_DELTA_MOMENTUM     = 0.15    # further OTM during strong momentum
SHORT_TARGET_DELTA_BAND_MOM     = (0.10, 0.25)
EXPECTED_MOVE_BUFFER            = 1.5     # roll strike must be ≥ spot + 1.5 SD
RECENT_HIGH_BUFFER_PCT          = 0.05    # also ≥ recent_high × 1.05

# Cost guard — refuse rolls that aren't math-positive
ROLL_DEBIT_MAX_PCT_OF_CREDIT    = 0.30

# LEAP rolls
LEAP_DTE_ROLL_TRIGGER           = 180     # 6 months — theta decay accelerates
LEAP_DTE_FORCE_CLOSE            = 90      # 3 months — gamma cliff
LEAP_DELTA_THESIS_BREAK         = 0.65    # below: thesis broken, don't roll

# Momentum detection
MOMENTUM_20D_RETURN_THRESHOLD   = 0.15    # 15% in 20 sessions = strong
MOMENTUM_NEW_HIGH_DAYS          = 5       # within 5 days of 20-day high


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RollDecision:
    should_roll: bool
    reason: str = ""
    new_leg: Optional[OptionLeg] = None
    estimated_net_debit: Optional[float] = None
    estimated_credit_capture: Optional[float] = None
    skip_short_until_event: Optional[str] = None  # "earnings" | "momentum" — short stays closed
    detail: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Public — short-call roll
# ---------------------------------------------------------------------------


async def maybe_roll_short_call(
    *,
    symbol: str,
    short_expiry: str,            # YYYYMMDD currently held
    short_strike: float,
    current_short_delta: Optional[float],
    current_short_mid: Optional[float],
    open_credit: Optional[float],
    underlying_spot: float,
    underlying_iv: Optional[float],   # short-leg IV from quote, used for expected-move math
    chain_strikes: list[float],
    chain_expirations: list[str],
    ibkr: Any,
    days_to_earnings: Optional[int] = None,
) -> RollDecision:
    """Decide if/how to roll the short call. Returns a fully-priced RollDecision."""
    cur_dte = _safe_dte(short_expiry)
    short_delta_abs = abs(current_short_delta or 0)

    # ---- Earnings: don't sell into print; close-only -----------------
    if days_to_earnings is not None and 0 <= days_to_earnings <= 2:
        return RollDecision(
            should_roll=False,
            reason=(f"earnings in {days_to_earnings}d — close short call, "
                    f"do not roll into the print. Re-sell day after."),
            skip_short_until_event="earnings",
        )

    # ---- Defensive: short call ITM and rising ------------------------
    if short_delta_abs >= SHORT_DELTA_DEFENSIVE:
        reason = (f"defensive: short call delta {short_delta_abs:.2f} ≥ "
                  f"{SHORT_DELTA_DEFENSIVE} — capping the LEAP")
        return await _build_roll(
            symbol=symbol, reason=reason, urgency="defensive",
            chain_strikes=chain_strikes, chain_expirations=chain_expirations,
            underlying_spot=underlying_spot, underlying_iv=underlying_iv,
            current_short_strike=short_strike, current_short_mid=current_short_mid,
            open_credit=open_credit, ibkr=ibkr,
        )

    # ---- DTE expiring + still OTM -----------------------------------
    if cur_dte <= SHORT_DTE_ROLL_TRIGGER and short_delta_abs < 0.30:
        reason = (f"DTE {cur_dte} ≤ {SHORT_DTE_ROLL_TRIGGER} and OTM "
                  f"(Δ={short_delta_abs:.2f}) — capture next theta cycle")
        return await _build_roll(
            symbol=symbol, reason=reason, urgency="time",
            chain_strikes=chain_strikes, chain_expirations=chain_expirations,
            underlying_spot=underlying_spot, underlying_iv=underlying_iv,
            current_short_strike=short_strike, current_short_mid=current_short_mid,
            open_credit=open_credit, ibkr=ibkr,
        )

    # ---- 80% profit captured ----------------------------------------
    if open_credit and current_short_mid is not None and current_short_mid > 0:
        profit_pct = 1 - (current_short_mid / open_credit)
        if profit_pct >= SHORT_PROFIT_PCT_FOR_ROLL:
            reason = (f"profit {profit_pct:.0%} ≥ "
                      f"{SHORT_PROFIT_PCT_FOR_ROLL:.0%} of original credit")
            return await _build_roll(
                symbol=symbol, reason=reason, urgency="profit",
                chain_strikes=chain_strikes, chain_expirations=chain_expirations,
                underlying_spot=underlying_spot, underlying_iv=underlying_iv,
                current_short_strike=short_strike, current_short_mid=current_short_mid,
                open_credit=open_credit, ibkr=ibkr,
            )

    return RollDecision(
        should_roll=False,
        reason=f"hold (DTE={cur_dte}, Δ={short_delta_abs:.2f})",
    )


# ---------------------------------------------------------------------------
# Public — LEAP forward roll
# ---------------------------------------------------------------------------


async def maybe_roll_leap_forward(
    *,
    symbol: str,
    leap_expiry: str,
    leap_strike: float,
    current_leap_delta: Optional[float],
    underlying_spot: float,
    chain_strikes: list[float],
    chain_expirations: list[str],
    ibkr: Any,
) -> RollDecision:
    """Roll LEAP forward at 6-month DTE. Refuse if thesis broken."""
    leap_dte = _safe_dte(leap_expiry)
    leap_delta_abs = abs(current_leap_delta or 0)

    if leap_dte > LEAP_DTE_ROLL_TRIGGER:
        return RollDecision(False, reason=f"LEAP DTE {leap_dte} > {LEAP_DTE_ROLL_TRIGGER}; hold")

    if leap_dte <= LEAP_DTE_FORCE_CLOSE:
        return RollDecision(
            False,
            reason=(f"LEAP DTE {leap_dte} ≤ {LEAP_DTE_FORCE_CLOSE} — gamma cliff, "
                    f"close don't roll"),
            detail={"recommend_close": True},
        )

    if leap_delta_abs < LEAP_DELTA_THESIS_BREAK:
        return RollDecision(
            False,
            reason=(f"LEAP delta {leap_delta_abs:.2f} < {LEAP_DELTA_THESIS_BREAK} — "
                    f"thesis broken; close, don't extend"),
            detail={"recommend_close": True},
        )

    # Find a fresh 18-24 month LEAP at the original delta target.
    from tradingagents.strategies.pmcc import (
        LEAP_DTE_MIN_DAYS, LEAP_DTE_MAX_DAYS, LEAP_DELTA_TARGET, LEAP_DELTA_RANGE,
        _candidate_leap_strikes,
    )
    new_leap_exps = _filter_expirations_by_dte(
        chain_expirations, LEAP_DTE_MIN_DAYS, LEAP_DTE_MAX_DAYS,
    )
    if not new_leap_exps:
        return RollDecision(False, reason="no replacement LEAP expirations available")
    new_leap_strikes = _candidate_leap_strikes(underlying_spot, chain_strikes)
    new_leap = await _pick_leg_closest_to_delta(
        ibkr, symbol, sorted(new_leap_exps)[:2], new_leap_strikes, "C",
        delta_target=LEAP_DELTA_TARGET, delta_range=LEAP_DELTA_RANGE,
        spot_for_fallback=underlying_spot, moneyness_target=0.70,
    )
    if new_leap is None:
        return RollDecision(False, reason="no qualifying LEAP replacement found")
    return RollDecision(
        should_roll=True,
        reason=f"LEAP DTE {leap_dte} ≤ {LEAP_DTE_ROLL_TRIGGER} — extend forward",
        new_leg=new_leap,
        detail={"current_dte": leap_dte},
    )


# ---------------------------------------------------------------------------
# Internals — strike picking + expected move + momentum
# ---------------------------------------------------------------------------


async def _build_roll(
    *,
    symbol: str, reason: str, urgency: str,
    chain_strikes: list[float], chain_expirations: list[str],
    underlying_spot: float, underlying_iv: Optional[float],
    current_short_strike: float, current_short_mid: Optional[float],
    open_credit: Optional[float],
    ibkr: Any,
) -> RollDecision:
    """Pick the replacement short call honouring expected move + recent high
    + momentum-aware delta. Apply cost guard at the end."""

    # Filter to 14-28 DTE expirations
    new_exps = _filter_expirations_by_dte(
        chain_expirations, SHORT_NEW_DTE_MIN, SHORT_NEW_DTE_MAX,
    )
    if not new_exps:
        return RollDecision(False, reason=f"{reason} — no expirations in {SHORT_NEW_DTE_MIN}-{SHORT_NEW_DTE_MAX}d window")
    # Prefer ~21-DTE midpoint
    new_exps = sorted(new_exps, key=lambda e: abs(_safe_dte(e) - 21))[:2]
    target_dte_for_em = _safe_dte(new_exps[0]) or 21

    # Strike floor: max of (existing strike + 1 increment, spot+1.5SD,
    # recent high × 1.05). The new strike must clear all three.
    expected_move = _expected_move(
        spot=underlying_spot, iv=underlying_iv, dte=target_dte_for_em,
    )
    em_floor = underlying_spot + EXPECTED_MOVE_BUFFER * expected_move if expected_move else 0.0

    recent_high = await _recent_high(symbol, days=20)
    rh_floor = (recent_high * (1 + RECENT_HIGH_BUFFER_PCT)) if recent_high else 0.0

    increment = _common_strike_increment(underlying_spot)
    above_current = current_short_strike + increment

    strike_floor = max(em_floor, rh_floor, above_current)
    logger.info(
        "%s roll: spot=%.2f iv=%.2f em=%.2f recent_high=%.2f → strike_floor=%.2f",
        symbol, underlying_spot, underlying_iv or 0, expected_move or 0,
        recent_high or 0, strike_floor,
    )

    # Strong-momentum mode: target lower delta (further OTM) to avoid capping
    momentum = await _is_strong_momentum(symbol, recent_high=recent_high)
    target_delta = SHORT_TARGET_DELTA_MOMENTUM if momentum else SHORT_TARGET_DELTA_NORMAL
    delta_band = SHORT_TARGET_DELTA_BAND_MOM if momentum else SHORT_DELTA_RANGE

    # Filter chain strikes to the increment-aligned strikes above the floor
    candidate_strikes = [
        s for s in chain_strikes
        if s >= strike_floor
        and abs(s - round(s / increment) * increment) < 0.01
    ]
    if not candidate_strikes:
        # Fall back to any chain strikes above the floor
        candidate_strikes = [s for s in chain_strikes if s >= strike_floor]
    if not candidate_strikes:
        return RollDecision(
            False,
            reason=(f"{reason} — no listed strikes above floor "
                    f"${strike_floor:.2f} (em={expected_move:.2f if expected_move else 0}, "
                    f"recent_high={recent_high})"),
        )
    # Sample 8 closest to (spot × 1.10) within the candidate set
    candidate_strikes = sorted(
        candidate_strikes, key=lambda s: abs(s - underlying_spot * 1.10),
    )[:8]

    new_leg = await _pick_leg_closest_to_delta(
        ibkr, symbol, new_exps, candidate_strikes, "C",
        delta_target=target_delta, delta_range=delta_band,
        spot_for_fallback=underlying_spot, moneyness_target=1.10,
    )
    if new_leg is None:
        return RollDecision(
            False,
            reason=(f"{reason} — no replacement passed delta band {delta_band} "
                    f"above strike floor"),
        )

    # Cost math: new_credit (short proceeds) - close_cost (buying back current)
    new_credit = (new_leg.bid + new_leg.ask) / 2 if (new_leg.bid and new_leg.ask) else (new_leg.last or new_leg.model_price or 0)
    close_cost = current_short_mid or 0
    net = round(new_credit - close_cost, 2)   # positive = roll for credit, negative = pay debit

    # Cost guard
    if open_credit and net < 0:
        debit = abs(net)
        max_allowed_debit = open_credit * ROLL_DEBIT_MAX_PCT_OF_CREDIT
        if debit > max_allowed_debit:
            return RollDecision(
                False,
                reason=(f"{reason} — roll would cost ${debit:.2f}, exceeds "
                        f"{ROLL_DEBIT_MAX_PCT_OF_CREDIT*100:.0f}% of original credit "
                        f"${open_credit:.2f}. Skip; let short expire instead."),
                detail={"debit": debit, "max_allowed": max_allowed_debit},
            )

    return RollDecision(
        should_roll=True,
        reason=(f"{reason}. New strike ${new_leg.strike:.0f} {new_leg.expiry} "
                f"(Δ={(new_leg.delta or target_delta):.2f}, momentum={momentum}, "
                f"net {'credit' if net >= 0 else 'debit'} ${abs(net):.2f})"),
        new_leg=new_leg,
        estimated_net_debit=(-net if net < 0 else 0.0),
        estimated_credit_capture=(net if net > 0 else 0.0),
        detail={
            "urgency": urgency, "momentum": momentum,
            "strike_floor": strike_floor, "expected_move": expected_move,
            "recent_high": recent_high,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_move(*, spot: float, iv: Optional[float], dte: int) -> Optional[float]:
    """1-SD price move over ``dte`` days, in dollars. None if IV missing."""
    if not iv or iv <= 0 or dte <= 0:
        return None
    return spot * iv * math.sqrt(dte / 365.0)


async def _recent_high(symbol: str, *, days: int = 20) -> Optional[float]:
    """N-day high via Polygon. Used as a technical resistance proxy when
    rolling — never roll to a strike below recent resistance × 1.05."""
    try:
        from datetime import date, timedelta
        from tradingagents.dataflows.providers.polygon import PolygonProvider
        p = PolygonProvider()
        df = await p.get_stock_data(
            symbol, date.today() - timedelta(days=days * 2), date.today(),
        )
        if df is None or len(df) == 0 or "High" not in df.columns:
            return None
        return float(df["High"].astype(float).iloc[-days:].max())
    except Exception as e:
        logger.warning("recent_high fetch failed for %s: %s", symbol, e)
        return None


async def _is_strong_momentum(
    symbol: str, *, recent_high: Optional[float] = None,
) -> bool:
    """Strong momentum = (20d return ≥ 15%) OR (close within N days of 20d high)."""
    try:
        from datetime import date, timedelta
        from tradingagents.dataflows.providers.polygon import PolygonProvider
        p = PolygonProvider()
        df = await p.get_stock_data(
            symbol, date.today() - timedelta(days=45), date.today(),
        )
        if df is None or len(df) < 22 or "Close" not in df.columns or "High" not in df.columns:
            return False
        closes = df["Close"].astype(float).iloc[-22:].tolist()
        highs  = df["High"].astype(float).iloc[-22:].tolist()
        # 20-day return
        ret_20d = closes[-1] / closes[-21] - 1
        if ret_20d >= MOMENTUM_20D_RETURN_THRESHOLD:
            return True
        # New 20-day high within last N days
        twentyday_high = max(highs)
        for i in range(1, MOMENTUM_NEW_HIGH_DAYS + 1):
            if i <= len(highs) and highs[-i] >= twentyday_high - 0.005:
                return True
        return False
    except Exception as e:
        logger.warning("momentum check failed for %s: %s", symbol, e)
        return False


def _safe_dte(yyyymmdd: str) -> int:
    """Days until ``yyyymmdd``. Defaults far-out on parse failure so we
    don't spuriously trigger DTE-based rolls."""
    if not yyyymmdd or len(yyyymmdd) < 8:
        return 999
    try:
        return (date.fromisoformat(f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}")
                - date.today()).days
    except Exception:
        return 999
