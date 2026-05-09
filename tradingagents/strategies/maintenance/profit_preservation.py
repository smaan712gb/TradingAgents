"""Profit-preservation trim ladder.

Implements the operator's exit-strategy spec for taking profits without
closing entire positions. Trims rather than full exits — the goal is to
recover capital and create dry powder for the next setup, not to abandon
a winner that the chokepoint thesis still supports.

Stocks:
  +50%  → trim 10% on a strong green day (volume + RSI confirmation)
  +100% → trim 20% if the theme is still hot, 30-50% if weakening
  +200% → recover original capital (≈ trim 33%); let the rest ride
          if theme + fundamentals stay strong

LEAPS (PMCC long-leg, with proportional short-call buyback):
  +50%  → trim 10-15% on a strong day
  +100% → trim 20-30%, even if theme still hot (creates dry powder)
  +200% → recover original capital, rest is house money

The "strong day" gate prevents trimming on quiet drift. The "theme hot"
modifier (filled by the Theme Health module in Phase C) prevents
prematurely cutting a winner that the framework still favours.

This module returns ``TrimDecision``; the maintenance loop applies the
trim via the partial-close primitive (single-leg sell for stocks,
walking-limit combo for PMCCs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trim ladder thresholds — tunable; defaults match the operator spec.
# ---------------------------------------------------------------------------

# Stocks
STOCK_BAND_50_TRIM_PCT      = 0.10      # +50% gain → trim 10%
STOCK_BAND_100_HOT_PCT      = 0.20      # +100% with hot theme → trim 20%
STOCK_BAND_100_WEAK_PCT     = 0.40      # +100% with weakening theme → trim 40%
STOCK_BAND_200_TRIM_PCT     = 0.33      # +200% → trim 33% (recovers original capital)

# LEAPS (PMCC long-leg)
LEAP_BAND_50_TRIM_PCT       = 0.12      # +50% → trim 12%
LEAP_BAND_100_TRIM_PCT      = 0.25      # +100% → trim 25% (regardless of theme)
LEAP_BAND_200_TRIM_PCT      = 0.33      # +200% → trim 33% (recovers original capital)

# Strong-day gate
STRONG_DAY_PCT_MOVE         = 0.03      # +3% today
STRONG_DAY_VOLUME_RATIO     = 1.5       # 1.5× 20d avg
STRONG_DAY_RSI_OVERBOUGHT   = 70.0      # RSI >= 70 = overbought
STRONG_DAY_MIN_SIGNALS      = 2         # need 2-of-3 to call it "strong"


@dataclass
class TrimDecision:
    should_trim: bool = False
    trim_pct: float = 0.0
    reason: str = ""
    band: Literal["50pct", "100pct", "200pct", "none"] = "none"
    instrument: Literal["stock", "leap", "none"] = "none"
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gain_pct(cost: float, current: float) -> float:
    if cost <= 0:
        return 0.0
    return (current - cost) / cost


def is_strong_day(
    *,
    pct_move_today: Optional[float],
    volume_ratio_vs_20d: Optional[float],
    rsi_14: Optional[float],
) -> bool:
    """At least 2 of: +3%+ today, 1.5× avg volume, RSI ≥ 70."""
    score = 0
    if pct_move_today is not None and pct_move_today >= STRONG_DAY_PCT_MOVE:
        score += 1
    if volume_ratio_vs_20d is not None and volume_ratio_vs_20d >= STRONG_DAY_VOLUME_RATIO:
        score += 1
    if rsi_14 is not None and rsi_14 >= STRONG_DAY_RSI_OVERBOUGHT:
        score += 1
    return score >= STRONG_DAY_MIN_SIGNALS


# ---------------------------------------------------------------------------
# Stock trim
# ---------------------------------------------------------------------------


async def evaluate_stock_trim(
    *,
    symbol: str,
    avg_price: float,
    current_price: float,
    pct_move_today: Optional[float] = None,
    volume_ratio_vs_20d: Optional[float] = None,
    rsi_14: Optional[float] = None,
    theme_hot: bool = True,
    already_trimmed_today: bool = False,
) -> TrimDecision:
    """Profit-preservation trim decision for an equity position.

    The strong-day gate fires only above the 50% gain band — at 200% we
    take capital back regardless of intraday tape.
    """
    if already_trimmed_today:
        return TrimDecision(reason=f"{symbol}: already trimmed today")

    gain = _gain_pct(avg_price, current_price)

    # +200% band — return original capital, no strong-day gate (urgency).
    if gain >= 2.0:
        return TrimDecision(
            should_trim=True, trim_pct=STOCK_BAND_200_TRIM_PCT,
            band="200pct", instrument="stock",
            reason=(f"{symbol} up {gain*100:.0f}% — recover original capital "
                    f"(trim {STOCK_BAND_200_TRIM_PCT*100:.0f}%)"
                    + ("" if theme_hot else "; theme weakening, consider larger trim")),
            detail={"gain_pct": gain, "theme_hot": theme_hot},
        )

    # +100% band — strong day required; theme-aware sizing.
    if gain >= 1.0:
        if not is_strong_day(pct_move_today=pct_move_today,
                             volume_ratio_vs_20d=volume_ratio_vs_20d,
                             rsi_14=rsi_14):
            return TrimDecision(
                reason=f"{symbol} up {gain*100:.0f}% but tape isn't strong; wait",
                detail={"gain_pct": gain},
            )
        trim_pct = STOCK_BAND_100_HOT_PCT if theme_hot else STOCK_BAND_100_WEAK_PCT
        return TrimDecision(
            should_trim=True, trim_pct=trim_pct,
            band="100pct", instrument="stock",
            reason=(f"{symbol} up {gain*100:.0f}% on strong day — trim "
                    f"{trim_pct*100:.0f}% "
                    f"({'theme hot' if theme_hot else 'theme weakening'})"),
            detail={"gain_pct": gain, "theme_hot": theme_hot},
        )

    # +50% band — strong day required; small trim.
    if gain >= 0.50:
        if not is_strong_day(pct_move_today=pct_move_today,
                             volume_ratio_vs_20d=volume_ratio_vs_20d,
                             rsi_14=rsi_14):
            return TrimDecision(
                reason=f"{symbol} up {gain*100:.0f}% but tape isn't strong; hold",
                detail={"gain_pct": gain},
            )
        return TrimDecision(
            should_trim=True, trim_pct=STOCK_BAND_50_TRIM_PCT,
            band="50pct", instrument="stock",
            reason=(f"{symbol} up {gain*100:.0f}% on strong day — "
                    f"trim {STOCK_BAND_50_TRIM_PCT*100:.0f}%"),
            detail={"gain_pct": gain},
        )

    return TrimDecision(
        reason=f"{symbol}: gain {gain*100:.0f}% below 50% threshold",
        detail={"gain_pct": gain},
    )


# ---------------------------------------------------------------------------
# LEAPS trim (long-leg of PMCC)
# ---------------------------------------------------------------------------


async def evaluate_leap_trim(
    *,
    symbol: str,
    leap_entry_debit: float,        # original net-debit per spread
    current_combo_mid: float,       # current net-debit (positive) the combo would close at
    pct_move_today: Optional[float] = None,
    volume_ratio_vs_20d: Optional[float] = None,
    rsi_14: Optional[float] = None,
    theme_hot: bool = True,
    already_trimmed_today: bool = False,
) -> TrimDecision:
    """Profit-preservation trim decision for the LEAP long-leg of a PMCC.

    Gain is measured on the *combo* — current closing net-debit vs the
    original entry debit. A combo bought for $25 that's now worth $60
    is "up 140%" on the combo basis, which is what the operator
    thinks about when they decide to take profits.
    """
    if already_trimmed_today:
        return TrimDecision(reason=f"{symbol}: PMCC already trimmed today")

    gain = _gain_pct(leap_entry_debit, current_combo_mid)

    # +200% — recover capital, no strong-day gate.
    if gain >= 2.0:
        return TrimDecision(
            should_trim=True, trim_pct=LEAP_BAND_200_TRIM_PCT,
            band="200pct", instrument="leap",
            reason=(f"{symbol} PMCC up {gain*100:.0f}% — recover original "
                    f"capital (trim {LEAP_BAND_200_TRIM_PCT*100:.0f}% "
                    f"of LEAP, buy-back same %% of short)"),
            detail={"gain_pct": gain, "theme_hot": theme_hot},
        )

    # +100% — trim regardless of theme (creates dry powder).
    if gain >= 1.0:
        if not is_strong_day(pct_move_today=pct_move_today,
                             volume_ratio_vs_20d=volume_ratio_vs_20d,
                             rsi_14=rsi_14):
            return TrimDecision(
                reason=f"{symbol} PMCC up {gain*100:.0f}% but tape isn't strong; wait",
                detail={"gain_pct": gain},
            )
        return TrimDecision(
            should_trim=True, trim_pct=LEAP_BAND_100_TRIM_PCT,
            band="100pct", instrument="leap",
            reason=(f"{symbol} PMCC up {gain*100:.0f}% on strong day — "
                    f"trim {LEAP_BAND_100_TRIM_PCT*100:.0f}% (creates dry powder)"),
            detail={"gain_pct": gain, "theme_hot": theme_hot},
        )

    # +50% — small trim on strong-day signal.
    if gain >= 0.50:
        if not is_strong_day(pct_move_today=pct_move_today,
                             volume_ratio_vs_20d=volume_ratio_vs_20d,
                             rsi_14=rsi_14):
            return TrimDecision(
                reason=f"{symbol} PMCC up {gain*100:.0f}% but tape isn't strong; hold",
                detail={"gain_pct": gain},
            )
        return TrimDecision(
            should_trim=True, trim_pct=LEAP_BAND_50_TRIM_PCT,
            band="50pct", instrument="leap",
            reason=(f"{symbol} PMCC up {gain*100:.0f}% on strong day — "
                    f"trim {LEAP_BAND_50_TRIM_PCT*100:.0f}%"),
            detail={"gain_pct": gain},
        )

    return TrimDecision(
        reason=f"{symbol} PMCC: gain {gain*100:.0f}% below 50% threshold",
        detail={"gain_pct": gain},
    )
