"""Momentum exhaustion detector.

Implements the operator's "momentum exhaustion" rule from the exit
strategy. Trim signal fires when at least 50% of available indicators
trip simultaneously. Indicators currently wired:

  * Distance from 20-day moving average — > 25% = stretched
  * RSI(14) — > 75 = overbought
  * Volume ratio — > 2.5 × 20-day average = climactic
  * Gap-up today — open > prior close + 1 × ATR_30d = above expected move
  * Closing-auction sell imbalance — large negative imbalance late
    session ($2M+ paired sell pressure) on a stretched name = textbook
    distribution print. Only available 15:50-16:00 ET via IBKR's
    ``get_auction_imbalance``.
  * Insider selling acceleration — Form 4 sales by officers / directors /
    10%-owners over the last 30 days at >=2x the trailing 30-180 day
    monthly baseline, with at least two distinct sellers. FMP's
    ``get_insider_sell_pressure`` returns the pre-computed flag.
  * Analyst upgrade-after-run — at least one analyst upgrade in the
    last 7 days *while* the stock is up >=25% over 60 days. Late-cycle
    chase pattern. Pre-computed by ``analyst_grades.evaluate_analyst_pressure``.

Indicators in the operator spec that the system doesn't yet wire:
  * Short-term call IV at extreme percentile  — needs IV history
  * Social sentiment euphoria                 — needs sentiment feed

The decision gracefully ignores missing inputs; the threshold is "X of
Y available signals" rather than absolute count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tunables.
DISTANCE_MA20_TRIGGER         = 0.25        # +25% above 20d MA
RSI_OVERBOUGHT_TRIGGER        = 75.0
VOLUME_CLIMAX_TRIGGER         = 2.5         # 2.5× 20d avg
GAP_UP_ATR_MULTIPLIER         = 1.0         # gap > 1 × ATR_30d
AUCTION_IMBALANCE_USD         = 2_000_000   # |imbalance × auction_price| > $2M = significant
INSIDER_RATIO_TRIGGER         = 2.0         # 30d sales >= 2x baseline monthly avg
SIGNALS_REQUIRED_PCT          = 0.5         # 50%+ of available signals tripped


@dataclass
class ExhaustionDecision:
    exhausted: bool = False
    score: float = 0.0                      # 0.0-1.0, fraction of signals tripped
    signals_tripped: list[str] = field(default_factory=list)
    signals_available: list[str] = field(default_factory=list)
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def evaluate_momentum_exhaustion(
    *,
    symbol: str,
    current_price: float,
    ma_20d: Optional[float],
    rsi_14: Optional[float],
    volume_ratio: Optional[float],
    open_today: Optional[float] = None,
    prior_close: Optional[float] = None,
    atr_30d: Optional[float] = None,
    auction_imbalance: Optional[float] = None,    # signed; negative = sell pressure
    auction_price: Optional[float] = None,
    insider_pressure: Optional[dict] = None,      # FMP get_insider_sell_pressure result
    analyst_pressure: Optional[Any] = None,        # analyst_grades.GradePressure
) -> ExhaustionDecision:
    """Score momentum-exhaustion signals; trip when >=50% of available signals fire."""
    available: list[str] = []
    tripped: list[str] = []

    # Distance from 20d MA
    if ma_20d is not None and ma_20d > 0:
        available.append("distance_ma20")
        dist = (current_price - ma_20d) / ma_20d
        if dist > DISTANCE_MA20_TRIGGER:
            tripped.append(f"distance_ma20={dist:+.0%}")

    # RSI
    if rsi_14 is not None:
        available.append("rsi_14")
        if rsi_14 >= RSI_OVERBOUGHT_TRIGGER:
            tripped.append(f"rsi={rsi_14:.0f}")

    # Volume
    if volume_ratio is not None:
        available.append("volume_ratio")
        if volume_ratio >= VOLUME_CLIMAX_TRIGGER:
            tripped.append(f"volume={volume_ratio:.1f}x")

    # Gap-up
    if (open_today is not None and prior_close is not None and atr_30d
            and atr_30d > 0):
        available.append("gap_up_atr")
        gap = open_today - prior_close
        if gap >= GAP_UP_ATR_MULTIPLIER * atr_30d:
            tripped.append(f"gap_up={gap:+.2f} (>={atr_30d:.2f} ATR)")

    # Closing-auction sell imbalance — only available 15:50-16:00 ET.
    # The dollar value of paired sell pressure is the signal; large
    # negative numbers on a stretched stock = institutional unloading.
    if auction_imbalance is not None and auction_price and auction_price > 0:
        available.append("auction_imbalance")
        imbalance_usd = auction_imbalance * auction_price
        if imbalance_usd <= -AUCTION_IMBALANCE_USD:
            tripped.append(f"auction_sell_imbalance=${abs(imbalance_usd)/1e6:.1f}M")

    # Insider selling acceleration — Form 4 sales by officers / directors
    # over the last 30 days running >=2x the trailing baseline. Pre-computed
    # by FMP's get_insider_sell_pressure; we just read the flag and the
    # supporting context for the audit row.
    if isinstance(insider_pressure, dict):
        available.append("insider_sells")
        if insider_pressure.get("accelerating"):
            sells_m = float(insider_pressure.get("sells_30d_usd") or 0) / 1e6
            ratio = float(insider_pressure.get("ratio") or 0)
            n = int(insider_pressure.get("n_sellers_30d") or 0)
            tripped.append(
                f"insider_sells_30d=${sells_m:.1f}M ({ratio:.1f}x baseline, {n} sellers)"
            )

    # Analyst upgrade after a major run — late-cycle chase signal.
    # ``analyst_pressure`` is a GradePressure dataclass; we read attrs.
    if analyst_pressure is not None and getattr(analyst_pressure, "pct_60d", None) is not None:
        available.append("analyst_upgrade_after_run")
        if getattr(analyst_pressure, "upgrade_after_run", False):
            n_up = len(getattr(analyst_pressure, "upgrades_recent", []) or [])
            pct = analyst_pressure.pct_60d
            tripped.append(
                f"analyst_upgrade_after_run={n_up} upgrade(s) in 7d / +{pct*100:.0f}% / 60d"
            )

    if not available:
        return ExhaustionDecision(reason=f"{symbol}: no momentum indicators available")

    score = len(tripped) / len(available)
    exhausted = score >= SIGNALS_REQUIRED_PCT and len(tripped) >= 2
    reason = (
        f"{symbol}: {len(tripped)}/{len(available)} momentum signals tripped — "
        + ", ".join(tripped) if tripped else f"{symbol}: no exhaustion signals"
    )
    return ExhaustionDecision(
        exhausted=exhausted, score=score,
        signals_tripped=tripped, signals_available=available,
        reason=reason,
        detail={
            "current_price": current_price,
            "ma_20d": ma_20d,
            "rsi_14": rsi_14,
            "volume_ratio": volume_ratio,
            "open_today": open_today,
            "prior_close": prior_close,
            "atr_30d": atr_30d,
            "auction_imbalance": auction_imbalance,
            "auction_price": auction_price,
            "insider_pressure": insider_pressure,
            "analyst_upgrade_after_run": (
                getattr(analyst_pressure, "upgrade_after_run", None)
                if analyst_pressure is not None else None
            ),
        },
    )
