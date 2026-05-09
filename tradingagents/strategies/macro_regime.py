"""Macro regime overlay — VIX + SPX read.

Reads the volatility and broad-market regime to add a top-of-stack
guardrail on top of per-theme sector regimes. Subscribes to two IBKR
indexes (CBOE Streaming + CME S&P) — both already paid for in the
operator's IBKR data subscription.

Regime classification:

  calm        VIX <= 18, SPX intraday  >= -0.5%
  elevated    VIX 18-25 OR SPX -0.5% to -2%
  defensive   VIX 25-35 OR SPX -2% to -3.5%
  panic       VIX > 35  OR SPX < -3.5%

Effects (consumed by callers):

  * sizing_factor       — multiplier applied to NAV-based sizing.
                          calm 1.00 / elevated 0.75 / defensive 0.50 / panic 0.0
  * leap_roll_deferred  — True when VIX > 22 and DTE > 90 (don't roll
                          into expensive vol unless the gamma cliff is
                          forcing it).
  * earnings_window_mult — multiplier on the earnings close-out window
                          (2 days normal -> 3-4 days when VIX elevated).

Returns ``MacroRegime``; callers branch on regime + read the explicit
factors. The fetcher is best-effort — if IBKR returns nothing we fall
back to "calm" rather than blocking everything (graceful degradation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Tunables.
VIX_CALM_MAX            = 18.0
VIX_ELEVATED_MAX        = 25.0
VIX_DEFENSIVE_MAX       = 35.0
SPX_CALM_MIN_PCT        = -0.005           # -0.5%
SPX_ELEVATED_MIN_PCT    = -0.020           # -2.0%
SPX_DEFENSIVE_MIN_PCT   = -0.035           # -3.5%
VIX_LEAP_ROLL_DEFER     = 22.0


@dataclass
class MacroRegime:
    regime: str = "calm"                   # calm | elevated | defensive | panic
    vix_last: Optional[float] = None
    vix_change_pct: Optional[float] = None
    spx_last: Optional[float] = None
    spx_change_pct: Optional[float] = None
    sizing_factor: float = 1.0
    leap_roll_deferred: bool = False
    earnings_window_mult: float = 1.0
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _classify_regime(
    vix_last: Optional[float], spx_change_pct: Optional[float],
) -> str:
    """Pick the WORSE of the two read regimes — be conservative when signals disagree."""
    vix_regime = "calm"
    if vix_last is not None:
        if vix_last > VIX_DEFENSIVE_MAX:
            vix_regime = "panic"
        elif vix_last > VIX_ELEVATED_MAX:
            vix_regime = "defensive"
        elif vix_last > VIX_CALM_MAX:
            vix_regime = "elevated"

    spx_regime = "calm"
    if spx_change_pct is not None:
        if spx_change_pct < SPX_DEFENSIVE_MIN_PCT:
            spx_regime = "panic"
        elif spx_change_pct < SPX_ELEVATED_MIN_PCT:
            spx_regime = "defensive"
        elif spx_change_pct < SPX_CALM_MIN_PCT:
            spx_regime = "elevated"

    severity = {"calm": 0, "elevated": 1, "defensive": 2, "panic": 3}
    if severity[vix_regime] >= severity[spx_regime]:
        return vix_regime
    return spx_regime


async def get_macro_regime(ibkr: Any) -> MacroRegime:
    """Fetch VIX + SPX from IBKR and classify the macro regime.

    Best-effort — IBKR fetch failures yield a 'calm' default so the
    rest of the system still runs. The audit row records what was
    actually read so operators can see when the fallback fired.
    """
    vix_q: dict[str, Any] = {"last": None, "close": None, "change_pct": None}
    spx_q: dict[str, Any] = {"last": None, "close": None, "change_pct": None}
    try:
        vix_q = await ibkr.get_index_quote(symbol="VIX", exchange="CBOE")
    except Exception as e:
        logger.warning("macro: VIX fetch failed: %s", e)
    try:
        spx_q = await ibkr.get_index_quote(symbol="SPX", exchange="CBOE")
    except Exception as e:
        logger.warning("macro: SPX fetch failed: %s", e)

    vix_last = vix_q.get("last")
    vix_change = vix_q.get("change_pct")
    spx_last = spx_q.get("last")
    spx_change = spx_q.get("change_pct")

    regime = _classify_regime(vix_last, spx_change)

    sizing_factor = {
        "calm": 1.00, "elevated": 0.75, "defensive": 0.50, "panic": 0.0,
    }[regime]
    earnings_window_mult = {
        "calm": 1.0, "elevated": 1.5, "defensive": 2.0, "panic": 2.0,
    }[regime]
    leap_roll_deferred = (vix_last is not None and vix_last > VIX_LEAP_ROLL_DEFER)

    parts: list[str] = []
    if vix_last is not None:
        parts.append(f"VIX {vix_last:.1f}")
    if spx_change is not None:
        parts.append(f"SPX {spx_change*100:+.2f}%")
    rationale = (
        f"macro={regime}"
        + (f" ({', '.join(parts)})" if parts else " (degraded read; default calm)")
    )

    return MacroRegime(
        regime=regime,
        vix_last=vix_last, vix_change_pct=vix_change,
        spx_last=spx_last, spx_change_pct=spx_change,
        sizing_factor=sizing_factor,
        leap_roll_deferred=leap_roll_deferred,
        earnings_window_mult=earnings_window_mult,
        rationale=rationale,
        detail={
            "vix_calm_max": VIX_CALM_MAX,
            "vix_elevated_max": VIX_ELEVATED_MAX,
            "vix_defensive_max": VIX_DEFENSIVE_MAX,
            "vix_leap_roll_defer": VIX_LEAP_ROLL_DEFER,
        },
    )
