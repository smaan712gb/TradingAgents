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
    # True when NEITHER volatility source could be read. The regime then
    # defaults to 'calm' (sizing_factor 1.0) — i.e. the guardrail is INERT and
    # full-size entries proceed. Callers must surface this: a blind read that
    # looks identical to a genuinely calm tape is how a volatility gate silently
    # stops protecting anything. See _read_vix_spx.
    degraded: bool = False


_FMP_VIX_SYMBOL = "^VIX"
_FMP_SPX_SYMBOL = "^GSPC"


async def _fmp_index_quotes(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Batched index quotes via FMP — the vendor-independent volatility fallback.

    Deliberately NOT yfinance: yfinance now routes through curl_cffi, which
    carries its own CA bundle and ignores the stdlib SSL context, so behind a
    TLS-intercepting proxy it dies with 'curl (60) SSL certificate problem' even
    though every other client in the process works. FMP goes over the same httpx
    stack as the rest of the data layer, so if the system can fetch anything at
    all it can fetch this.

    Returns {SYMBOL: {last, close, change_pct}} for whatever resolved.
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        from ..dataflows.providers.fmp import FmpProvider
        fmp = FmpProvider()
        try:
            body = await fmp._http.get_json(
                "/stable/batch-quote",
                params={"symbols": ",".join(symbols), "apikey": fmp._api_key},
            )
        finally:
            try:
                await fmp.aclose()
            except Exception:  # noqa: BLE001
                pass
        for row in (body or []):
            sym = str(row.get("symbol") or "").upper()
            if not sym:
                continue
            try:
                last = float(row["price"]) if row.get("price") is not None else None
            except (TypeError, ValueError, KeyError):
                last = None
            chg = row.get("changePercentage", row.get("changesPercentage"))
            try:
                # FMP reports percent (-0.368); the rest of this module uses a
                # fraction (-0.00368). Converting here keeps the classifier's
                # thresholds in one unit.
                change_pct = float(chg) / 100.0 if chg is not None else None
            except (TypeError, ValueError):
                change_pct = None
            out[sym] = {"last": last, "close": None, "change_pct": change_pct}
    except Exception as e:  # noqa: BLE001
        logger.warning("macro: FMP index-quote fallback failed: %s", e)
    return out


async def _read_vix_spx(ibkr: Any) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """(vix, spx, sources) — broker first, FMP when the broker returns nothing.

    ``get_index_quote`` returns ``{last: None}`` WITHOUT raising when the account
    lacks a CBOE/CME index subscription. That silent empty is indistinguishable
    from a calm tape, so the macro guardrail reads 'calm' forever and never cuts
    size. The fallback must therefore be a genuinely different vendor, not a
    retry of the same entitlement-gated path.

    ``sources`` records which vendor produced each reading so the audit row and
    the operator alert can tell "calm" apart from "couldn't see".
    """
    empty = {"last": None, "close": None, "change_pct": None}
    sources: list[str] = []
    vix: dict[str, Any] = dict(empty)
    spx: dict[str, Any] = dict(empty)

    try:
        vix = await ibkr.get_index_quote(symbol="VIX", exchange="CBOE")
    except Exception as e:  # noqa: BLE001
        logger.warning("macro: VIX fetch via broker failed: %s", e)
    try:
        spx = await ibkr.get_index_quote(symbol="SPX", exchange="CBOE")
    except Exception as e:  # noqa: BLE001
        logger.warning("macro: SPX fetch via broker failed: %s", e)

    need: list[str] = []
    if vix.get("last") is not None:
        sources.append("vix:broker")
    else:
        need.append(_FMP_VIX_SYMBOL)
    if spx.get("change_pct") is not None:
        sources.append("spx:broker")
    else:
        need.append(_FMP_SPX_SYMBOL)

    if need:
        quotes = await _fmp_index_quotes(need)
        if _FMP_VIX_SYMBOL in quotes and quotes[_FMP_VIX_SYMBOL].get("last") is not None:
            vix = quotes[_FMP_VIX_SYMBOL]
            sources.append("vix:fmp")
        if _FMP_SPX_SYMBOL in quotes and quotes[_FMP_SPX_SYMBOL].get("change_pct") is not None:
            spx = quotes[_FMP_SPX_SYMBOL]
            sources.append("spx:fmp")

    return vix, spx, sources


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
    vix_q, spx_q, sources = await _read_vix_spx(ibkr)

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

    # Blind ONLY when neither source produced a volatility read. With no VIX and
    # no SPX the classifier returns 'calm' -> sizing_factor 1.0, which is
    # indistinguishable from a genuinely quiet tape: the guardrail is inert and
    # full-size entries proceed into whatever the market is actually doing.
    # Flag it loudly so callers can refuse to size on a blind read.
    degraded = vix_last is None and spx_change is None
    if degraded:
        logger.error(
            "macro: BLIND — no VIX and no SPX from broker or fallback. Regime "
            "defaults to 'calm' (sizing x1.0), so the volatility guardrail is "
            "INERT this tick: elevated/defensive/panic can never fire."
        )

    parts: list[str] = []
    if vix_last is not None:
        parts.append(f"VIX {vix_last:.1f}")
    if spx_change is not None:
        parts.append(f"SPX {spx_change*100:+.2f}%")
    rationale = (
        f"macro={regime}"
        + (f" ({', '.join(parts)}"
           + (f" via {'+'.join(sources)}" if sources else "") + ")"
           if parts else " (BLIND read — no VIX/SPX from any source; defaulting to calm)")
    )

    return MacroRegime(
        regime=regime,
        vix_last=vix_last, vix_change_pct=vix_change,
        spx_last=spx_last, spx_change_pct=spx_change,
        sizing_factor=sizing_factor,
        leap_roll_deferred=leap_roll_deferred,
        earnings_window_mult=earnings_window_mult,
        rationale=rationale,
        degraded=degraded,
        detail={
            "vix_calm_max": VIX_CALM_MAX,
            "vix_elevated_max": VIX_ELEVATED_MAX,
            "vix_defensive_max": VIX_DEFENSIVE_MAX,
            "vix_leap_roll_defer": VIX_LEAP_ROLL_DEFER,
        },
    )
