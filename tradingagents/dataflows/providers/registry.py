"""Provider registry — wires Polygon / UW / FMP / AlphaVantage / IBKR into
the existing dispatch table in `tradingagents/dataflows/interface.py`.

The base repo's interface already supports per-category and per-tool
vendor selection via:

    config["data_vendors"][category]      -> str  (default vendor for the category)
    config["tool_vendors"][method]        -> str  (override for a single method)

This module extends `VENDOR_METHODS` at import time with bound async-to-
sync adapters for the new providers, so:

    config["data_vendors"]["core_stock_apis"] = "polygon"
    config["tool_vendors"]["get_options_flow"] = "unusual_whales"
    config["tool_vendors"]["get_fundamentals"] = "fmp"

becomes the only thing the user touches to swap vendors.

All provider instances are *cached process-wide* — we don't want to open
a fresh httpx pool per tool call. They're created lazily on first use.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from .alphavantage_macro import AlphaVantageMacroProvider
from .base import (
    AuthError,
    ExecutionProvider,
    FundamentalsProvider,
    MacroSignalProvider,
    OptionsChainProvider,
    OptionsFlowProvider,
    ProviderError,
)
from .edgar import EdgarProvider
from .fmp import FmpProvider
from .ibkr import IbkrProvider
from .polygon import PolygonProvider
from .unusual_whales import UnusualWhalesProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process-wide singletons
# ---------------------------------------------------------------------------


_LOCK = threading.Lock()
_INSTANCES: dict[str, Any] = {}


_PROVIDER_CTORS: dict[str, Callable[[], Any]] = {
    "polygon":          lambda: PolygonProvider(),
    "unusual_whales":   lambda: UnusualWhalesProvider(),
    "fmp":              lambda: FmpProvider(),
    "alphavantage_macro": lambda: AlphaVantageMacroProvider(),
    "ibkr":             lambda: IbkrProvider(),
    "edgar":            lambda: EdgarProvider(),
}


def get_provider(name: str) -> Any:
    """Return a cached instance of the named provider. Lazy and thread-safe."""
    if name in _INSTANCES:
        return _INSTANCES[name]
    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        ctor = _PROVIDER_CTORS.get(name)
        if not ctor:
            raise ProviderError("registry", f"unknown provider {name!r}")
        try:
            inst = ctor()
        except AuthError:
            # Re-raise so startup fails loudly when keys are missing.
            raise
        _INSTANCES[name] = inst
        return inst


async def aclose_all() -> None:
    """Close all opened providers. Call on FastAPI shutdown."""
    for inst in list(_INSTANCES.values()):
        try:
            await inst.aclose()
        except Exception as e:  # pragma: no cover
            logger.warning("error closing provider %s: %s", getattr(inst, "name", "?"), e)
    _INSTANCES.clear()


# ---------------------------------------------------------------------------
# Sync adapters for the legacy `interface.py` dispatch
# ---------------------------------------------------------------------------
#
# The upstream tools were written when only sync vendors existed. We give
# each new async provider a sync adapter that runs its coroutine on the
# event loop attached to the current thread (or a fresh one). Wrapping
# happens once at module import via `register_into(interface_module)`.


def _run(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # We're already inside an event loop (e.g., LangGraph node running
    # under FastAPI). Schedule and wait via a dedicated thread to avoid
    # nested-loop errors.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
        return exe.submit(asyncio.run, coro).result()


def _polygon_get_stock_data(symbol, start_date, end_date, *_, **__) -> str:
    """Vendor adapter that matches the upstream `get_stock_data` tool signature.

    Upstream's tool routes ``(symbol, start_date, end_date)`` positionally to
    each registered vendor. Accept any extras the dispatcher may add and
    coerce dates defensively (LLM tool callers sometimes pass datetime
    objects or pandas Timestamps).
    """
    start = datetime.strptime(str(start_date)[:10], "%Y-%m-%d").date()
    end = datetime.strptime(str(end_date)[:10], "%Y-%m-%d").date()
    p: PolygonProvider = get_provider("polygon")
    df = _run(p.get_stock_data(symbol, start, end))
    return df.to_csv(date_format="%Y-%m-%d")


def _fmp_fundamentals_method(method: str) -> Callable[..., str]:
    @functools.wraps(getattr(FmpProvider, method))
    def wrapped(symbol: str, *_, **__) -> str:
        p: FmpProvider = get_provider("fmp")
        return _run(getattr(p, method)(symbol))
    return wrapped


def _uw_get_options_flow(symbol, hours=24, **_: Any) -> str:
    p: UnusualWhalesProvider = get_provider("unusual_whales")
    return _run(p.summarize_flow_for_agent(symbol, hours=int(hours)))


def _av_macro_summary(series_ids, lookback_days=365, **_: Any) -> str:
    if isinstance(series_ids, str):
        series_ids = [s.strip() for s in series_ids.split(",")]
    p: AlphaVantageMacroProvider = get_provider("alphavantage_macro")
    return _run(p.summarize_for_agent(series_ids, int(lookback_days)))


# ---------------------------------------------------------------------------
# Integration with the upstream interface.VENDOR_METHODS
# ---------------------------------------------------------------------------


def register_into(interface_module: Any) -> None:
    """Extend the upstream `tradingagents.dataflows.interface` dispatch
    table with the new vendors.

    Call this once during app startup, after `set_config` and before the
    first analyst tool is invoked. The FastAPI service does this in its
    lifespan handler.
    """
    vm = interface_module.VENDOR_METHODS
    cats = interface_module.TOOLS_CATEGORIES

    # core_stock_apis ← polygon
    vm.setdefault("get_stock_data", {})["polygon"] = _polygon_get_stock_data

    # fundamental_data ← fmp
    for method in ("get_fundamentals", "get_income_statement",
                   "get_balance_sheet", "get_cashflow"):
        vm.setdefault(method, {})["fmp"] = _fmp_fundamentals_method(method)

    # New tool category — options_flow ← unusual_whales
    cats.setdefault(
        "options_flow",
        {
            "description": "Options flow, gamma exposure, max pain (Unusual Whales)",
            "tools": ["get_options_flow"],
        },
    )
    vm.setdefault("get_options_flow", {})["unusual_whales"] = _uw_get_options_flow

    # New tool category — macro_signals ← alphavantage_macro
    cats.setdefault(
        "macro_signals",
        {
            "description": "Macro / global indicators",
            "tools": ["get_macro_signal"],
        },
    )
    vm.setdefault("get_macro_signal", {})["alphavantage_macro"] = _av_macro_summary

    # Make the new vendor names visible in the validation list.
    if hasattr(interface_module, "VENDOR_LIST"):
        for new in ("polygon", "unusual_whales", "fmp", "alphavantage_macro", "ibkr"):
            if new not in interface_module.VENDOR_LIST:
                interface_module.VENDOR_LIST.append(new)

    logger.info(
        "providers.registry: registered polygon, unusual_whales, fmp, alphavantage_macro into interface dispatch"
    )
