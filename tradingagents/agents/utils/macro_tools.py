"""LangChain tool wrapper for macro signals.

Mirrors the pattern in ``options_flow_tools.py``. Calls into the
dataflow dispatch layer, which routes to the configured ``macro_signals``
vendor (default: alphavantage_macro). Bound to the News Analyst so the
agent can ground its commentary in macro context (yields, CPI, jobs).
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows import interface


@tool
def get_macro_signal(
    series: Annotated[
        str,
        "Comma-separated series IDs (e.g. 'TREASURY_YIELD_10Y,FED_FUNDS_RATE,CPI_YOY')",
    ],
    lookback_days: Annotated[int, "Days of history to consider"] = 365,
) -> str:
    """Return a markdown summary of the requested macro series.

    Each series is reported with its latest value, the change over the
    lookback window, and the unit. Use this to ground reasoning about
    rates, inflation, jobs, and growth alongside ticker-specific signals.

    Supported series IDs: TREASURY_YIELD_3M | TREASURY_YIELD_2Y |
    TREASURY_YIELD_10Y | TREASURY_YIELD_30Y | FED_FUNDS_RATE | CPI_YOY |
    INFLATION | REAL_GDP | RETAIL_SALES | UNEMPLOYMENT | NONFARM_PAYROLL.
    """
    method_map = interface.VENDOR_METHODS.get("get_macro_signal", {})
    vendor = interface.get_vendor("macro_signals", "get_macro_signal")
    if not vendor:
        return "(macro signals are not configured for this run)"
    fn = method_map.get(vendor)
    if not fn:
        return f"(macro vendor '{vendor}' has no get_macro_signal handler registered)"
    return fn(series_ids=series, lookback_days=lookback_days)
