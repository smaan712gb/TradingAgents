"""LangChain tools that surface options-flow / gamma data to agents.

Mirrors the upstream `core_stock_tools.py` and `fundamental_data_tools.py`
patterns: a `@tool`-decorated function that calls into the dataflow
dispatch layer (which routes to whichever vendor is configured for that
method — currently only Unusual Whales).

These tools are bound to a new `OptionsFlowAnalyst` node in the graph;
they're also reachable from the existing `MarketAnalyst` if you bind
them there. The intent is that the OptionsFlowAnalyst owns the heavy
gamma/max-pain reasoning so the existing market analyst stays a pure
technicals agent.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows import interface


@tool
def get_options_flow(
    symbol: Annotated[str, "Ticker symbol, e.g. NVDA"],
    hours: Annotated[int, "Lookback window in hours, defaults to 24"] = 24,
) -> str:
    """Return a markdown summary of recent options flow for ``symbol``:
    aggregate premium, bullish/bearish bucketing, top alerts by premium,
    notable gamma walls, and the nearest-expiry max-pain price.

    Source: configured `options_flow` vendor (default: Unusual Whales).
    """
    method = interface.VENDOR_METHODS["get_options_flow"]
    vendor = interface.get_vendor("options_flow", "get_options_flow")
    fn = method[vendor]
    return fn(symbol=symbol, hours=hours)


@tool
def get_macro_signal(
    series: Annotated[str, "Comma-separated series IDs (e.g. 'TREASURY_YIELD_10Y,FED_FUNDS_RATE,CPI_YOY')"],
    lookback_days: Annotated[int, "Days of history to consider"] = 365,
) -> str:
    """Return a markdown summary of the requested macro series, with the
    latest value, the change vs. start of window, and the unit. Use this
    for the Alternative Data Agent / News Analyst to ground its reasoning
    on macro context.

    Supported series IDs: TREASURY_YIELD_3M | TREASURY_YIELD_2Y |
    TREASURY_YIELD_10Y | TREASURY_YIELD_30Y | FED_FUNDS_RATE | CPI_YOY |
    INFLATION | REAL_GDP | RETAIL_SALES | UNEMPLOYMENT | NONFARM_PAYROLL.
    """
    method = interface.VENDOR_METHODS["get_macro_signal"]
    vendor = interface.get_vendor("macro_signals", "get_macro_signal")
    fn = method[vendor]
    return fn(series_ids=series, lookback_days=lookback_days)
