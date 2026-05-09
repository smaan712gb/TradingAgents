"""Unified data-vendor SDK for TradingAgents Pro.

Providers wrap external APIs behind small Protocols (`base.py`) and a
shared async HTTP client (`http.py`). The `registry` module hooks them
into the upstream `tradingagents/dataflows/interface.py` dispatch so
they appear as drop-in vendors alongside `yfinance` and `alpha_vantage`.

Typical wiring (done once at app startup):

    from tradingagents.dataflows import interface
    from tradingagents.dataflows.providers import registry

    registry.register_into(interface)

After that the rest of the framework picks them up via config:

    config["data_vendors"]["core_stock_apis"] = "polygon"
    config["data_vendors"]["fundamental_data"] = "fmp"
    config["tool_vendors"]["get_options_flow"] = "unusual_whales"
    config["tool_vendors"]["get_macro_signal"] = "alphavantage_macro"
"""

from .base import (
    AuthError,
    ExecutionProvider,
    FlowAlert,
    FundamentalsProvider,
    GammaLevel,
    LiveTradingDisabledError,
    MacroSignal,
    MacroSignalProvider,
    OptionContract,
    OptionRight,
    OptionsChainProvider,
    OptionsFlowProvider,
    ProviderError,
    RateLimitError,
    TradeIntent,
)
from .registry import aclose_all, get_provider, register_into

__all__ = [
    "AuthError",
    "ExecutionProvider",
    "FlowAlert",
    "FundamentalsProvider",
    "GammaLevel",
    "LiveTradingDisabledError",
    "MacroSignal",
    "MacroSignalProvider",
    "OptionContract",
    "OptionRight",
    "OptionsChainProvider",
    "OptionsFlowProvider",
    "ProviderError",
    "RateLimitError",
    "TradeIntent",
    "aclose_all",
    "get_provider",
    "register_into",
]
