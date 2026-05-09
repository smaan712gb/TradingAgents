"""Provider Protocols and shared types.

Every external data vendor (Polygon, Unusual Whales, FMP, AlphaVantage,
IBKR, yfinance) implements one or more of the Protocols defined here.
The `registry` module loads provider instances at startup and the
existing `tradingagents/dataflows/interface.py` dispatches to them by
category + method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Common value objects
# ---------------------------------------------------------------------------


class OptionRight(str, Enum):
    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class OptionContract:
    ticker: str
    expiry: date
    strike: Decimal
    right: OptionRight
    bid: Optional[Decimal]
    ask: Optional[Decimal]
    last: Optional[Decimal]
    open_interest: int
    volume: int
    iv: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]
    theta: Optional[float]


@dataclass(frozen=True)
class GammaLevel:
    ticker: str
    captured_at: datetime
    level_type: str
    price: Decimal
    notional: Optional[Decimal]
    notes: str = ""


@dataclass(frozen=True)
class FlowAlert:
    ticker: str
    triggered_at: datetime
    contract: str
    side: str
    premium: Decimal
    size: int
    open_interest: int
    iv: Optional[float]
    sweep: bool
    repeat_count: int
    # `raw` is the vendor's original payload, kept for audit. We exclude
    # it from equality and hashing so FlowAlert remains usable in sets.
    raw: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class MacroSignal:
    series_id: str
    observed_at: date
    value: float
    unit: str
    source: str = "alphavantage"


@dataclass(frozen=True)
class TradeIntent:
    ticker: str
    side: str
    qty: int
    order_type: str
    limit_px: Optional[Decimal] = None
    stop_px: Optional[Decimal] = None
    tif: str = "DAY"
    account_mode: str = "paper"


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class CoreStockProvider(Protocol):
    async def get_stock_data(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame: ...


@runtime_checkable
class TechnicalIndicatorProvider(Protocol):
    async def get_indicators(
        self,
        symbol: str,
        indicator: str,
        curr_date: date,
        look_back_days: int,
        interval: str = "daily",
        time_period: int = 14,
    ) -> str: ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    async def get_fundamentals(self, symbol: str, period: str = "annual") -> str: ...
    async def get_income_statement(self, symbol: str, period: str = "annual") -> str: ...
    async def get_balance_sheet(self, symbol: str, period: str = "annual") -> str: ...
    async def get_cashflow(self, symbol: str, period: str = "annual") -> str: ...


@runtime_checkable
class OptionsFlowProvider(Protocol):
    async def get_options_flow(
        self,
        symbol: str,
        since: datetime,
        min_premium: Decimal = Decimal("100000"),
    ) -> list[FlowAlert]: ...

    async def get_gamma_levels(self, symbol: str) -> list[GammaLevel]: ...
    async def get_max_pain(self, symbol: str, expiry: Optional[date] = None) -> Decimal: ...
    async def get_dark_pool_prints(self, symbol: str, since: datetime) -> list[dict[str, Any]]: ...


@runtime_checkable
class OptionsChainProvider(Protocol):
    async def get_options_chain(self, symbol: str, expiry: date) -> list[OptionContract]: ...
    async def get_options_snapshot(self, symbol: str) -> dict[str, Any]: ...


@runtime_checkable
class MacroSignalProvider(Protocol):
    async def get_macro_signal(self, series_id: str, lookback_days: int = 365) -> list[MacroSignal]: ...


@runtime_checkable
class NewsProvider(Protocol):
    async def get_news(self, symbol: str, since: date, until: date) -> str: ...
    async def get_global_news(self, since: date, until: date, topics: list[str] | None = None) -> str: ...
    async def get_insider_transactions(self, symbol: str, since: date) -> str: ...


@runtime_checkable
class ExecutionProvider(Protocol):
    async def get_account_summary(self) -> dict[str, Any]: ...
    async def get_positions(self) -> list[dict[str, Any]]: ...
    async def submit_trade(self, intent: TradeIntent) -> dict[str, Any]: ...
    async def cancel_order(self, order_id: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable


class RateLimitError(ProviderError):
    def __init__(self, provider: str, retry_after_s: Optional[float] = None) -> None:
        super().__init__(provider, "rate limit hit", retryable=True)
        self.retry_after_s = retry_after_s


class AuthError(ProviderError):
    def __init__(self, provider: str) -> None:
        super().__init__(provider, "authentication failed (check API key)", retryable=False)


class LiveTradingDisabledError(ProviderError):
    """Raised by IBKR provider if anything tries to trade outside paper mode."""

    def __init__(self) -> None:
        super().__init__("ibkr", "live trading is disabled — paper mode only", retryable=False)
