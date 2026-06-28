"""Pydantic types for the theme-scorecard layer.

These shapes are the contract the FastAPI surface depends on. The
frontend reads runs via the same shapes; if these change, the API
adapter in `api/app/main.py` must be updated in lockstep.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


Decision = Literal["Buy", "Hold", "Avoid"]


class ThemeInput(BaseModel):
    """User-defined investment theme — input to a theme run."""

    id: str
    name: str
    thesis: str
    chokepoint: str = ""
    tickers: list[str]
    # Optional per-ticker quant-signal block injected by the caller (Agentic
    # Edge's quant overlay). The runner stays app-agnostic: it just renders
    # whatever text it's handed into the per-ticker scoring prompt. Empty by
    # default so callers that don't supply it are unaffected.
    extra_context: dict[str, str] = {}


class AgentReport(BaseModel):
    """One analyst's prose report for one ticker, captured during a run.

    Stored on the run so the frontend can show what each agent concluded
    when the user clicks the corresponding node in the diagram.
    """

    agent_id: str  # frontend agent id (market, fundamentals, news, options, social, ...)
    ticker: str
    summary: str
    raw: Optional[str] = None  # full report text if useful for inspection


class TickerScore(BaseModel):
    """Per-ticker score emitted by ScorecardScorer after the subgraph completes."""

    ticker: str
    mtf_setup: float = Field(..., ge=0, le=10, description="Multi-timeframe setup quality")
    options_sentiment: float = Field(..., ge=0, le=10, description="Aggregate options-flow tilt")
    thesis_fit: float = Field(..., ge=0, le=10, description="How well this ticker captures the theme")
    composite: float = Field(..., ge=0, le=10)
    decision: Decision
    conviction: int = Field(..., ge=1, le=5, description="Research Manager's conviction 1–5")
    drivers: list[str] = Field(default_factory=list, max_length=6)
    risks: list[str] = Field(default_factory=list, max_length=4)
    rationale: str = Field("", description="One-paragraph justification grounded in agent reports")


class SymbolFinalState(BaseModel):
    """What we keep from the upstream subgraph's final_state per ticker.

    The upstream returns a large state dict; we cherry-pick the fields
    the scorecard layer needs and ignore the rest.
    """

    ticker: str
    market_report: str = ""
    sentiment_report: str = ""
    news_report: str = ""
    fundamentals_report: str = ""
    investment_plan: str = ""
    trader_investment_plan: str = ""
    final_trade_decision: str = ""
    options_flow_report: str = ""  # optional — populated when the option flow tool was used
    macro_report: str = ""
    bull_history: str = ""
    bear_history: str = ""
    judge_decision: str = ""
    risk_judge_decision: str = ""

    @classmethod
    def from_upstream(cls, ticker: str, final_state: dict) -> "SymbolFinalState":
        invest = final_state.get("investment_debate_state", {}) or {}
        risk = final_state.get("risk_debate_state", {}) or {}
        return cls(
            ticker=ticker,
            market_report=final_state.get("market_report", "") or "",
            sentiment_report=final_state.get("sentiment_report", "") or "",
            news_report=final_state.get("news_report", "") or "",
            fundamentals_report=final_state.get("fundamentals_report", "") or "",
            investment_plan=final_state.get("investment_plan", "") or "",
            trader_investment_plan=final_state.get("trader_investment_plan", "") or "",
            final_trade_decision=final_state.get("final_trade_decision", "") or "",
            options_flow_report=final_state.get("options_flow_report", "") or "",
            macro_report=final_state.get("macro_report", "") or "",
            bull_history=invest.get("bull_history", "") or "",
            bear_history=invest.get("bear_history", "") or "",
            judge_decision=invest.get("judge_decision", "") or "",
            risk_judge_decision=risk.get("judge_decision", "") or "",
        )


class ScoreReport(BaseModel):
    """The output of ThemeRanker — the final per-theme deliverable."""

    theme_id: str
    theme_name: str
    generated_at: datetime
    scores: list[TickerScore]
    ranking: list[str]  # tickers sorted best → worst by composite
    best_positioned: list[str]  # top N (typically 3) names
    summary: str  # prose justification grounded in scores
