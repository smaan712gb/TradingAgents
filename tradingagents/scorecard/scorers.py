"""Scorecard sub-scorers and the ScorecardScorer node.

After the upstream subgraph finishes for a ticker, we have:

* market_report, sentiment_report, news_report, fundamentals_report
* options_flow_report and macro_report (when those tools were used)
* investment_debate (bull/bear), research-manager judgement
* trader_investment_plan, risk-debate, final_trade_decision

The scorecard layer turns this prose into a structured ``TickerScore``
on a 0–10 scale across three sub-scores (MTF setup, options sentiment,
theme-thesis fit), a weighted composite, a Buy/Hold/Avoid decision, and
short driver/risk bullets. The LLM call is grounded — every sub-score
must cite text from the agent reports — so we use the *quick* model for
cost.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from .schema import Decision, SymbolFinalState, ThemeInput, TickerScore

logger = logging.getLogger(__name__)


_SCORER_SYSTEM = """You are the Scorecard Scorer for an investment-research system.

Your job: turn the analyst reports for ONE ticker into a structured score.

Score on a 0–10 scale across three axes, where 10 is best:

* mtf_setup       — multi-timeframe price setup quality (trend, momentum, vol)
* options_sentiment — aggregate options-flow tilt (bullish premium / gamma walls / max-pain)
* thesis_fit      — how directly this ticker captures the theme's chokepoint

Then write:

* a Buy/Hold/Avoid decision (Buy ≥ 7.0 composite AND conviction ≥ {min_conviction};
  Hold = 5.5–6.9 OR low conviction; Avoid < 5.5 OR strong bear case)
* up to 6 short drivers (concrete and specific, drawn from the reports)
* up to 4 short risks
* a 1–2 sentence rationale grounded in the reports

Conviction is the Research Manager's 1–5 rating of how clearly the bull/bear
debate resolved. Read the research-manager judgement and infer it.

Output JSON only, matching this schema exactly:
{schema}
""".strip()


_SCORER_USER = """\
Theme: {theme_name}
Theme thesis: {thesis}
Theme chokepoint: {chokepoint}

Ticker: {ticker}

=== Market report ===
{market}

=== Fundamentals report ===
{fundamentals}

=== News / catalyst report ===
{news}

=== Options flow report ===
{options_flow}

=== Macro context ===
{macro}

=== Social sentiment report ===
{social}

=== Bull case ===
{bull}

=== Bear case ===
{bear}

=== Research Manager judgement ===
{judge}

=== Trader plan ===
{trader}

=== Risk-debate judgement ===
{risk_judge}

=== Final trade decision ===
{final}

Score this ticker now. Return JSON only."""


_TICKER_SCORE_SCHEMA_HINT = """\
{
  "ticker": "TICKER",
  "mtf_setup": 0.0-10.0,
  "options_sentiment": 0.0-10.0,
  "thesis_fit": 0.0-10.0,
  "composite": 0.0-10.0,
  "decision": "Buy" | "Hold" | "Avoid",
  "conviction": 1-5,
  "drivers": ["short driver 1", "short driver 2", ...],
  "risks": ["short risk 1", "short risk 2", ...],
  "rationale": "1–2 sentence grounded justification"
}"""


def _truncate(s: str | None, n: int = 4000) -> str:
    if not s:
        return "(no report — analyst was not run or returned empty)"
    if len(s) > n:
        return s[:n] + "\n…(truncated)"
    return s


def _strip_code_fence(text: str) -> str:
    """Best-effort: strip ```json fences if the model wrapped the output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _coerce_score(parsed: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    """Defensive normalisation: clamp ranges and recompute composite.

    LLMs occasionally drift outside ranges or produce a composite that
    doesn't match the weighted sum. We trust the sub-scores (they're
    grounded in cited evidence) and recompute the composite.
    """
    def clamp(v: Any, lo: float, hi: float) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return (lo + hi) / 2
        return max(lo, min(hi, f))

    parsed["mtf_setup"] = clamp(parsed.get("mtf_setup"), 0, 10)
    parsed["options_sentiment"] = clamp(parsed.get("options_sentiment"), 0, 10)
    parsed["thesis_fit"] = clamp(parsed.get("thesis_fit"), 0, 10)
    parsed["composite"] = round(
        weights["mtf_setup"] * parsed["mtf_setup"]
        + weights["options_sentiment"] * parsed["options_sentiment"]
        + weights["thesis_fit"] * parsed["thesis_fit"],
        2,
    )
    parsed["conviction"] = int(clamp(parsed.get("conviction", 3), 1, 5))

    # Normalise common label variants. Models occasionally use sell-side
    # equity-research vocabulary ("OVERWEIGHT", "UNDERWEIGHT", "STRONG BUY")
    # rather than the system's three-bucket scale. Map the common variants
    # before falling back to composite-based assignment.
    raw_decision = str(parsed.get("decision") or "").strip().upper()
    label_map = {
        "BUY": "Buy", "STRONG BUY": "Buy", "OVERWEIGHT": "Buy", "ACCUMULATE": "Buy",
        "HOLD": "Hold", "NEUTRAL": "Hold", "MARKET PERFORM": "Hold",
        "SELL": "Avoid", "STRONG SELL": "Avoid", "UNDERWEIGHT": "Avoid", "AVOID": "Avoid",
    }
    if raw_decision in label_map:
        parsed["decision"] = label_map[raw_decision]
    elif parsed.get("decision") not in ("Buy", "Hold", "Avoid"):
        # Last resort — derive from composite.
        c = parsed["composite"]
        parsed["decision"] = "Buy" if c >= 7.0 else "Hold" if c >= 5.5 else "Avoid"
    return parsed


async def score_ticker(
    *,
    llm: Any,
    theme: ThemeInput,
    state: SymbolFinalState,
    weights: dict[str, float],
    min_conviction: int = 3,
) -> TickerScore:
    """Run the ScorecardScorer for a single ticker.

    Uses LangChain's ``with_structured_output`` when available; falls
    back to a JSON parse if the provider doesn't support function calls
    (deepseek-reasoner). Errors return a Hold score so a single bad
    ticker doesn't tank the whole theme.
    """
    system = _SCORER_SYSTEM.format(
        min_conviction=min_conviction, schema=_TICKER_SCORE_SCHEMA_HINT
    )
    user = _SCORER_USER.format(
        theme_name=theme.name,
        thesis=theme.thesis,
        chokepoint=theme.chokepoint or "(none provided)",
        ticker=state.ticker,
        market=_truncate(state.market_report),
        fundamentals=_truncate(state.fundamentals_report),
        news=_truncate(state.news_report),
        options_flow=_truncate(state.options_flow_report),
        macro=_truncate(state.macro_report),
        social=_truncate(state.sentiment_report),
        bull=_truncate(state.bull_history, 2500),
        bear=_truncate(state.bear_history, 2500),
        judge=_truncate(state.judge_decision, 2000),
        trader=_truncate(state.trader_investment_plan, 1500),
        risk_judge=_truncate(state.risk_judge_decision, 1500),
        final=_truncate(state.final_trade_decision, 1500),
    )

    parsed: dict[str, Any] | None = None
    try:
        try:
            # Use a permissive shadow schema (decision: str) for the structured
            # call so unexpected labels like "OVERWEIGHT" don't trigger a
            # ValidationError that loses the rest of the otherwise-valid score.
            # We coerce decision back to the strict Buy/Hold/Avoid enum below.
            from pydantic import BaseModel, Field as _F

            class _LooseScore(BaseModel):
                ticker: str = ""
                mtf_setup: float = _F(0, ge=0, le=10)
                options_sentiment: float = _F(0, ge=0, le=10)
                thesis_fit: float = _F(0, ge=0, le=10)
                composite: float = _F(0, ge=0, le=10)
                decision: str = "Hold"
                conviction: int = _F(3, ge=1, le=5)
                drivers: list[str] = _F(default_factory=list)
                risks: list[str] = _F(default_factory=list)
                rationale: str = ""

            structured_llm = llm.with_structured_output(_LooseScore, include_raw=False)
            result = await structured_llm.ainvoke(
                [("system", system), ("human", user)]
            )
            if isinstance(result, _LooseScore):
                parsed = result.model_dump()
            elif isinstance(result, dict):
                parsed = dict(result)
            else:
                parsed = None
        except (NotImplementedError, AttributeError, TypeError):
            # Provider doesn't support structured output — fall back to free-text parse.
            response = await llm.ainvoke([("system", system), ("human", user)])
            text = getattr(response, "content", str(response))
            text = _strip_code_fence(text)
            parsed = json.loads(text)

        if parsed is None:
            raise ValueError("scorer returned no parseable result")
        parsed["ticker"] = state.ticker
        parsed = _coerce_score(parsed, weights)
        return TickerScore.model_validate(parsed)

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Scorecard scorer failed for %s: %s", state.ticker, e)
        return TickerScore(
            ticker=state.ticker,
            mtf_setup=5.0,
            options_sentiment=5.0,
            thesis_fit=5.0,
            composite=5.0,
            decision="Hold",
            conviction=3,
            drivers=[],
            risks=["Scorer failed — defaulted to neutral"],
            rationale="Scorer was unable to produce a structured score from this run.",
        )
