"""ThemeRunner — orchestrates a full theme scorecard run end-to-end.

Drives the upstream ``TradingAgentsGraph`` per-ticker (analyst fan-out,
bull/bear debate, research manager, trader, risk debate, portfolio
manager) and translates each node's completion into a frontend-shaped
``RunEvent``. Then runs ``score_ticker`` and ``rank_theme`` to produce
the final ``ScoreReport``.

The runner is deliberately decoupled from FastAPI — it talks to the API
layer via a single ``on_event`` async callback and a single async
``score_callback``. The API adapter wires those into its SSE queue.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.config_pro import build_config
from tradingagents.dataflows import interface as ta_interface
from tradingagents.dataflows.providers import registry as providers_registry
from tradingagents.graph.trading_graph import TradingAgentsGraph

from .ranker import rank_theme
from .schema import (
    AgentReport,
    ScoreReport,
    SymbolFinalState,
    ThemeInput,
    TickerScore,
)
from .scorers import score_ticker

logger = logging.getLogger(__name__)


# Map from upstream LangGraph node names to frontend agent ids.
# Frontend ids are stable contracts the UI's diagram and event handlers depend on.
NODE_TO_AGENT_ID: dict[str, str] = {
    "Market Analyst":        "market",
    "Social Analyst":        "social",
    "News Analyst":          "news",
    "Fundamentals Analyst":  "fundamentals",
    "Bull Researcher":       "bull",
    "Bear Researcher":       "bear",
    "Research Manager":      "research_manager",
    "Trader":                "trader",
    "Aggressive Analyst":    "risk_aggressive",
    "Conservative Analyst":  "risk_conservative",
    "Neutral Analyst":       "risk_neutral",
    "Portfolio Manager":     "portfolio_manager",
}

# State-key per agent id where the upstream stores its final report.
AGENT_REPORT_KEY: dict[str, str] = {
    "market":              "market_report",
    "social":              "sentiment_report",
    "news":                "news_report",
    "fundamentals":        "fundamentals_report",
    "research_manager":    "investment_plan",
    "trader":              "trader_investment_plan",
    "portfolio_manager":   "final_trade_decision",
}


@dataclass
class RunEvent:
    """One event emitted as the run progresses.

    type:
      "agent_started"  — agent began work on this ticker
      "agent_finished" — agent finished; ``summary`` contains a short report excerpt
      "ticker_scored"  — a TickerScore is available
      "ranker_started" — theme ranker began
      "ranker_finished"— theme ranker finished; ``report`` contains the ScoreReport
      "tool_used"      — an underlying tool fired (used to surface options-flow / macro)
      "error"          — non-fatal; the run continues for other tickers
    """

    type: str
    agent_id: Optional[str] = None
    ticker: Optional[str] = None
    summary: Optional[str] = None
    score: Optional[TickerScore] = None
    report: Optional[ScoreReport] = None
    error: Optional[str] = None


EventCallback = Callable[[RunEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# Tool-invocation tracker — surfaces options-flow / macro tool calls as
# their own "agent" events on the frontend, since the upstream graph runs
# those as tools bound to existing analysts rather than dedicated nodes.
# ---------------------------------------------------------------------------


class _ToolEventCallbackHandler(BaseCallbackHandler):
    """LangChain callback that emits frontend events when our new tools fire.

    The upstream binds ``get_options_flow`` and ``get_macro_signal`` to the
    News and Market analyst tool nodes. When those execute we want the
    frontend's "Options Flow Analyst" and the macro context line to light up
    (and to surface their report text) as if they were dedicated nodes.
    """

    TOOL_AGENT_MAP = {
        "get_options_flow":  "options",
        "get_gamma_levels":  "options",
        "get_max_pain":      "options",
        "get_macro_signal":  "macro",
    }

    def __init__(
        self,
        ticker: str,
        loop: asyncio.AbstractEventLoop,
        emit: EventCallback,
        captured: dict[str, str],
    ) -> None:
        self.ticker = ticker
        self._loop = loop
        self._emit = emit
        self._captured = captured
        self._active: dict[str, str] = {}

    def _schedule(self, coro):
        """LangChain calls handlers synchronously — schedule into our loop."""
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or ""
        agent_id = self.TOOL_AGENT_MAP.get(name)
        if not agent_id:
            return
        self._active[str(run_id)] = agent_id
        self._schedule(self._emit(RunEvent(
            type="agent_started",
            agent_id=agent_id,
            ticker=self.ticker,
        )))

    def on_tool_end(self, output, *, run_id, **kwargs):
        agent_id = self._active.pop(str(run_id), None)
        if not agent_id:
            return
        text = output if isinstance(output, str) else str(output)
        self._captured[agent_id] = text
        excerpt = text.strip().split("\n\n", 1)[0][:600]
        self._schedule(self._emit(RunEvent(
            type="agent_finished",
            agent_id=agent_id,
            ticker=self.ticker,
            summary=excerpt or "(empty result)",
        )))

    def on_tool_error(self, error, *, run_id, **kwargs):
        agent_id = self._active.pop(str(run_id), None)
        if not agent_id:
            return
        self._schedule(self._emit(RunEvent(
            type="agent_finished",
            agent_id=agent_id,
            ticker=self.ticker,
            summary=f"(tool failed: {error})",
        )))


# ---------------------------------------------------------------------------
# ThemeRunner
# ---------------------------------------------------------------------------


class ThemeRunner:
    """Drives a full theme scorecard run.

    Lifecycle:
      1. Build merged config from build_config() once.
      2. Register provider vendors into upstream interface dispatch.
      3. For each ticker:
         a. Build a fresh TradingAgentsGraph (the upstream graph is stateful per ticker).
         b. Run the underlying graph with stream_mode="values" so we can see
            each step's state and emit per-node events.
         c. After the graph completes, run score_ticker → TickerScore.
      4. Run rank_theme → ScoreReport.
    """

    def __init__(
        self,
        *,
        on_event: Optional[EventCallback] = None,
        max_ticker_concurrency: int = 1,
        trade_date: Optional[str] = None,
    ) -> None:
        self._on_event = on_event or (lambda _ev: _noop())
        self._max_concurrency = max(1, max_ticker_concurrency)
        self._trade_date = trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._config = build_config()
        self._scorecard_cfg = self._config.get("scorecard", {})
        self._weights = self._scorecard_cfg.get("weights", {
            "mtf_setup": 0.4, "options_sentiment": 0.3, "thesis_fit": 0.3,
        })
        self._min_conviction = int(self._scorecard_cfg.get("min_conviction_for_trade", 3))
        self._concurrency_per_theme = int(
            self._scorecard_cfg.get("concurrency_per_theme", self._max_concurrency)
        )
        # Wire our providers into the upstream dispatch table once.
        try:
            providers_registry.register_into(ta_interface)
        except Exception as e:  # pragma: no cover — should never block the run
            logger.warning("provider registration into interface failed: %s", e)

    async def run(self, theme: ThemeInput) -> ScoreReport:
        """Run the full scorecard for a theme and return the final ScoreReport."""
        if not theme.tickers:
            empty = ScoreReport(
                theme_id=theme.id, theme_name=theme.name,
                generated_at=datetime.now(timezone.utc),
                scores=[], ranking=[], best_positioned=[],
                summary="Theme has no tickers.",
            )
            await self._emit(RunEvent(type="ranker_finished", report=empty))
            return empty

        # Run tickers with bounded concurrency (1 by default — DeepSeek API
        # rate limits and provider rate limits dominate; parallelism above 1
        # rarely helps and makes failure modes harder to read).
        sem = asyncio.Semaphore(self._concurrency_per_theme)
        scores: list[TickerScore] = []

        async def run_one(ticker: str) -> Optional[TickerScore]:
            async with sem:
                try:
                    score = await self._run_ticker(theme, ticker)
                    if score is not None:
                        scores.append(score)
                    return score
                except Exception as e:
                    logger.exception("Ticker %s failed: %s", ticker, e)
                    await self._emit(RunEvent(
                        type="error", ticker=ticker, error=str(e),
                    ))
                    return None

        await asyncio.gather(*(run_one(t) for t in theme.tickers))

        # Ranker
        await self._emit(RunEvent(type="ranker_started", agent_id="ranker"))
        report = await self._run_ranker(theme, scores)
        await self._emit(RunEvent(
            type="ranker_finished", agent_id="ranker", report=report, summary=report.summary,
        ))
        return report

    # ------------------------------------------------------------------
    # Per-ticker pipeline
    # ------------------------------------------------------------------

    async def _run_ticker(self, theme: ThemeInput, ticker: str) -> Optional[TickerScore]:
        loop = asyncio.get_running_loop()

        # Captured tool reports — populated by the tool callback handler.
        # We store them so the scorer sees options/macro context even though
        # the upstream graph state doesn't carry these as first-class fields.
        captured_tool_reports: dict[str, str] = {}
        tool_handler = _ToolEventCallbackHandler(
            ticker=ticker, loop=loop,
            emit=self._on_event, captured=captured_tool_reports,
        )

        # Build the graph in a thread — TradingAgentsGraph constructor does
        # synchronous I/O (creates dirs, instantiates LLM clients).
        def _build() -> TradingAgentsGraph:
            return TradingAgentsGraph(
                selected_analysts=["market", "fundamentals", "news", "social"],
                debug=False,
                config=self._config,
                callbacks=[tool_handler],
            )

        ta_graph = await asyncio.to_thread(_build)

        # Stream the graph node-by-node so we can emit events.
        emitted_started: set[str] = set()
        emitted_finished: set[str] = set()
        last_state: dict[str, Any] = {}
        graph = ta_graph.graph
        propagator = ta_graph.propagator
        init_state = propagator.create_initial_state(ticker, self._trade_date, past_context="")
        args = propagator.get_graph_args()

        async def stream_loop():
            # `graph.stream(...)` is sync — wrap into a thread-driven async iter
            # so we can emit events as nodes complete without blocking the loop.
            queue: asyncio.Queue[Optional[dict[str, Any]]] = asyncio.Queue()

            def producer():
                try:
                    for chunk in graph.stream(init_state, **args):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(e)})
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            producer_task = asyncio.create_task(asyncio.to_thread(producer))

            chunk_count = 0
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    if isinstance(chunk, dict) and "__error__" in chunk:
                        logger.error(
                            "Upstream graph.stream raised for %s: %s",
                            ticker, chunk["__error__"],
                        )
                        await self._emit(RunEvent(
                            type="error", ticker=ticker,
                            error=str(chunk["__error__"]),
                        ))
                        break
                    chunk_count += 1
                    # Log the keys present in each chunk so we can see
                    # which analysts are actually contributing reports.
                    keys_with_text = sorted(
                        k for k, v in chunk.items()
                        if isinstance(v, str) and v.strip()
                    )
                    logger.debug(
                        "Ticker %s chunk #%d: text keys=%s",
                        ticker, chunk_count, keys_with_text or "(none)",
                    )
                    last_state.update(chunk)
                    await self._emit_state_progress(
                        ticker, chunk, emitted_started, emitted_finished,
                    )
            finally:
                await producer_task
            if chunk_count == 0:
                logger.warning(
                    "Ticker %s: graph.stream() yielded zero chunks — the "
                    "upstream pipeline likely failed before any analyst ran",
                    ticker,
                )

        await stream_loop()

        # Emit a final "finished" for any node we saw started but never finished
        # (defensive — graph.stream's "values" mode usually emits once per step).
        for agent_id in list(emitted_started - emitted_finished):
            await self._emit(RunEvent(
                type="agent_finished",
                agent_id=agent_id, ticker=ticker,
                summary="(no report text emitted)",
            ))
            emitted_finished.add(agent_id)

        # End-of-stream diagnostic — if the upstream LangGraph silently
        # produced no analyst reports (graph.stream returned no chunks, or
        # chunks didn't carry the report-key state), surface the gap as
        # explicit warnings per missing agent. Without this, the scorer
        # gets empty inputs and falls back to theme-only scoring while
        # the operator has no idea why the per-stock analyst panels are
        # blank on the run page.
        missing_analysts = [
            (agent_id, key) for agent_id, key in AGENT_REPORT_KEY.items()
            if not (last_state.get(key) or "").strip()
        ]
        if missing_analysts:
            agent_ids = [a for a, _ in missing_analysts]
            logger.warning(
                "Ticker %s: upstream graph produced no reports for %s — "
                "scorer will run with empty inputs (likely a tool/provider "
                "failure inside the analyst node)",
                ticker, ", ".join(agent_ids),
            )
            # Emit a started+finished pair per missing analyst so the run
            # page shows them as "Done" with the actual reason, not as
            # "Idle" forever (which would leave the operator wondering).
            for agent_id, key in missing_analysts:
                if agent_id in emitted_finished:
                    continue
                if agent_id not in emitted_started:
                    emitted_started.add(agent_id)
                    await self._emit(RunEvent(
                        type="agent_started", agent_id=agent_id, ticker=ticker,
                    ))
                emitted_finished.add(agent_id)
                await self._emit(RunEvent(
                    type="agent_finished", agent_id=agent_id, ticker=ticker,
                    summary=(
                        f"⚠ No {key} produced. Likely cause: an analyst tool "
                        f"hit a provider error (rate-limit / missing key / "
                        f"stale chain) and returned empty. Check API logs "
                        f"around the time of this run for the underlying "
                        f"failure."
                    ),
                ))

        # Build the SymbolFinalState the scorer wants.
        sfs = SymbolFinalState.from_upstream(ticker, last_state)
        sfs.options_flow_report = captured_tool_reports.get("options", "")
        sfs.macro_report = captured_tool_reports.get("macro", "")

        # Score this ticker
        await self._emit(RunEvent(
            type="agent_started", agent_id="scorecard", ticker=ticker,
        ))
        try:
            score = await score_ticker(
                llm=ta_graph.quick_thinking_llm,
                theme=theme,
                state=sfs,
                weights=self._weights,
                min_conviction=self._min_conviction,
            )
        except Exception as e:
            logger.exception("Scorecard scorer failed for %s: %s", ticker, e)
            await self._emit(RunEvent(
                type="error", ticker=ticker, error=f"scorer: {e}",
            ))
            return None

        await self._emit(RunEvent(
            type="agent_finished",
            agent_id="scorecard", ticker=ticker,
            summary=f"{score.decision} · composite {score.composite:.1f}",
        ))
        await self._emit(RunEvent(type="ticker_scored", ticker=ticker, score=score))
        return score

    async def _emit_state_progress(
        self,
        ticker: str,
        chunk: dict[str, Any],
        emitted_started: set[str],
        emitted_finished: set[str],
    ) -> None:
        """Translate a state-mode chunk into agent_started / agent_finished events.

        With ``stream_mode="values"``, each chunk is the *full* current state
        after a node ran. We detect which agent's slot just got populated and
        emit one finished event per fresh report. We emit "started" greedily
        when the upstream begins processing the next analyst.
        """
        # When sender is set, that node just spoke. Use it to emit a started
        # event for the next analyst in the chain (best-effort).
        sender = chunk.get("sender")
        if isinstance(sender, str):
            agent_id = NODE_TO_AGENT_ID.get(sender)
            if agent_id and agent_id not in emitted_started:
                emitted_started.add(agent_id)
                await self._emit(RunEvent(
                    type="agent_started", agent_id=agent_id, ticker=ticker,
                ))

        # Detect newly populated reports — emit finished + carry the summary.
        for agent_id, key in AGENT_REPORT_KEY.items():
            if agent_id in emitted_finished:
                continue
            text = chunk.get(key)
            if not text:
                continue
            if agent_id not in emitted_started:
                emitted_started.add(agent_id)
                await self._emit(RunEvent(
                    type="agent_started", agent_id=agent_id, ticker=ticker,
                ))
            emitted_finished.add(agent_id)
            excerpt = _excerpt(text, 600)
            await self._emit(RunEvent(
                type="agent_finished",
                agent_id=agent_id, ticker=ticker,
                summary=excerpt,
            ))

        # Bull / bear / risk-debate aren't first-class state keys; surface
        # progress from their sub-state dicts.
        invest = chunk.get("investment_debate_state") or {}
        for agent_id, key in (("bull", "bull_history"), ("bear", "bear_history")):
            if agent_id in emitted_finished:
                continue
            text = invest.get(key)
            if not text:
                continue
            if agent_id not in emitted_started:
                emitted_started.add(agent_id)
                await self._emit(RunEvent(
                    type="agent_started", agent_id=agent_id, ticker=ticker,
                ))
            emitted_finished.add(agent_id)
            await self._emit(RunEvent(
                type="agent_finished",
                agent_id=agent_id, ticker=ticker,
                summary=_excerpt(text, 600),
            ))

        risk = chunk.get("risk_debate_state") or {}
        for agent_id, key in (
            ("risk_aggressive",  "aggressive_history"),
            ("risk_conservative","conservative_history"),
            ("risk_neutral",     "neutral_history"),
        ):
            if agent_id in emitted_finished:
                continue
            text = risk.get(key)
            if not text:
                continue
            if agent_id not in emitted_started:
                emitted_started.add(agent_id)
                await self._emit(RunEvent(
                    type="agent_started", agent_id=agent_id, ticker=ticker,
                ))
            emitted_finished.add(agent_id)
            await self._emit(RunEvent(
                type="agent_finished",
                agent_id=agent_id, ticker=ticker,
                summary=_excerpt(text, 600),
            ))

    # ------------------------------------------------------------------
    # Theme ranker
    # ------------------------------------------------------------------

    async def _run_ranker(
        self, theme: ThemeInput, scores: list[TickerScore],
    ) -> ScoreReport:
        # Spin up a one-shot LLM client matching deep_think_llm — we don't
        # need a full TradingAgentsGraph just to run the ranker prompt.
        from tradingagents.llm_clients import create_llm_client
        deep = create_llm_client(
            provider=self._config["llm_provider"],
            model=self._config["deep_think_llm"],
            base_url=self._config.get("backend_url"),
        ).get_llm()
        return await rank_theme(llm=deep, theme=theme, scores=scores)

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    async def _emit(self, ev: RunEvent) -> None:
        try:
            await self._on_event(ev)
        except Exception as e:  # pragma: no cover
            logger.warning("event callback raised: %s", e)


def _excerpt(text: str, n: int) -> str:
    """First non-empty paragraph, capped at n chars."""
    text = (text or "").strip()
    if not text:
        return "(empty)"
    para = text.split("\n\n", 1)[0]
    if len(para) > n:
        return para[: n - 1].rstrip() + "…"
    return para


async def _noop() -> None:  # pragma: no cover
    return None
