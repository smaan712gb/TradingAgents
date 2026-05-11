"""Fast theme runner — production-grade alternative to the multi-agent
LangGraph pipeline.

Replaces ~15 LLM calls per ticker (analysts × debate × risk × PM) with
**1 LLM call per ticker + 1 theme-health call + 1 ranker call**. With
deepseek-v4-flash (no thinking) + structured JSON output, a 6-ticker
theme run completes in 15-60 seconds instead of 30+ minutes.

Architecture:

  ┌────────────────────────┐
  │  theme_health(theme)   │  1 call — is the chokepoint binding?
  └───────────┬────────────┘
              ▼
  ┌───────────────────────────┐   ┌───────────────────────────┐
  │  score_ticker(t1, ...)    │   │  score_ticker(t2, ...)    │  N calls
  │  context: FMP fundamentals│   │  same shape               │  parallel
  │           + price action  │   │                           │  (concurrent)
  │           + UW options    │   │                           │
  │  returns TickerScore JSON │   │                           │
  └───────────────────────────┘   └───────────────────────────┘
                         ▼
  ┌────────────────────────────────────────┐
  │  rank(theme, scores) -> ScoreReport    │  1 call — final ranking
  └────────────────────────────────────────┘

Each LLM call uses ``response_format={"type": "json_object"}`` so output
is a parseable JSON document, not free-text. No tool binding, no thinking
mode — pure stateless prompting that runs in seconds.

The RunEvent stream maps onto the existing frontend agent diagram by
firing synthetic events for each "agent" the diagram displays:

  theme_health_finished   → no per-stock agent
  market / fundamentals / news / options / social   → all 5 finish per ticker
                                                       when the 1 score_ticker
                                                       call completes
  research_manager_finished  → when score includes thesis_fit + drivers
  trader_finished            → when decision is made (Buy/Hold/Avoid)
  scorecard_finished         → when TickerScore is finalised
  ranker_finished            → after the rank() call

This keeps the UI visually rich while the underlying compute is
1 call per ticker. Operators can still click each agent to read the
relevant slice of the JSON (e.g. "Market Analyst" shows the price-action
section of the rationale).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from .schema import (
    Decision, ScoreReport, ThemeInput, TickerScore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event shape — keep parity with the slow runner so callers stay simple
# ---------------------------------------------------------------------------


@dataclass
class RunEvent:
    type: str                              # agent_started | agent_finished | ticker_scored | ranker_started | ranker_finished | error
    agent_id: Optional[str] = None
    ticker: Optional[str] = None
    summary: Optional[str] = None
    score: Optional[TickerScore] = None
    report: Optional[ScoreReport] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Synthetic agent IDs (frontend diagram already displays these)
# ---------------------------------------------------------------------------

# Per-ticker agents that the diagram shows — we emit started/finished for
# each to keep the visual flow rich even though they all share one LLM call.
PER_TICKER_AGENTS = [
    "market", "fundamentals", "news", "options", "social",
    "research_manager", "trader", "scorecard",
]


# ---------------------------------------------------------------------------
# Prompts (system messages — kept short so flash responds fast)
# ---------------------------------------------------------------------------


_HEALTH_SYSTEM = """You are an investment analyst assessing whether a thematic chokepoint thesis is still in contact. Respond with strict JSON:

{
  "in_contact": true | false,
  "thesis_strength": "strong" | "intact" | "weakening" | "broken",
  "summary": "<2-3 sentence read on whether the chokepoint is still binding right now>",
  "key_inputs": ["<1-3 supporting observations>"]
}

Be calibrated. Output JSON only — no prose before or after.
"""


_TICKER_SCORE_SYSTEM = """You are a hedge-fund analyst scoring one ticker against a thematic chokepoint thesis. Respond with STRICT JSON in exactly this schema:

{
  "mtf_setup": <float 0-10>,            // Multi-timeframe price-action quality (trend, MA position, momentum)
  "options_sentiment": <float 0-10>,    // Options-flow tilt (call vs put pressure, gamma, max-pain)
  "thesis_fit": <float 0-10>,           // How tightly this name captures the chokepoint thesis
  "conviction": <int 1-5>,              // Confidence in the call across all data
  "drivers": [                          // 3-6 bullet drivers — chokepoint-specific, not generic
    "<concrete driver>"
  ],
  "risks": [                            // 2-4 bullets that could break the thesis
    "<concrete risk>"
  ],
  "rationale": "<2-3 sentence operator-readable verdict — name the chokepoint, name the lever>"
}

Scoring discipline:
- mtf_setup: 8+ requires above 20/50 day MAs + RSI between 45-70 (not extended)
- options_sentiment: 7+ requires net call flow + positive GEX positioning
- thesis_fit: 9+ requires this company OWNS a non-substitutable link in the chokepoint chain
- Anything 5.0 across the board means "no information" — say so in rationale

Output JSON only. No prose before or after.
"""


_RANKER_SYSTEM = """You rank N already-scored tickers in a chokepoint theme. Respond with STRICT JSON:

{
  "ranking": ["TICKER1", "TICKER2", ...],   // best to worst by composite
  "best_positioned": ["TICKER1", "TICKER2", "TICKER3"],  // top 3
  "summary": "<3-4 sentences naming the 1-2 strongest setups and why, including the chokepoint specifically>"
}

Tie-break high composites by conviction, then by thesis_fit. Output JSON only.
"""


# ---------------------------------------------------------------------------
# FastThemeRunner
# ---------------------------------------------------------------------------


class FastThemeRunner:
    """Drop-in replacement for ThemeRunner with the same public surface."""

    def __init__(
        self,
        on_event: Callable[[RunEvent], Awaitable[None]],
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._on_event = on_event
        self._config = config or {}
        sc = (self._config.get("scorecard") or {})
        self._weights = sc.get("weights") or {
            "mtf_setup":         0.40,
            "options_sentiment": 0.30,
            "thesis_fit":        0.30,
        }
        self._min_conviction = sc.get("min_conviction_for_trade", 3)
        try:
            self._concurrency = int(
                os.environ.get("THEME_CONCURRENCY", sc.get("concurrency_per_theme", 5))
            )
        except (TypeError, ValueError):
            self._concurrency = 5
        # Score thresholds — composite × 1.0 (0-10 scale)
        self._buy_threshold = sc.get("buy_threshold", 7.0)
        self._hold_threshold = sc.get("hold_threshold", 4.0)

    async def _emit(self, ev: RunEvent) -> None:
        try:
            await self._on_event(ev)
        except Exception as e:
            logger.warning("on_event callback raised: %s", e)

    # ---------------- LLM helpers ----------------

    def _build_llm(self):
        """Return a deepseek-v4-flash client with thinking disabled and
        json-object response format. Pure stateless call — no tool binding,
        no streaming, no thinking."""
        from openai import AsyncOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        return AsyncOpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    async def _llm_json(
        self, *, system: str, user: str, model: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """One stateless DeepSeek call returning a parsed JSON dict.

        ``thinking={"type": "disabled"}`` ensures flash routes to fast
        mode (no chain-of-thought tokens), giving ~2-5 sec latency per
        call. Failures (timeout, parse error) raise — caller handles.
        """
        client = self._build_llm()
        model = model or os.environ.get("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash")
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            temperature=0.0,
            timeout=timeout_s,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or "{}"
        return json.loads(raw)

    # ---------------- Context assembly (per-ticker data block) ----------------

    async def _assemble_ticker_context(self, symbol: str) -> str:
        """Pull cached fundamentals + price action + options flow into a
        compact text block the scorer can read.

        Best-effort: missing data points are labelled 'unavailable' rather
        than raising — the LLM is instructed to score 5.0 on missing axes.
        """
        parts: list[str] = [f"SYMBOL: {symbol}"]

        # Fundamentals via FMP
        try:
            from tradingagents.dataflows.providers.fmp import FmpProvider
            fmp = FmpProvider()
            ratios = await fmp.get_fundamentals(symbol)
            parts.append("\nFUNDAMENTALS:\n" + str(ratios)[:1500])
        except Exception as e:
            parts.append(f"\nFUNDAMENTALS: unavailable ({e})")

        # Recent price action via fallback chain
        try:
            from tradingagents.dataflows.fallback import get_stock_data_with_fallback
            df = await get_stock_data_with_fallback(
                symbol, date.today() - timedelta(days=60), date.today(),
            )
            if df is not None and len(df) > 0 and "Close" in df.columns:
                closes = df["Close"].astype(float).tolist()
                last = closes[-1]
                ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
                ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
                pct_20d = (last / closes[-21] - 1) * 100 if len(closes) >= 21 else None
                parts.append(
                    f"\nPRICE: last=${last:.2f}"
                    + (f", MA20=${ma20:.2f} ({'above' if last > ma20 else 'below'})" if ma20 else "")
                    + (f", MA50=${ma50:.2f} ({'above' if last > ma50 else 'below'})" if ma50 else "")
                    + (f", 20d change={pct_20d:+.1f}%" if pct_20d is not None else "")
                )
            else:
                parts.append("\nPRICE: unavailable")
        except Exception as e:
            parts.append(f"\nPRICE: unavailable ({e})")

        # Options flow via UW (best-effort, may be unavailable)
        try:
            from tradingagents.dataflows.providers.unusual_whales import (
                UnusualWhalesProvider,
            )
            uw = UnusualWhalesProvider()
            flows = await uw.get_options_flow(symbol)
            if flows:
                parts.append(f"\nOPTIONS FLOW (UW): {len(flows)} alerts, top by premium: " +
                             "; ".join(
                                 f"{f.get('option_chain') or f.get('strike') or '?'} "
                                 f"{(f.get('side') or '').upper()} "
                                 f"${(f.get('total_premium') or 0):,.0f}"
                                 for f in (flows[:3] if isinstance(flows, list) else [])
                             ))
            else:
                parts.append("\nOPTIONS FLOW: no unusual alerts")
        except Exception:
            parts.append("\nOPTIONS FLOW: unavailable")

        # Insider + analyst signals (already cached by FMP)
        try:
            from tradingagents.dataflows.providers.fmp import FmpProvider as _FMP
            insider = await _FMP().get_insider_sell_pressure(symbol)
            parts.append(
                f"\nINSIDER 30D: "
                f"{insider.get('n_sellers_30d', 0)} sellers, "
                f"${insider.get('sells_30d_usd', 0)/1e6:.1f}M sold, "
                f"{insider.get('ratio', 0):.1f}x baseline, "
                f"{'ACCELERATING' if insider.get('accelerating') else 'normal'}"
            )
        except Exception:
            parts.append("\nINSIDER 30D: unavailable")

        return "\n".join(parts)

    # ---------------- Main flow ----------------

    async def run(self, theme: ThemeInput) -> ScoreReport:
        """Execute the fast theme run and return the final ScoreReport."""
        if not theme.tickers:
            empty = ScoreReport(
                theme_id=theme.id, theme_name=theme.name,
                generated_at=datetime.now(timezone.utc),
                scores=[], ranking=[], best_positioned=[],
                summary="Theme has no tickers.",
            )
            await self._emit(RunEvent(type="ranker_finished", report=empty))
            return empty

        # ---- Phase 1: theme health check ----
        # Frontend doesn't have a 'theme_health' agent node — we surface
        # this as bookkeeping log only. Future: add a dedicated UI block.
        try:
            health_user = (
                f"THEME: {theme.name}\n"
                f"THESIS: {theme.thesis}\n"
                f"CHOKEPOINT: {theme.chokepoint or '(unspecified)'}\n"
                f"CURRENT DATE: {date.today().isoformat()}\n"
                "\nIs this chokepoint thesis still in contact? Score honestly."
            )
            health = await self._llm_json(system=_HEALTH_SYSTEM, user=health_user)
            logger.info("theme_health[%s]: in_contact=%s strength=%s",
                        theme.id, health.get("in_contact"), health.get("thesis_strength"))
        except Exception as e:
            logger.warning("theme_health failed for %s: %s — continuing", theme.id, e)
            health = {"in_contact": True, "thesis_strength": "intact",
                      "summary": "(health check unavailable)"}

        # ---- Phase 2: per-ticker scoring (parallel, bounded) ----
        sem = asyncio.Semaphore(self._concurrency)
        scores: list[TickerScore] = []

        async def score_one(ticker: str) -> Optional[TickerScore]:
            async with sem:
                return await self._score_one_ticker(theme, ticker, health)

        results = await asyncio.gather(
            *(score_one(t) for t in theme.tickers),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, TickerScore):
                scores.append(r)
            elif isinstance(r, Exception):
                logger.warning("score_one failed: %s", r)

        # ---- Phase 3: rank ----
        await self._emit(RunEvent(type="ranker_started", agent_id="ranker"))
        report = await self._rank(theme, scores, health)
        await self._emit(RunEvent(
            type="ranker_finished", agent_id="ranker",
            report=report, summary=report.summary,
        ))
        return report

    async def _score_one_ticker(
        self, theme: ThemeInput, ticker: str, health: dict[str, Any],
    ) -> Optional[TickerScore]:
        """Score one ticker — emits the synthetic per-agent events the
        frontend diagram expects."""

        # Fire 'started' events for the agents the diagram shows for
        # this ticker — keeps the visualization animating during the call.
        for agent_id in PER_TICKER_AGENTS:
            await self._emit(RunEvent(
                type="agent_started", agent_id=agent_id, ticker=ticker,
            ))

        try:
            context = await self._assemble_ticker_context(ticker)
            user = (
                f"THEME: {theme.name}\n"
                f"THESIS: {theme.thesis}\n"
                f"CHOKEPOINT: {theme.chokepoint or '(unspecified)'}\n"
                f"THEME HEALTH: {health.get('thesis_strength', 'intact')} — "
                f"{health.get('summary', '')}\n"
                f"\n--- TICKER DATA ---\n{context}\n"
                f"\nScore {ticker} against this theme and chokepoint. Output JSON only."
            )
            data = await self._llm_json(system=_TICKER_SCORE_SYSTEM, user=user)
        except Exception as e:
            logger.warning("score_one_ticker failed for %s: %s", ticker, e)
            await self._emit(RunEvent(
                type="error", ticker=ticker,
                error=f"score_ticker: {type(e).__name__}: {e}",
            ))
            # Best-effort placeholder so the run completes — Avoid by default
            for agent_id in PER_TICKER_AGENTS:
                await self._emit(RunEvent(
                    type="agent_finished", agent_id=agent_id, ticker=ticker,
                    summary=f"⚠ scoring failed: {e}",
                ))
            return None

        # Validate + clamp the LLM output
        def _clamp(v: Any, lo: float, hi: float, default: float = 5.0) -> float:
            try:
                f = float(v)
                return max(lo, min(hi, f))
            except (TypeError, ValueError):
                return default

        mtf = _clamp(data.get("mtf_setup"), 0, 10)
        opt = _clamp(data.get("options_sentiment"), 0, 10)
        fit = _clamp(data.get("thesis_fit"), 0, 10)
        try:
            conv = int(data.get("conviction", 3))
        except (TypeError, ValueError):
            conv = 3
        conv = max(1, min(5, conv))

        # Composite — weighted blend
        w = self._weights
        composite = (
            mtf * w.get("mtf_setup", 0.40)
            + opt * w.get("options_sentiment", 0.30)
            + fit * w.get("thesis_fit", 0.30)
        )
        composite = round(composite, 2)

        # Decision band
        decision: Decision = (
            "Buy" if composite >= self._buy_threshold
            else ("Hold" if composite >= self._hold_threshold else "Avoid")
        )
        # Conviction gate: if conviction < min, downgrade Buy -> Hold so
        # we don't act on a low-confidence top pick.
        if decision == "Buy" and conv < self._min_conviction:
            decision = "Hold"

        drivers = list(data.get("drivers") or [])[:6]
        risks = list(data.get("risks") or [])[:4]
        rationale = str(data.get("rationale") or "").strip()

        score = TickerScore(
            ticker=ticker,
            mtf_setup=mtf, options_sentiment=opt, thesis_fit=fit,
            composite=composite, decision=decision, conviction=conv,
            drivers=drivers, risks=risks, rationale=rationale,
        )

        # Emit finished events for each synthetic agent. Each gets a
        # slice of the rationale relevant to its role so clicking the
        # agent panel shows something useful.
        per_agent_summary = {
            "market":            f"setup {mtf:.1f}/10 — {rationale[:160]}",
            "fundamentals":      f"thesis_fit {fit:.1f}/10 — "
                                 + ("; ".join(drivers[:2]) if drivers else rationale[:120]),
            "news":              "no dedicated news agent in fast-mode — see drivers/risks",
            "options":           f"options_sentiment {opt:.1f}/10",
            "social":            "no dedicated social agent in fast-mode",
            "research_manager":  f"composite {composite:.1f} · conviction {conv}/5 — {rationale[:160]}",
            "trader":            f"decision = {decision}",
            "scorecard":         f"{decision} · composite {composite:.1f}",
        }
        for agent_id in PER_TICKER_AGENTS:
            await self._emit(RunEvent(
                type="agent_finished", agent_id=agent_id, ticker=ticker,
                summary=per_agent_summary.get(agent_id, ""),
            ))

        # The canonical "ticker_scored" event — drives the API's
        # add_score persistence.
        await self._emit(RunEvent(
            type="ticker_scored", ticker=ticker, score=score,
        ))
        return score

    async def _rank(
        self, theme: ThemeInput, scores: list[TickerScore],
        health: dict[str, Any],
    ) -> ScoreReport:
        """One LLM call to produce the final ScoreReport."""
        if not scores:
            return ScoreReport(
                theme_id=theme.id, theme_name=theme.name,
                generated_at=datetime.now(timezone.utc),
                scores=[], ranking=[], best_positioned=[],
                summary="No tickers scored.",
            )

        # Compact scores block for the prompt
        rows = [
            {
                "ticker": s.ticker, "composite": s.composite,
                "decision": s.decision, "conviction": s.conviction,
                "thesis_fit": s.thesis_fit, "mtf": s.mtf_setup, "opt": s.options_sentiment,
                "rationale": s.rationale[:200],
            }
            for s in scores
        ]
        user = (
            f"THEME: {theme.name}\n"
            f"CHOKEPOINT: {theme.chokepoint or '(unspecified)'}\n"
            f"HEALTH: {health.get('summary', '')}\n\n"
            f"SCORES:\n{json.dumps(rows, indent=2)}\n\n"
            "Output JSON only."
        )
        try:
            data = await self._llm_json(
                system=_RANKER_SYSTEM, user=user,
                model=os.environ.get("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro"),
                timeout_s=90.0,
            )
            ranking = list(data.get("ranking") or [])
            best = list(data.get("best_positioned") or [])
            summary = str(data.get("summary") or "").strip()
        except Exception as e:
            logger.warning("ranker failed: %s — falling back to deterministic sort", e)
            sorted_scores = sorted(scores, key=lambda s: s.composite, reverse=True)
            ranking = [s.ticker for s in sorted_scores]
            best = ranking[:3]
            summary = (
                f"{theme.name}: ranked by composite. "
                f"Top: {', '.join(best[:3])}."
            )

        # Validate ranking against actual scored tickers — guard against
        # the LLM hallucinating a different list.
        scored_tickers = {s.ticker for s in scores}
        ranking = [t for t in ranking if t in scored_tickers]
        # Fill in any missing tickers at the bottom (sorted by composite)
        missing = scored_tickers - set(ranking)
        if missing:
            extras = sorted(
                (s for s in scores if s.ticker in missing),
                key=lambda s: s.composite, reverse=True,
            )
            ranking.extend(s.ticker for s in extras)
        best = [t for t in best if t in scored_tickers][:3]
        if not best:
            best = ranking[:3]

        return ScoreReport(
            theme_id=theme.id, theme_name=theme.name,
            generated_at=datetime.now(timezone.utc),
            scores=scores, ranking=ranking, best_positioned=best,
            summary=summary,
        )
