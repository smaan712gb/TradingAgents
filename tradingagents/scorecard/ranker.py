"""Theme Ranker — fan-in node that turns N TickerScores into a ScoreReport.

Inputs: every ``TickerScore`` for the theme + the theme thesis/chokepoint.
Output: a ``ScoreReport`` with a ranking, the top "best positioned" list,
and a 2–3 sentence prose justification grounded in the scores.

Uses the deep model because the cross-ticker prioritisation benefits from
better reasoning, and the input is small (just the structured scores).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from .schema import ScoreReport, ThemeInput, TickerScore

logger = logging.getLogger(__name__)


_RANKER_SYSTEM = """You are the Theme Ranker for an investment-research system.

You receive structured per-ticker scores plus the theme thesis. Your job
is to write a 2–3 sentence summary that:

* names the top pick(s) and why (cite their drivers)
* flags any obvious holds and the reason
* stays grounded in the scores you were given — do not invent new claims

Output JSON only:
{
  "summary": "2–3 sentences, plain English, no jargon",
  "best_positioned": ["TOP_TICKER", "SECOND", "THIRD"]
}

best_positioned should hold up to three tickers ranked by composite, but
exclude any with decision == "Avoid". If fewer than three qualify, return
fewer.
""".strip()


_RANKER_USER = """Theme: {theme_name}
Theme thesis: {thesis}
Theme chokepoint: {chokepoint}

Ticker scores (sorted by composite, best first):
{score_table}

Write the summary now. Return JSON only."""


def _format_score_table(scores: list[TickerScore]) -> str:
    """Render structured scores into a compact table the LLM can scan."""
    lines = []
    for s in sorted(scores, key=lambda x: x.composite, reverse=True):
        lines.append(
            f"- {s.ticker}: composite {s.composite:.1f} ({s.decision}, conviction {s.conviction}/5)\n"
            f"    setup {s.mtf_setup:.1f} · options {s.options_sentiment:.1f} · thesis fit {s.thesis_fit:.1f}\n"
            f"    drivers: {', '.join(s.drivers) or '—'}\n"
            f"    risks: {', '.join(s.risks) or '—'}"
        )
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _safe_json_parse(text: str) -> dict[str, Any]:
    text = _strip_code_fence(text)
    return json.loads(text)


async def rank_theme(
    *,
    llm: Any,
    theme: ThemeInput,
    scores: list[TickerScore],
    top_n: int = 3,
) -> ScoreReport:
    """Run the Theme Ranker over the per-ticker scores, return ScoreReport."""
    if not scores:
        return ScoreReport(
            theme_id=theme.id,
            theme_name=theme.name,
            generated_at=datetime.now(timezone.utc),
            scores=[],
            ranking=[],
            best_positioned=[],
            summary="No tickers were scored — the theme has no actionable picks this run.",
        )

    ordered = sorted(scores, key=lambda s: s.composite, reverse=True)
    ranking = [s.ticker for s in ordered]

    # Local default in case the ranker LLM call fails — keep us shipping.
    fallback_best = [s.ticker for s in ordered if s.decision != "Avoid"][:top_n]
    fallback_summary_parts: list[str] = []
    if ordered:
        top = ordered[0]
        fallback_summary_parts.append(
            f"Top pick: {top.ticker} (composite {top.composite:.1f}, {top.decision})."
        )
        if top.drivers:
            fallback_summary_parts.append(f"Drivers: {'; '.join(top.drivers[:2])}.")
    holds = [s.ticker for s in ordered if s.decision == "Hold"]
    if holds:
        fallback_summary_parts.append(f"Held: {', '.join(holds)}.")
    fallback_summary = " ".join(fallback_summary_parts) or "No actionable picks this run."

    system = _RANKER_SYSTEM
    user = _RANKER_USER.format(
        theme_name=theme.name,
        thesis=theme.thesis,
        chokepoint=theme.chokepoint or "(none provided)",
        score_table=_format_score_table(scores),
    )

    summary = fallback_summary
    best_positioned = fallback_best

    try:
        response = await llm.ainvoke([("system", system), ("human", user)])
        text = getattr(response, "content", str(response))
        parsed = _safe_json_parse(text)
        if isinstance(parsed, dict):
            llm_summary = str(parsed.get("summary", "")).strip()
            if llm_summary:
                summary = llm_summary
            llm_best = parsed.get("best_positioned")
            if isinstance(llm_best, list):
                # Trust the LLM's ordering but keep only valid tickers.
                valid = [t for t in llm_best if isinstance(t, str) and t in ranking][:top_n]
                if valid:
                    best_positioned = valid
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning("Theme ranker LLM step failed, using fallback summary: %s", e)
    except Exception as e:  # pragma: no cover — defensive: never block the run
        logger.exception("Unexpected theme ranker failure: %s", e)

    return ScoreReport(
        theme_id=theme.id,
        theme_name=theme.name,
        generated_at=datetime.now(timezone.utc),
        scores=scores,
        ranking=ranking,
        best_positioned=best_positioned,
        summary=summary,
    )
