"""Earnings call transcript analyzer.

Two analysis paths, in order of preference:

  1. **DeepSeek pass** (``evaluate_transcript_with_llm``) — sends the
     transcript through ``deepseek-v4-pro`` with thinking enabled and
     reasoning_effort=high to extract structured guidance / demand /
     margin / inventory / executive signals as JSON. Most accurate.
  2. **Regex keyword fallback** (``evaluate_transcript``) — fast
     deterministic pattern scan for high-signal phrases. Used when
     DEEPSEEK_API_KEY isn't configured or when the LLM call fails.

Both produce the same ``TranscriptSignal`` dataclass. The maintenance
loop calls ``evaluate_transcript_smart`` which picks the best
available path.

DeepSeek output is cached 30 days because earnings transcripts don't
change after the call. The FMP transcript fetch is also 30-day cached,
so subsequent ticks pay near-zero cost.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Severity-weighted keyword patterns. Higher weights for the phrases
# that most often precede a real thesis break.
_KEYWORD_WEIGHTS: list[tuple[str, int, str]] = [
    # Guidance cuts (highest signal)
    (r"lower(?:ing|ed)?\s+(?:our\s+)?(?:full[- ]year|FY|guidance|outlook)", 3, "guidance cut"),
    (r"reduc(?:ing|ed)\s+(?:our\s+)?(?:full[- ]year|FY|guidance|outlook|range)", 3, "guidance cut"),
    (r"below\s+(?:our\s+)?(?:expectations|guidance|prior\s+range)", 3, "below guidance"),
    (r"taking\s+down\s+(?:our\s+)?guidance", 3, "guidance cut"),
    (r"updated\s+(?:our\s+)?(?:full[- ]year|FY)\s+(?:guidance|outlook)\s+(?:to|down)", 2, "guidance update down"),

    # Demand / market language
    (r"slow(?:er|down|ing)\s+(?:than\s+expected|in\s+demand)", 2, "demand slowdown"),
    (r"pause\s+in\s+(?:orders|demand|spending)", 2, "demand pause"),
    (r"weaker\s+than\s+expected", 2, "weaker than expected"),
    (r"challenging\s+(?:macro|demand|environment)", 2, "challenging environment"),
    (r"softness\s+in\s+(?:demand|orders)", 2, "demand softness"),
    (r"customer\s+(?:hesitancy|caution|delay)", 2, "customer hesitation"),

    # Margin / costs
    (r"margin\s+(?:pressure|compression|headwind)", 2, "margin pressure"),
    (r"ASP\s+(?:compression|decline|pressure)", 2, "ASP compression"),
    (r"unfavou?rable\s+(?:mix|pricing)", 1, "unfavorable mix"),
    (r"input\s+cost(?:s)?\s+(?:rising|elevated|pressuring)", 1, "input cost pressure"),

    # Inventory
    (r"excess\s+inventory", 2, "excess inventory"),
    (r"channel\s+(?:correction|rebalancing|adjustment)", 2, "channel correction"),
    (r"inventory\s+(?:correction|destocking|burn)", 2, "inventory destocking"),

    # Executive / structural uncertainty
    (r"interim\s+(?:CEO|CFO|chief)", 3, "interim executive"),
    (r"transition\s+period", 1, "transition period"),
    (r"strategic\s+review", 2, "strategic review"),
    (r"evaluating\s+(?:strategic\s+)?alternatives", 3, "evaluating alternatives"),
    (r"restructur(?:e|ing)", 1, "restructuring"),
]

# Severity threshold to trip the thesis-break signal.
THESIS_BREAK_SEVERITY_THRESHOLD = 4   # 2x weight-2 hits, or 1x weight-3 + 1 weight-1


@dataclass
class TranscriptSignal:
    has_transcript: bool = False
    period: Optional[str] = None         # e.g. "Q1 2026"
    call_date: Optional[str] = None
    severity_score: int = 0
    matches: list[dict[str, Any]] = field(default_factory=list)
    thesis_break: bool = False
    rationale: str = ""


def evaluate_transcript(transcript_row: Optional[dict[str, Any]]) -> TranscriptSignal:
    """Pattern-match the transcript content and aggregate a severity score."""
    if not transcript_row or not transcript_row.get("content"):
        return TranscriptSignal(rationale="no transcript available")

    content = str(transcript_row.get("content") or "").lower()
    matches: list[dict[str, Any]] = []
    severity = 0
    seen_categories: set[str] = set()

    for pattern, weight, category in _KEYWORD_WEIGHTS:
        if category in seen_categories:
            continue   # de-dup categories so one pattern doesn't double-count
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            seen_categories.add(category)
            severity += weight
            # Pull a small context window around the match for the audit row
            start = max(0, m.start() - 80)
            end = min(len(content), m.end() + 80)
            snippet = content[start:end].replace("\n", " ").strip()
            matches.append({
                "category": category,
                "weight": weight,
                "snippet": snippet[:240],
            })

    # FMP returns period as 'Q1'/'Q2'/etc. plus a separate year. Older
    # response shapes used a numeric ``quarter``; handle both.
    period_str = transcript_row.get("period") or (
        f"Q{transcript_row.get('quarter')}" if transcript_row.get("quarter") else None
    )
    year_val = transcript_row.get("year")
    period = (
        f"{period_str} {year_val}" if period_str and year_val
        else (period_str or (str(year_val) if year_val else None))
    )
    thesis_break = severity >= THESIS_BREAK_SEVERITY_THRESHOLD
    if matches:
        cats = ", ".join(m["category"] for m in matches[:5])
        rationale = (
            f"transcript {period or '(unknown period)'} severity {severity}: {cats}"
        )
    else:
        rationale = f"transcript {period or '(unknown period)'} clean (severity 0)"

    return TranscriptSignal(
        has_transcript=True,
        period=period,
        call_date=str(transcript_row.get("date") or ""),
        severity_score=severity,
        matches=matches,
        thesis_break=thesis_break,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# DeepSeek-powered analyzer (preferred path)
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = """You are a hedge-fund analyst evaluating an earnings call transcript for thesis-break signals on a single ticker. Read the transcript and respond with strict JSON in this exact schema:

{
  "guidance_change": "raised" | "held" | "narrowed" | "lowered" | "withdrew" | "unclear",
  "guidance_evidence": "<short quote or paraphrase>",
  "demand_signal": "bullish" | "neutral" | "bearish",
  "demand_evidence": "<short quote>",
  "margin_trajectory": "expanding" | "stable" | "compressing" | "unclear",
  "margin_evidence": "<short quote>",
  "inventory_health": "tight" | "normal" | "elevated" | "destocking" | "unclear",
  "executive_signals": "<flat string: any flags like interim CEO, departures, restructuring, strategic review, or 'none'>",
  "thesis_impact": "positive" | "neutral" | "negative",
  "thesis_break_likely": true | false,
  "key_risks": ["<bullet>", "<bullet>"],
  "key_strengths": ["<bullet>", "<bullet>"],
  "summary": "<2-3 sentence operator-readable summary of what the call signalled about the next 1-2 quarters>"
}

Be calibrated and conservative. ``thesis_break_likely`` is true ONLY when one or more of:
  - guidance was lowered or withdrawn
  - demand commentary explicitly bearish
  - margin compression flagged with a specific cause
  - executive uncertainty (departures, interim, strategic review)
  - inventory destocking is acknowledged as a multi-quarter problem

Routine cycle commentary, normal seasonality, or a single soft data point in an otherwise strong call is NOT a thesis break.

Quote sparingly — pick the most material 1-2 sentences per field. Don't invent. If the transcript doesn't address a field, set it to "unclear" / "neutral" and explain in summary.

Output JSON only. No prose before or after.
"""


# Map LLM JSON output back into the TranscriptSignal severity-style format
# the maint loop already expects, so the regex and LLM paths are
# interchangeable downstream.
_LLM_SEVERITY_WEIGHTS = {
    "guidance_lowered":     3,
    "guidance_withdrew":    4,
    "demand_bearish":       2,
    "margin_compressing":   2,
    "inventory_destocking": 2,
    "executive_concern":    3,
}


async def evaluate_transcript_with_llm(
    transcript_row: Optional[dict[str, Any]],
    *, model: str = "deepseek-v4-pro",
) -> Optional[TranscriptSignal]:
    """Send the transcript through DeepSeek for structured extraction.

    Returns None when DEEPSEEK_API_KEY isn't set or the call fails;
    caller falls back to the regex path.
    """
    if not transcript_row or not transcript_row.get("content"):
        return None
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    content = str(transcript_row.get("content") or "")
    # Truncate at 80K chars for safety; DeepSeek-v4-pro has 128K tokens
    # and 80K chars ~= 20K tokens, well within budget. Long-tail Q&A
    # detail past that point rarely changes the analysis.
    if len(content) > 80_000:
        content = content[:80_000]

    symbol = str(transcript_row.get("symbol") or "?")
    period = transcript_row.get("period") or (
        f"Q{transcript_row.get('quarter')}" if transcript_row.get("quarter") else "?"
    )
    year = transcript_row.get("year") or "?"
    user_prompt = (
        f"Symbol: {symbol}\nPeriod: {period} {year}\n\n"
        f"Transcript:\n{content}"
    )

    cache_key = _llm_cache_key(symbol, period, year, content)
    cached = _LLM_CACHE.get(cache_key)
    if cached is not None:
        return _signal_from_llm(transcript_row, cached)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("openai package not installed — install with `pip install openai`")
        return None

    try:
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "enabled"}},
            reasoning_effort="high",
            temperature=0.0,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning("DeepSeek transcript analysis failed for %s: %s", symbol, e)
        return None

    _LLM_CACHE[cache_key] = parsed
    return _signal_from_llm(transcript_row, parsed)


# Process-local cache. Earnings transcripts don't change after the call
# so the cache survives forever (transcript content hash is the key —
# different content on a re-fetch would generate a fresh entry).
_LLM_CACHE: dict[str, dict[str, Any]] = {}


def _llm_cache_key(symbol: str, period: Any, year: Any, content: str) -> str:
    h = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{symbol}:{period}:{year}:{h}"


def _signal_from_llm(
    transcript_row: dict[str, Any], llm: dict[str, Any],
) -> TranscriptSignal:
    """Translate the LLM JSON output into TranscriptSignal."""
    matches: list[dict[str, Any]] = []
    severity = 0

    guidance = (llm.get("guidance_change") or "").lower()
    if guidance == "lowered":
        matches.append({"category": "guidance_lowered", "weight": _LLM_SEVERITY_WEIGHTS["guidance_lowered"],
                        "snippet": llm.get("guidance_evidence", "")})
        severity += _LLM_SEVERITY_WEIGHTS["guidance_lowered"]
    elif guidance == "withdrew":
        matches.append({"category": "guidance_withdrew", "weight": _LLM_SEVERITY_WEIGHTS["guidance_withdrew"],
                        "snippet": llm.get("guidance_evidence", "")})
        severity += _LLM_SEVERITY_WEIGHTS["guidance_withdrew"]

    if (llm.get("demand_signal") or "").lower() == "bearish":
        matches.append({"category": "demand_bearish", "weight": _LLM_SEVERITY_WEIGHTS["demand_bearish"],
                        "snippet": llm.get("demand_evidence", "")})
        severity += _LLM_SEVERITY_WEIGHTS["demand_bearish"]

    if (llm.get("margin_trajectory") or "").lower() == "compressing":
        matches.append({"category": "margin_compressing", "weight": _LLM_SEVERITY_WEIGHTS["margin_compressing"],
                        "snippet": llm.get("margin_evidence", "")})
        severity += _LLM_SEVERITY_WEIGHTS["margin_compressing"]

    if (llm.get("inventory_health") or "").lower() == "destocking":
        matches.append({"category": "inventory_destocking", "weight": _LLM_SEVERITY_WEIGHTS["inventory_destocking"],
                        "snippet": "destocking acknowledged"})
        severity += _LLM_SEVERITY_WEIGHTS["inventory_destocking"]

    exec_signals = (llm.get("executive_signals") or "").lower()
    if exec_signals and exec_signals not in ("none", "n/a", "no concerns", ""):
        matches.append({"category": "executive_concern", "weight": _LLM_SEVERITY_WEIGHTS["executive_concern"],
                        "snippet": str(llm.get("executive_signals"))[:240]})
        severity += _LLM_SEVERITY_WEIGHTS["executive_concern"]

    period_str = transcript_row.get("period") or (
        f"Q{transcript_row.get('quarter')}" if transcript_row.get("quarter") else None
    )
    year_val = transcript_row.get("year")
    period = (
        f"{period_str} {year_val}" if period_str and year_val
        else (period_str or (str(year_val) if year_val else None))
    )

    summary = str(llm.get("summary") or "")
    rationale = (
        f"LLM transcript {period or '(unknown period)'} severity {severity} "
        f"(impact={llm.get('thesis_impact','?')}): {summary}"
    )

    return TranscriptSignal(
        has_transcript=True,
        period=period,
        call_date=str(transcript_row.get("date") or ""),
        severity_score=severity,
        matches=matches,
        thesis_break=bool(llm.get("thesis_break_likely")),
        rationale=rationale,
    )


async def evaluate_transcript_smart(
    transcript_row: Optional[dict[str, Any]],
) -> TranscriptSignal:
    """Prefer the DeepSeek path; fall back to regex when the LLM is
    unavailable, the API key is missing, or the call errors out.

    The maintenance loop calls this — it doesn't need to know which
    path produced the result.
    """
    llm_result = await evaluate_transcript_with_llm(transcript_row)
    if llm_result is not None:
        return llm_result
    return evaluate_transcript(transcript_row)
