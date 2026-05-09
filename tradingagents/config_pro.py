"""Pro configuration overlay.

Merges with the upstream `tradingagents.default_config.DEFAULT_CONFIG`.
Point your bootstrap at this dict instead of the upstream default.

Differences from upstream:

1. LLM provider switched to DeepSeek with the configured V4 family
   (model IDs come from env so a swap is one .env edit away).
2. ``data_vendors`` and ``tool_vendors`` point at our providers
   (Polygon, FMP, Unusual Whales, AlphaVantage, IBKR). yfinance stays
   registered as a graceful fallback for stock data when Polygon isn't
   configured.
3. ``scorecard`` config block — weights and conviction gate consumed by
   ``tradingagents/scorecard/scorers.py`` and ``ranker.py``.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _load_env_file() -> None:
    """Load ``.env`` from the Agentic Edge project root (best-effort).

    The project's `.env` lives next to ``api/`` and ``web/`` — sibling to
    the cloned upstream repo. Walk up from this file to find it. If
    python-dotenv is installed we use it; otherwise a tiny manual parser
    keeps us going.
    """
    env_paths: list[Path] = []
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.exists():
            env_paths.append(candidate)
        # Also check sibling directories for a co-located project (typical
        # layout: c:/Projects/Agentic Edge/.env  next to  c:/Projects/TradingAgents/).
        for sibling in parent.iterdir() if parent.exists() else []:
            if sibling.is_dir() and sibling.name in ("Agentic Edge", "agentic-edge"):
                cand = sibling / ".env"
                if cand.exists():
                    env_paths.append(cand)
        if parent == parent.parent:
            break

    if not env_paths:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
        for p in env_paths:
            load_dotenv(p, override=False)
        logger.info("config_pro: loaded env from %s", [str(p) for p in env_paths])
        return
    except ImportError:
        pass

    # Manual fallback parser — handles KEY=VALUE, ignores comments and blanks.
    # Normalises keys to upper-case (Windows env vars are case-insensitive but
    # most consumer libraries assume upper-case; .env files in the wild often
    # mix cases, so we canonicalise here).
    for path in env_paths:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip().upper()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError as e:  # pragma: no cover
            logger.warning("config_pro: could not read %s: %s", path, e)


def build_config() -> dict:
    """Build the merged config used by the FastAPI runner."""
    _load_env_file()

    cfg = deepcopy(DEFAULT_CONFIG)

    # Pull DeepSeek model names from env so the user can swap models without
    # touching code. Architecture default IDs are the V4 family per the doc;
    # if the IDs in env differ (e.g. deepseek-chat / deepseek-reasoner),
    # those are honoured here.
    deep_model = os.environ.get("DEEPSEEK_PRO_MODEL", "deepseek-v4-pro")
    quick_model = os.environ.get("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash")
    backend_url = os.environ.get("DEEPSEEK_BASE_URL")

    # DeepSeek V4 family is live; deepseek-chat / deepseek-reasoner are
    # being deprecated 2026-07-24 and silently route to v4-flash anyway.
    # We pass the V4 names straight through. The analyst path needs
    # tool_choice support, which deepseek-v4-flash provides ONLY when
    # thinking is disabled — the openai_client subclass injects that.

    cfg.update({
        "llm_provider":     "deepseek",
        "deep_think_llm":   deep_model,
        "quick_think_llm":  quick_model,
        "backend_url":      backend_url,
        "max_debate_rounds":      int(os.environ.get("MAX_DEBATE_ROUNDS", "2")),
        "max_risk_discuss_rounds": int(os.environ.get("MAX_RISK_ROUNDS",  "1")),
        "checkpoint_enabled": False,  # in-memory runs only, for now
    })

    # Vendor selection. Polygon is preferred when its key is present;
    # otherwise fall back to yfinance for stock data so a missing
    # POLYGON_API_KEY doesn't block the whole run.
    has_polygon = bool(os.environ.get("POLYGON_API_KEY"))
    has_alphavantage = bool(os.environ.get("ALPHA_VANTAGE_API_KEY"))
    has_uw = bool(os.environ.get("UW_API_KEY"))
    has_fmp = bool(os.environ.get("FMP_API_KEY"))

    cfg["data_vendors"] = {
        "core_stock_apis":      "polygon" if has_polygon else "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data":     "fmp" if has_fmp else "yfinance",
        "news_data":            "alpha_vantage" if has_alphavantage else "yfinance",
        "options_flow":         "unusual_whales" if has_uw else None,
        "macro_signals":        "alphavantage_macro" if has_alphavantage else None,
    }
    cfg["tool_vendors"] = {
        # Per-tool overrides go here — none needed by default.
    }

    # Scorecard layer config — consumed by scorecard/scorers.py + ranker.py.
    cfg["scorecard"] = {
        "weights": {
            "mtf_setup":         0.40,
            "options_sentiment": 0.30,
            "thesis_fit":        0.30,
        },
        "min_conviction_for_trade": 3,   # 1–5 scale enforced by scorer prompt
        "concurrency_per_theme":   1,    # 1 by default — DeepSeek + provider rate limits dominate
    }
    return cfg
