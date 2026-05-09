"""Theme-level scorecard layer that wraps the upstream ticker subgraph.

Composition:

    ThemeRunner
    ├── for each ticker in theme: (parallel)
    │     TradingAgentsGraph(ticker).propagate()
    │       → final_state (analyst reports, debate, trade decision)
    │     → ScorecardScorer
    │       → TickerScore (mtf, options, thesis_fit, composite, drivers, risks)
    └── ThemeRanker
          → ScoreReport (ranked list + prose justification)
"""

from .schema import (
    AgentReport,
    ScoreReport,
    SymbolFinalState,
    ThemeInput,
    TickerScore,
)
from .runner import ThemeRunner, RunEvent

__all__ = [
    "AgentReport",
    "ScoreReport",
    "SymbolFinalState",
    "ThemeInput",
    "TickerScore",
    "ThemeRunner",
    "RunEvent",
]
