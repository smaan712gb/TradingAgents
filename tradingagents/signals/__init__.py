"""Signal layer — data the agents and gates consume but never trade.

Modules:
  sector_regime — sector-ETF MTF state per theme (regime classification)
  whale_flow    — UW flow ingestion (Phase B-1, forthcoming)
"""

from .sector_regime import (
    RegimeContext,
    THEME_TO_ETFS,
    classify_regime,
    get_theme_regime,
)

__all__ = [
    "RegimeContext",
    "THEME_TO_ETFS",
    "classify_regime",
    "get_theme_regime",
]
