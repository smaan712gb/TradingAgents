"""Rotation comparison engine.

For every held position, looks for a better candidate in any active
chokepoint theme. Operator spec for rotation:

  * New theme score must clear ROTATION_THEME_THRESHOLD (default 80)
  * New ticker score must clear ROTATION_TICKER_THRESHOLD (default 75)
  * Better upside/risk than current holding
  * Recent accumulation or whale flow (advisory, not gating in v1)
  * Not already overextended (paired with momentum exhaustion check)

Returns a RotationCandidate when a meaningful upgrade is available;
the maintenance loop surfaces it as a flag (operator decides to rotate).
Auto-execution of a rotation is intentionally out of scope — moving
capital between names is a high-stakes call we want a human to approve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Tunables.
ROTATION_THEME_THRESHOLD      = 80.0        # 0-100 scale
ROTATION_TICKER_THRESHOLD     = 75.0        # 0-100 scale (composite × 10)
MIN_SCORE_DELTA               = 10.0        # candidate must beat current by this margin
LOOKBACK_DAYS                 = 7
COMPOSITE_SCALE_FACTOR        = 10.0        # ticker_scores.composite is 0-10


@dataclass
class RotationCandidate:
    current_symbol: str
    current_score: Optional[float]
    candidate_symbol: str
    candidate_score: float
    candidate_theme_id: str
    candidate_theme_health: float
    score_delta: float
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


async def find_rotation_candidate(
    session: AsyncSession,
    *,
    held_symbol: str,
    held_score: Optional[float],
) -> Optional[RotationCandidate]:
    """Find a better-positioned ticker than ``held_symbol``, if one exists."""
    from api.app.db import Run, TickerScore, Theme

    now = datetime.now(timezone.utc)
    lookback_start = now - timedelta(days=LOOKBACK_DAYS)

    # Per-theme average composite over the lookback (theme health proxy).
    theme_health_rows = (
        await session.execute(
            select(
                Run.theme_id, func.avg(TickerScore.composite).label("avg_comp"),
            )
            .join(TickerScore, TickerScore.run_id == Run.id)
            .where(Run.status == "done")
            .where(Run.finished_at >= lookback_start)
            .group_by(Run.theme_id)
        )
    ).all()
    theme_health: dict[str, float] = {
        tid: float(c) * COMPOSITE_SCALE_FACTOR
        for tid, c in theme_health_rows if c is not None
    }
    hot_themes = {
        tid for tid, h in theme_health.items() if h >= ROTATION_THEME_THRESHOLD
    }
    if not hot_themes:
        return None

    # Latest composite per (symbol, theme) pair within hot themes.
    candidate_rows = (
        await session.execute(
            select(
                TickerScore.symbol, Run.theme_id,
                func.max(TickerScore.composite).label("comp"),
            )
            .join(Run, Run.id == TickerScore.run_id)
            .where(Run.status == "done")
            .where(Run.finished_at >= lookback_start)
            .where(Run.theme_id.in_(list(hot_themes)))
            .where(TickerScore.symbol != held_symbol)
            .group_by(TickerScore.symbol, Run.theme_id)
        )
    ).all()

    best: Optional[RotationCandidate] = None
    held_scaled = held_score if held_score is not None else 0.0
    for sym, theme_id, comp in candidate_rows:
        if comp is None:
            continue
        cand_score = float(comp) * COMPOSITE_SCALE_FACTOR
        if cand_score < ROTATION_TICKER_THRESHOLD:
            continue
        delta = cand_score - held_scaled
        if delta < MIN_SCORE_DELTA:
            continue
        if best is None or cand_score > best.candidate_score:
            best = RotationCandidate(
                current_symbol=held_symbol,
                current_score=held_scaled,
                candidate_symbol=sym,
                candidate_score=cand_score,
                candidate_theme_id=theme_id,
                candidate_theme_health=theme_health.get(theme_id, 0.0),
                score_delta=delta,
                reason=(
                    f"{sym} (theme {theme_id}, health {theme_health.get(theme_id, 0):.0f}, "
                    f"score {cand_score:.0f}) beats {held_symbol} ({held_scaled:.0f}) "
                    f"by {delta:+.0f} pts"
                ),
                detail={
                    "lookback_days": LOOKBACK_DAYS,
                    "theme_threshold": ROTATION_THEME_THRESHOLD,
                    "ticker_threshold": ROTATION_TICKER_THRESHOLD,
                    "min_score_delta": MIN_SCORE_DELTA,
                },
            )
    return best
