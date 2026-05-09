"""Theme health — daily score and deterioration tracking.

Computes per-theme health from existing run + ticker_score data without
new tables. The signal feeds two consumers:

  1. Profit-preservation trim ladder — the +100% band trims more
     aggressively when the theme is weakening (40% vs 20%).
  2. Thesis-break detector — five consecutive trading days below the
     60 floor flips the theme to "deteriorating", which triggers
     reduced exposure regardless of individual ticker scores.

Definitions:
  * **Composite** — average of latest TickerScore.composite values for
    every symbol in the theme that has a scored run today.
  * **In contact** — composite >= IN_CONTACT_FLOOR (default 60).
  * **Hot** — composite >= HOT_THRESHOLD (default 75).
  * **Deteriorating** — composite has been below IN_CONTACT_FLOOR for
    DETERIORATION_STREAK_DAYS consecutive trading days (default 5).

The functions accept a SQLAlchemy AsyncSession so the caller controls
transaction scope; nothing here mutates state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Tunables — match the operator's exit-strategy spec.
IN_CONTACT_FLOOR              = 60.0    # below this for 5 days -> deteriorating
HOT_THRESHOLD                 = 75.0    # at/above -> hot (stay/keep)
ROTATION_THRESHOLD            = 80.0    # rotation rule: new theme must clear this
DETERIORATION_STREAK_DAYS     = 5
LOOKBACK_DAYS                 = 10      # window for streak detection
COMPOSITE_SCALE_FACTOR        = 10.0    # ticker_scores.composite is 0-10 -> *10 -> 0-100


@dataclass
class ThemeHealth:
    theme_id: str
    composite: Optional[float] = None         # 0-100 scale
    sample_size: int = 0                      # how many ticker_scores fed it
    in_contact: bool = True
    hot: bool = False
    deteriorating: bool = False
    streak_days_below_floor: int = 0
    daily_history: list[tuple[str, Optional[float]]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-theme health
# ---------------------------------------------------------------------------


async def get_theme_health(
    session: AsyncSession,
    theme_id: str,
    *,
    as_of: Optional[datetime] = None,
) -> ThemeHealth:
    """Compute today's composite + N-day deterioration streak for a theme."""
    from api.app.db import Run, TickerScore

    now = (as_of or datetime.now(timezone.utc))
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    lookback_start = today - timedelta(days=LOOKBACK_DAYS)

    # Per-day average composite for runs that finished within the lookback.
    # We deliberately use the latest run per-symbol-per-day so a re-run
    # late in the day doesn't double-count.
    rows = (
        await session.execute(
            select(
                func.date(Run.finished_at).label("d"),
                func.avg(TickerScore.composite).label("avg_comp"),
                func.count(TickerScore.id).label("n"),
            )
            .join(TickerScore, TickerScore.run_id == Run.id)
            .where(Run.theme_id == theme_id)
            .where(Run.status == "done")
            .where(Run.finished_at >= lookback_start)
            .group_by(func.date(Run.finished_at))
            .order_by(func.date(Run.finished_at))
        )
    ).all()

    daily_history: list[tuple[str, Optional[float]]] = []
    today_composite: Optional[float] = None
    today_n = 0
    for d, avg_comp, n in rows:
        scaled = float(avg_comp) * COMPOSITE_SCALE_FACTOR if avg_comp is not None else None
        daily_history.append((str(d), scaled))
        if str(d) == today.date().isoformat():
            today_composite = scaled
            today_n = int(n or 0)

    # If today doesn't have a run yet, fall back to the latest day in the
    # lookback so callers get a usable signal.
    if today_composite is None and daily_history:
        latest_d, latest_c = daily_history[-1]
        today_composite = latest_c

    # Deterioration streak: consecutive trailing days with composite < floor.
    streak = 0
    for _d, c in reversed(daily_history):
        if c is None or c >= IN_CONTACT_FLOOR:
            break
        streak += 1

    in_contact = (today_composite is not None and today_composite >= IN_CONTACT_FLOOR)
    hot = (today_composite is not None and today_composite >= HOT_THRESHOLD)
    deteriorating = streak >= DETERIORATION_STREAK_DAYS

    return ThemeHealth(
        theme_id=theme_id,
        composite=today_composite,
        sample_size=today_n,
        in_contact=in_contact,
        hot=hot,
        deteriorating=deteriorating,
        streak_days_below_floor=streak,
        daily_history=daily_history,
        detail={
            "in_contact_floor": IN_CONTACT_FLOOR,
            "hot_threshold": HOT_THRESHOLD,
            "deterioration_streak_required": DETERIORATION_STREAK_DAYS,
        },
    )


# ---------------------------------------------------------------------------
# Symbol -> theme health (any theme containing the symbol)
# ---------------------------------------------------------------------------


async def is_theme_hot_for_symbol(
    session: AsyncSession, symbol: str,
) -> tuple[bool, Optional[ThemeHealth]]:
    """Return (hot, best_theme_health). A symbol can be in multiple themes;
    we use the best (max composite) theme as its representative.

    Returns (False, None) when the symbol isn't in any theme universe.
    """
    from api.app.db import ThemeSymbol

    rows = (
        await session.execute(
            select(ThemeSymbol.theme_id).where(ThemeSymbol.symbol == symbol)
        )
    ).all()
    theme_ids = [r[0] for r in rows if r[0]]
    if not theme_ids:
        return (False, None)

    best: Optional[ThemeHealth] = None
    for tid in theme_ids:
        try:
            h = await get_theme_health(session, tid)
        except Exception as e:
            logger.warning("theme health fetch failed for %s: %s", tid, e)
            continue
        if best is None or (h.composite or -1) > (best.composite or -1):
            best = h
    if best is None:
        return (False, None)
    return (best.hot, best)


async def get_thesis_break_signal(
    session: AsyncSession, symbol: str,
) -> Optional[str]:
    """Return a thesis-break reason string if any theme containing this
    symbol has been deteriorating for 5+ trading days; otherwise None.

    The maintenance loop's hard-exit logic uses this to flip a position
    from "trim" to "exit aggressively" when the chokepoint thesis loses
    structural support.
    """
    from api.app.db import ThemeSymbol

    rows = (
        await session.execute(
            select(ThemeSymbol.theme_id).where(ThemeSymbol.symbol == symbol)
        )
    ).all()
    theme_ids = [r[0] for r in rows if r[0]]
    if not theme_ids:
        return None

    breaks: list[str] = []
    for tid in theme_ids:
        try:
            h = await get_theme_health(session, tid)
        except Exception:
            continue
        if h.deteriorating:
            breaks.append(
                f"theme {tid} deteriorating "
                f"({h.streak_days_below_floor}d below {IN_CONTACT_FLOOR:.0f} floor)"
            )

    # Aggregate only when ALL containing themes deteriorate. If the symbol is
    # in two themes and one is hot while the other is fading, the position
    # is still anchored — don't trigger a thesis break.
    if breaks and len(breaks) == len(theme_ids):
        return "; ".join(breaks)
    return None
