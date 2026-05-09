"""SEC 8-K filings watcher.

On each maintenance-loop tick, sweeps the universe-wide 8-K feed,
filters to symbols we care about (held positions + active theme
universes), and emits ``FilingEvent`` records for each new filing.

The severity heuristic uses two cheap signals — without parsing the
filing body — so the watcher stays light:

  * ``hasFinancials=True`` and within ±3 days of an earnings calendar
    date → "earnings" (medium severity, expected event)
  * ``hasFinancials=True`` outside the earnings window → "guidance"
    (high severity — companies don't file financials inline unless
    something is happening, e.g. preliminary results, restatements)
  * ``hasFinancials=False`` filed during regular hours → "material_event"
    (medium severity — could be a contract win/loss, leadership change,
    M&A; needs operator eyes)
  * ``hasFinancials=False`` filed after-hours or pre-market → "after_hours"
    (high severity — material events filed off-session usually move price
    on the next open)

A future pass will pull the filing body and parse the SEC item codes
(Item 5.02 = officer departure → automatic thesis-break, Item 2.02 =
earnings, etc.). Until then the heuristic keeps us in the right
ballpark and the operator gets the link to the filing in the alert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# Tunables.
EARNINGS_WINDOW_DAYS         = 3       # days around earnings call date
AFTER_HOURS_ET_HOUR          = 16      # >= 16:00 ET = after-hours filing
PREMARKET_ET_HOUR            = 9       # < 09:30 ET = pre-market filing


@dataclass
class FilingEvent:
    symbol: str
    filing_date: str                   # 'YYYY-MM-DD HH:MM:SS' as returned by FMP
    accepted_date: Optional[str] = None
    form_type: str = "8-K"
    has_financials: bool = False
    severity: str = "material_event"   # earnings | guidance | material_event | after_hours
    link: Optional[str] = None
    final_link: Optional[str] = None
    rationale: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.strptime(str(s), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None


def _classify_severity(
    *,
    has_financials: bool,
    accepted_dt: Optional[datetime],
    days_to_earnings: Optional[int],
) -> tuple[str, str]:
    """Return (severity, rationale)."""
    # Financials filed inline
    if has_financials:
        if days_to_earnings is not None and abs(days_to_earnings) <= EARNINGS_WINDOW_DAYS:
            return "earnings", f"earnings filing (calendar within {abs(days_to_earnings)}d)"
        return "guidance", "8-K filed with financials outside the earnings window — preliminary results / restatement / off-cycle update"

    # Time-of-day signal (ET)
    if accepted_dt is not None:
        from zoneinfo import ZoneInfo
        et = accepted_dt.astimezone(ZoneInfo("America/New_York"))
        if et.hour >= AFTER_HOURS_ET_HOUR or et.hour < PREMARKET_ET_HOUR:
            return "after_hours", f"after-hours filing at {et.strftime('%H:%M ET')} — material events filed off-session usually move price on the next open"

    return "material_event", "8-K material event during regular hours"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def find_new_8k_events(
    session: AsyncSession,
    *,
    fmp_provider: Any,
    symbols_of_interest: set[str],
    since: Optional[datetime] = None,
    earnings_dates_by_symbol: Optional[dict[str, date]] = None,
) -> list[FilingEvent]:
    """Sweep recent 8-Ks and return events for symbols we care about
    that haven't been audited already.

    Dedup uses the auto_actions table — if a row with action_type
    'filing_alert' exists for this (symbol, accepted_date) tuple, the
    event is skipped. That keeps the watcher idempotent across
    restarts and paginated sweeps.
    """
    if not symbols_of_interest:
        return []

    from api.app.db import AutoAction

    if since is None:
        since = datetime.now(timezone.utc) - timedelta(hours=24)

    # FMP wants date strings; widen by a day to be safe across timezones.
    from_str = (since - timedelta(days=1)).date().isoformat()
    to_str = (datetime.now(timezone.utc) + timedelta(days=1)).date().isoformat()

    raw_rows: list[dict[str, Any]] = []
    page = 0
    while page < 5:    # cap pagination — 500 filings is a 5-day window of all 8-Ks
        try:
            chunk = await fmp_provider.get_recent_8ks(
                from_date=from_str, to_date=to_str,
                page=page, limit=100,
            )
        except Exception as e:
            logger.warning("8-K sweep failed at page %d: %s", page, e)
            break
        if not chunk:
            break
        raw_rows.extend(chunk)
        if len(chunk) < 100:
            break
        page += 1

    # Filter to symbols of interest first (cheap)
    upper = {s.upper() for s in symbols_of_interest}
    candidates = [
        r for r in raw_rows
        if str(r.get("symbol") or "").upper() in upper
    ]
    if not candidates:
        return []

    # Dedup against auto_actions written previously
    already = (
        await session.execute(
            select(AutoAction.symbol, AutoAction.payload)
            .where(AutoAction.action_type == "filing_alert")
            .where(AutoAction.timestamp >= since - timedelta(days=2))
        )
    ).all()
    seen_keys: set[tuple[str, str]] = set()
    for sym, payload in already:
        if isinstance(payload, dict):
            seen_keys.add(((sym or "").upper(),
                           str(payload.get("accepted_date") or payload.get("filing_date") or "")))

    out: list[FilingEvent] = []
    for r in candidates:
        sym = str(r.get("symbol") or "").upper()
        accepted = r.get("acceptedDate") or r.get("filingDate")
        key = (sym, str(accepted or ""))
        if key in seen_keys:
            continue
        accepted_dt = _parse_dt(r.get("acceptedDate") or r.get("filingDate"))
        # Days-to-earnings for severity classification
        dte_earn: Optional[int] = None
        if earnings_dates_by_symbol:
            ed = earnings_dates_by_symbol.get(sym)
            if ed and accepted_dt:
                dte_earn = (ed - accepted_dt.date()).days
        severity, why = _classify_severity(
            has_financials=bool(r.get("hasFinancials")),
            accepted_dt=accepted_dt,
            days_to_earnings=dte_earn,
        )
        out.append(FilingEvent(
            symbol=sym,
            filing_date=str(r.get("filingDate") or ""),
            accepted_date=str(r.get("acceptedDate") or ""),
            form_type=str(r.get("formType") or "8-K"),
            has_financials=bool(r.get("hasFinancials")),
            severity=severity,
            link=r.get("link"),
            final_link=r.get("finalLink"),
            rationale=why,
            detail={"days_to_earnings": dte_earn},
        ))
    return out


def thesis_break_signal_from_filings(
    events: list[FilingEvent],
) -> Optional[str]:
    """Compose a thesis-break reason string when a high-severity 8-K hit.

    'earnings' alone isn't a break — the position survives earnings
    routinely. 'guidance' or 'after_hours' material on a held name is
    where eyes need to go and where the trim/exit logic should lean
    defensive on the next tick.
    """
    breaks = [
        e for e in events
        if e.severity in ("guidance", "after_hours")
    ]
    if not breaks:
        return None
    return "; ".join(
        f"{e.symbol} 8-K ({e.severity}, {e.rationale})"
        for e in breaks
    )
