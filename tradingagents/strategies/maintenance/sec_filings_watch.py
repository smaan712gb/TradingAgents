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


# EDGAR-direct, CONTENT-GATED filing break detection for HELD names.
#
# Two problems this solves that the universe-wide FMP 8-K feed (above) does not:
#   1. LATENCY — the FMP feed lagged a real WDC 8-K by ~25h (filed on EDGAR
#      06-03 08:00 ET, FMP surfaced it 06-04). EDGAR is the authoritative,
#      near-real-time source; for names we HOLD (the ones that can auto-close)
#      we must not be a day late. So we pull held-name filings straight from
#      EDGAR each tick.
#   2. CONTENT — the FMP heuristic flags on TIMING (an after-hours 8-K) without
#      reading the body, which false-positived a benign WDC convertible-notes
#      exchange as a thesis-break. Here we fetch the body and run the SAME
#      keyword-severity engine as earnings transcripts; ONLY a body-confirmed
#      break (severity >= threshold) is tagged "guidance" and allowed to drive
#      an auto-close. Everything else is "material_event" (alert only).
#
# Covers 8-K (domestic) AND 6-K/20-F (foreign private issuers like TSM/ASML/ARM,
# which never file 8-K). One list call per held symbol; bodies fetched once and
# cached by accession, then re-emitted cheaply each tick so a real break keeps
# re-triggering the close until the position is flat.
_EDGAR_HELD_FORMS = {"8-K", "8-K/A", "6-K", "6-K/A", "20-F", "20-F/A"}
_EDGAR_SCAN_CACHE: dict[str, tuple[str, str, str, str, str]] = {}


async def find_new_edgar_breaks(
    *,
    edgar_provider: Any,
    held_symbols: set[str],
    since: Optional[datetime] = None,
) -> list[FilingEvent]:
    """EDGAR-direct, content-scored 8-K/6-K/20-F sweep for HELD names.

    Returns FilingEvent objects the maint loop merges into the stream. Events
    are tagged ``detail["source"]="edgar"``; only those with severity
    "guidance" (body keyword severity >= break threshold) are eligible to drive
    an auto-close — the timing-only FMP feed never auto-closes."""
    if not held_symbols or edgar_provider is None:
        return []
    from .earnings_transcript import evaluate_transcript

    cutoff = (since or (datetime.now(timezone.utc) - timedelta(days=5))).date()
    out: list[FilingEvent] = []
    for sym in {s.upper() for s in held_symbols}:
        try:
            cik = await edgar_provider.lookup_cik_by_ticker(sym)
        except Exception:
            cik = None
        if not cik:
            continue
        try:
            refs = await edgar_provider.list_filings(cik, forms=_EDGAR_HELD_FORMS, since=cutoff)
        except Exception as e:
            logger.debug("edgar-break sweep: list_filings failed for %s: %s", sym, e)
            continue
        for ref in refs:
            acc = ref.accession_no
            if acc in _EDGAR_SCAN_CACHE:
                severity, rationale, link, fdate, form = _EDGAR_SCAN_CACHE[acc]
            else:
                try:
                    text = await edgar_provider.fetch_filing_text(ref, cik)
                except Exception:
                    text = ""
                sig = evaluate_transcript({"content": text}) if text else None
                if sig and sig.thesis_break:
                    severity = "guidance"
                    rationale = (f"{ref.form} CONTENT break — keyword severity "
                                 f"{sig.severity_score}: "
                                 f"{', '.join(m['category'] for m in sig.matches[:5])}")
                else:
                    severity = "material_event"
                    rationale = (f"{ref.form} filed (content severity "
                                 f"{sig.severity_score if sig else 0}; below break threshold "
                                 f"— no auto-close)")
                link = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{ref.accession_nodashes}/"
                fdate = ref.filed_at.isoformat() if ref.filed_at else ""
                form = ref.form
                _EDGAR_SCAN_CACHE[acc] = (severity, rationale, link, fdate, form)
            out.append(FilingEvent(
                symbol=sym, filing_date=fdate, accepted_date=fdate,
                form_type=form, has_financials=False,
                severity=severity, link=link, final_link=link, rationale=rationale,
                detail={"source": "edgar"},
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
