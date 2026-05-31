"""SEC EDGAR provider — filing discovery + document fetch.

Transport-only: this class knows *which* SEC URLs to hit, the mandatory
User-Agent, and the fair-access rate cap. Parsing lives in
``tradingagents/dataflows/edgar_parse.py`` so it stays pure and testable.

SEC serves two hosts:

  * ``data.sec.gov`` — JSON APIs. The per-CIK ``/submissions/CIK##########.json``
    filing index is the poller's primary endpoint.
  * ``www.sec.gov`` — the filing *archives* (``/Archives/edgar/data/...``)
    that hold the 13F information-table XML, Form 4 XML, and 13D/G cover
    pages, plus the legacy company-search endpoint used for CIK lookup.

**SEC requires a descriptive User-Agent with a contact email** or it returns
403 to everything. We source it from ``EDGAR_USER_AGENT_EMAIL``; with no
email configured the provider refuses to construct (AuthError), so the
poller no-ops loudly rather than hammering SEC anonymously.

Fair-access policy is ~10 requests/second. We hold the per-host concurrency
to 5 (well under the cap given typical round-trip latency) and lean on the
shared client's retry/backoff for the occasional 429/503.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Any, Callable, Optional

from .base import AuthError, ProviderError
from .http import AsyncHttpClient, HttpClientConfig
from ..edgar_parse import (
    FilingRef,
    Form4,
    Holding13F,
    Sc13Signal,
    dedupe_holdings,
    parse_13f_infotable,
    parse_form4,
    parse_sc13_coverpage,
    parse_submissions,
)

logger = logging.getLogger(__name__)


def _cik10(cik: str | int) -> str:
    """Zero-pad a CIK to the 10-digit form data.sec.gov expects."""
    digits = re.sub(r"\D", "", str(cik))
    return digits.zfill(10)


def _cik_int(cik: str | int) -> str:
    """Un-padded CIK for the /Archives/edgar/data/{cik}/ path segment."""
    return str(int(re.sub(r"\D", "", str(cik)) or "0"))


class EdgarProvider:
    name = "edgar"

    def __init__(self, contact_email: Optional[str] = None) -> None:
        contact_email = contact_email or os.getenv("EDGAR_USER_AGENT_EMAIL")
        if not contact_email:
            # No anonymous access — SEC 403s requests without a contact UA.
            raise AuthError("edgar")
        self._contact = contact_email
        ua = f"Agentic Edge Research ({contact_email})"
        cfg = HttpClientConfig(timeout_s=20.0, user_agent=ua)
        # Two hosts, one client each. Concurrency 5 keeps us comfortably
        # under SEC's ~10 req/s fair-access ceiling.
        self._data = AsyncHttpClient("edgar", "https://data.sec.gov", cfg=cfg, host_concurrency=5)
        self._www = AsyncHttpClient("edgar", "https://www.sec.gov", cfg=cfg, host_concurrency=5)

    async def aclose(self) -> None:
        await self._data.aclose()
        await self._www.aclose()

    # -- filing discovery -------------------------------------------------

    async def list_filings(
        self,
        cik: str | int,
        *,
        forms: Optional[set[str]] = None,
        since: Optional[date] = None,
    ) -> list[FilingRef]:
        """All filings for a CIK, newest first, optionally filtered by form
        type and filing date. Walks the paginated ``filings.files`` archives
        when ``since`` predates the inline ``recent`` block."""
        cik10 = _cik10(cik)
        payload = await self._data.get_json(f"/submissions/CIK{cik10}.json")
        refs = parse_submissions(payload)

        # Older filings live in additional column-oriented files. Only fetch
        # them if the caller wants history older than what `recent` holds.
        extra_files = (payload.get("filings", {}) or {}).get("files") or []
        if since is not None and extra_files:
            oldest_recent = min((r.filed_at for r in refs if r.filed_at), default=None)
            if oldest_recent is None or oldest_recent > since:
                for f in extra_files:
                    name = f.get("name")
                    if not name:
                        continue
                    try:
                        chunk = await self._data.get_json(f"/submissions/{name}")
                        refs.extend(parse_submissions(chunk))
                    except ProviderError as e:
                        logger.warning("edgar: failed to fetch submissions chunk %s: %s", name, e)

        if forms:
            wanted = {f.upper() for f in forms}
            refs = [r for r in refs if r.form.upper() in wanted]
        if since:
            refs = [r for r in refs if r.filed_at is None or r.filed_at >= since]
        refs.sort(key=lambda r: (r.filed_at or date.min), reverse=True)
        return refs

    # -- document fetch ---------------------------------------------------

    async def fetch_13f_holdings(self, ref: FilingRef, cik: str | int) -> list[Holding13F]:
        """Locate and parse the information table for a 13F-HR filing,
        deduped to parent-level positions and value-normalised to whole USD."""
        name = await self._find_document(
            cik, ref.accession_nodashes,
            predicate=_is_infotable, fallback=_is_other_xml,
        )
        if not name:
            logger.warning("edgar: no info table found for %s (%s)", ref.accession_no, cik)
            return []
        xml = await self._fetch_archive_doc(cik, ref.accession_nodashes, name)
        return dedupe_holdings(parse_13f_infotable(xml, filed_at=ref.filed_at))

    async def fetch_form4(self, ref: FilingRef, cik: str | int) -> Optional[Form4]:
        name = ref.primary_document if ref.primary_document.lower().endswith(".xml") else None
        if name is None:
            name = await self._find_document(
                cik, ref.accession_nodashes, predicate=_is_other_xml,
            )
        if not name:
            return None
        xml = await self._fetch_archive_doc(cik, ref.accession_nodashes, name)
        return parse_form4(xml)

    async def fetch_sc13(self, ref: FilingRef, cik: str | int) -> Sc13Signal:
        """Best-effort cover-page extract for a 13D/13G. The high-signal fact
        (filing exists) comes from list_filings; this enriches it."""
        name = ref.primary_document or await self._find_document(
            cik, ref.accession_nodashes, predicate=lambda n: n.lower().endswith((".htm", ".html", ".txt")),
        )
        if not name:
            return Sc13Signal()
        try:
            text = await self._fetch_archive_doc(cik, ref.accession_nodashes, name)
        except ProviderError:
            return Sc13Signal()
        # Strip tags cheaply for the regex heuristics.
        text = re.sub(r"<[^>]+>", " ", text)
        return parse_sc13_coverpage(text)

    async def lookup_cik_by_name(self, company: str) -> Optional[str]:
        """Resolve a CIK from a company name via the legacy company-search
        Atom feed. Best-effort — used to fill in managers whose CIK we don't
        have hard-coded (e.g. Atreides at first run)."""
        try:
            atom = await self._www.get_text(
                "/cgi-bin/browse-edgar",
                params={"action": "getcompany", "company": company, "type": "13F",
                        "dateb": "", "owner": "include", "count": "10", "output": "atom"},
            )
        except ProviderError as e:
            logger.warning("edgar: CIK lookup for %r failed: %s", company, e)
            return None
        m = re.search(r"CIK=(\d{10})", atom) or re.search(r"<cik>(\d+)</cik>", atom, re.I)
        return _cik10(m.group(1)) if m else None

    # -- internals --------------------------------------------------------

    async def _find_document(
        self,
        cik: str | int,
        accession_nodashes: str,
        predicate: Callable[[str], bool],
        fallback: Optional[Callable[[str], bool]] = None,
    ) -> Optional[str]:
        """List a filing folder's contents and return the first file name
        matching ``predicate`` (then ``fallback`` if none match)."""
        idx = await self._www.get_json(
            f"/Archives/edgar/data/{_cik_int(cik)}/{accession_nodashes}/index.json"
        )
        items = ((idx.get("directory") or {}).get("item")) or []
        names = [it.get("name", "") for it in items if it.get("name")]
        for n in names:
            if predicate(n):
                return n
        if fallback:
            for n in names:
                if fallback(n):
                    return n
        return None

    async def _fetch_archive_doc(self, cik: str | int, accession_nodashes: str, name: str) -> str:
        return await self._www.get_text(
            f"/Archives/edgar/data/{_cik_int(cik)}/{accession_nodashes}/{name}"
        )


# Filing-agent naming for the 13F info table varies wildly
# ("form13fInfoTable.xml", "infotable.xml", "0001-...-infotable.xml"). Match
# on substrings, and explicitly exclude the cover document primary_doc.xml.
def _is_infotable(name: str) -> bool:
    n = name.lower()
    if not n.endswith(".xml") or n == "primary_doc.xml":
        return False
    return any(tok in n for tok in ("infotable", "informationtable", "info_table", "13finfo", "table"))


def _is_other_xml(name: str) -> bool:
    n = name.lower()
    return n.endswith(".xml") and n != "primary_doc.xml"
