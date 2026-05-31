"""Pure parsers for SEC EDGAR filings — no I/O, fully unit-testable.

The EDGAR *provider* (`dataflows/providers/edgar.py`) owns transport (which
URLs, rate limits, the mandatory User-Agent). This module owns the *shape*:
turn raw bytes/JSON from SEC into typed dataclasses the hedge-fund tracker
can store and diff.

Three filing families matter for Phase 1:

  * **13F-HR information table** (XML) — a manager's full long-equity/options
    book as of a quarter end. The bulk signal.
  * **Form 4** (XML ownership document) — insider transactions; carries the
    issuer ticker directly.
  * **13D / 13G / amendments** (text/HTML cover pages) — activist/passive
    stake disclosures. Best-effort field extraction; the high-signal fact
    ("manager X filed a 13D on subject Y today") comes from the submissions
    index alone, so detailed cover-page parsing is opportunistic.

Two correctness traps this module handles explicitly:

  1. **Value units.** EDGAR release 22.4.1 (2023-01-03) switched the 13F
     Value column from *thousands of dollars* to *whole dollars*. The first
     reports affected are the Q4-2022 (period 2022-12-31) filings submitted
     in early 2023. The discriminator is therefore the **filing date**, not
     the period-of-report. We normalise everything to whole USD on ingest.
  2. **Namespaces.** 13F infotable and Form 4 XML come with assorted default
     and prefixed namespaces (``ns1:``, ``n1:``, none) that vary by filing
     agent. We match on the *local* tag name so the parser is namespace-blind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

# EDGAR 22.4.1 cut over the 13F Value column from thousands → whole dollars
# for filings submitted on/after 2023-01-03. Anything filed before is in
# thousands. We compare on filed date and use Jan 1 as the (holiday-safe)
# boundary — no 13F was filed 2023-01-01/02 (weekend + federal holiday).
_WHOLE_DOLLAR_CUTOVER = date(2023, 1, 1)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilingRef:
    """One row from a CIK's submissions index — enough to decide whether we
    already have it and where to fetch the documents."""
    accession_no: str           # dashed form, e.g. "0001541617-24-000012"
    form: str                   # "13F-HR", "SC 13D", "4", ...
    filed_at: Optional[date]
    period_end: Optional[date]
    primary_document: str       # filename of the primary doc within the folder

    @property
    def accession_nodashes(self) -> str:
        return self.accession_no.replace("-", "")


@dataclass
class Holding13F:
    """One ``infoTable`` row, value normalised to whole USD."""
    issuer_name: str
    cusip: str
    value_usd: float            # whole dollars, post-normalisation
    shares: float               # sshPrnamt when type == SH; prn amount otherwise
    shares_type: str            # "SH" | "PRN"
    put_call: Optional[str] = None   # "Put" | "Call" | None (long underlying)
    title_of_class: str = ""

    @property
    def put_call_flag(self) -> str:
        """Canonical short flag used as part of the holdings unique key.
        '' for the common long-stock row, 'C'/'P' for option rows."""
        if not self.put_call:
            return ""
        return self.put_call.strip()[:1].upper()


@dataclass
class Form4Transaction:
    security_title: str
    transaction_date: Optional[date]
    code: str                   # P (purchase), S (sale), A (grant), etc.
    shares: float
    price_per_share: Optional[float]
    acquired_disposed: str      # "A" | "D"
    is_derivative: bool


@dataclass
class Form4:
    issuer_name: str
    issuer_ticker: str
    issuer_cik: str
    reporting_owner: str
    transactions: list[Form4Transaction] = field(default_factory=list)

    @property
    def net_open_market_shares(self) -> float:
        """Net shares from open-market buys (P) minus sales (S). Positive =
        net insider buying — the high-signal direction."""
        net = 0.0
        for t in self.transactions:
            if t.code == "P":
                net += t.shares
            elif t.code == "S":
                net -= t.shares
        return net


@dataclass
class Sc13Signal:
    """Best-effort extract from a 13D/13G cover page."""
    subject_company: str = ""
    cusip: str = ""
    percent_of_class: Optional[float] = None
    event_date: Optional[date] = None


# ---------------------------------------------------------------------------
# Namespace-blind XML helpers
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    """Strip any ``{namespace}`` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _iter_local(elem: ET.Element, name: str) -> Iterable[ET.Element]:
    """Yield descendants whose local tag name equals ``name``."""
    for child in elem.iter():
        if _local(child.tag) == name:
            yield child


def _find_local(elem: ET.Element, name: str) -> Optional[ET.Element]:
    for child in _iter_local(elem, name):
        return child
    return None


def _text_local(elem: Optional[ET.Element], name: str) -> str:
    if elem is None:
        return ""
    node = _find_local(elem, name)
    if node is None or node.text is None:
        return ""
    return node.text.strip()


def _direct_child_text(elem: ET.Element, name: str) -> str:
    """Text of the first *direct* child with local name ``name`` (avoids
    grabbing a same-named node nested deeper, e.g. votingAuthority/None)."""
    for child in elem:
        if _local(child.tag) == name and child.text:
            return child.text.strip()
    return ""


def _to_float(s: str) -> float:
    if not s:
        return 0.0
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return 0.0


def _to_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Submissions index (data.sec.gov/submissions/CIK##########.json)
# ---------------------------------------------------------------------------


def parse_submissions(payload: dict[str, Any]) -> list[FilingRef]:
    """Turn one ``submissions`` JSON block into FilingRefs.

    Handles both the top-level ``filings.recent`` block and a bare
    ``recent``-shaped dict (the paginated ``filings.files[*]`` archives have
    the same column-oriented shape but no ``filings`` wrapper)."""
    recent = payload.get("filings", {}).get("recent") if "filings" in payload else payload
    if not isinstance(recent, dict):
        return []
    accessions = recent.get("accessionNumber") or []
    forms = recent.get("form") or []
    filed = recent.get("filingDate") or []
    period = recent.get("reportDate") or []
    primary = recent.get("primaryDocument") or []

    out: list[FilingRef] = []
    for i, acc in enumerate(accessions):
        out.append(
            FilingRef(
                accession_no=acc,
                form=(forms[i] if i < len(forms) else "").strip(),
                filed_at=_to_date(filed[i]) if i < len(filed) else None,
                period_end=_to_date(period[i]) if i < len(period) else None,
                primary_document=(primary[i] if i < len(primary) else "").strip(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 13F-HR information table
# ---------------------------------------------------------------------------


def normalize_13f_value(raw_value: float, *, filed_at: Optional[date]) -> float:
    """Normalise a 13F Value column figure to whole USD.

    Pre-2023-01-03 filings report in thousands; multiply by 1000. Filings
    from the cutover onward are already whole dollars. When the filing date
    is unknown we assume *thousands* (the conservative legacy default) — the
    backfill always knows filed_at, so this only bites truly malformed input.
    """
    if filed_at is None or filed_at < _WHOLE_DOLLAR_CUTOVER:
        return raw_value * 1000.0
    return raw_value


def parse_13f_infotable(xml_text: str | bytes, *, filed_at: Optional[date]) -> list[Holding13F]:
    """Parse a 13F information table XML into normalised holdings.

    Each ``infoTable`` row → one Holding13F with value in whole USD. Option
    rows (putCall present) are kept distinct from the long-underlying row for
    the same issuer so they don't collapse during dedup."""
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    xml_text = xml_text.strip()
    if not xml_text:
        return []
    root = ET.fromstring(xml_text)

    holdings: list[Holding13F] = []
    for info in _iter_local(root, "infoTable"):
        cusip = _direct_child_text(info, "cusip").upper()
        raw_value = _to_float(_direct_child_text(info, "value"))
        # shrsOrPrnAmt wraps sshPrnamt + sshPrnamtType.
        shrs = _find_local(info, "shrsOrPrnAmt")
        shares = _to_float(_text_local(shrs, "sshPrnamt"))
        shares_type = (_text_local(shrs, "sshPrnamtType") or "SH").upper()
        put_call = _direct_child_text(info, "putCall") or None
        holdings.append(
            Holding13F(
                issuer_name=_direct_child_text(info, "nameOfIssuer").strip(),
                cusip=cusip,
                value_usd=normalize_13f_value(raw_value, filed_at=filed_at),
                shares=shares,
                shares_type=shares_type,
                put_call=put_call,
                title_of_class=_direct_child_text(info, "titleOfClass").strip(),
            )
        )
    return holdings


def dedupe_holdings(holdings: list[Holding13F]) -> list[Holding13F]:
    """Collapse rows that share (cusip, put_call_flag) by summing value and
    shares. A single 13F can list the same security across multiple
    sub-managers / investment-discretion buckets (Coatue's many advisers are
    the canonical case); the parent-level position is the sum."""
    merged: dict[tuple[str, str], Holding13F] = {}
    for h in holdings:
        key = (h.cusip, h.put_call_flag)
        cur = merged.get(key)
        if cur is None:
            # Copy so we don't mutate the caller's objects when summing.
            merged[key] = Holding13F(
                issuer_name=h.issuer_name, cusip=h.cusip, value_usd=h.value_usd,
                shares=h.shares, shares_type=h.shares_type, put_call=h.put_call,
                title_of_class=h.title_of_class,
            )
        else:
            cur.value_usd += h.value_usd
            cur.shares += h.shares
    return list(merged.values())


# ---------------------------------------------------------------------------
# Form 4 ownership document
# ---------------------------------------------------------------------------


def parse_form4(xml_text: str | bytes) -> Form4:
    """Parse a Form 4 ownership XML. Captures issuer ticker (handy — most
    other filings only give CUSIP) and non-derivative + derivative
    transactions with their P/S/A codes."""
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    root = ET.fromstring(xml_text.strip())

    issuer = _find_local(root, "issuer")
    owner_node = _find_local(root, "reportingOwner")
    reporting_owner = ""
    if owner_node is not None:
        reporting_owner = _text_local(owner_node, "rptOwnerName")

    f4 = Form4(
        issuer_name=_text_local(issuer, "issuerName"),
        issuer_ticker=_text_local(issuer, "issuerTradingSymbol").upper(),
        issuer_cik=_text_local(issuer, "issuerCik"),
        reporting_owner=reporting_owner,
    )

    for is_deriv, table_name, txn_name in (
        (False, "nonDerivativeTable", "nonDerivativeTransaction"),
        (True, "derivativeTable", "derivativeTransaction"),
    ):
        table = _find_local(root, table_name)
        if table is None:
            continue
        for txn in _iter_local(table, txn_name):
            coding = _find_local(txn, "transactionCoding")
            amounts = _find_local(txn, "transactionAmounts")
            f4.transactions.append(
                Form4Transaction(
                    security_title=_value_of(_find_local(txn, "securityTitle")),
                    transaction_date=_to_date(_value_of(_find_local(txn, "transactionDate"))),
                    code=_text_local(coding, "transactionCode"),
                    shares=_to_float(_value_of(_find_local(amounts, "transactionShares"))),
                    price_per_share=(
                        _to_float(_value_of(_find_local(amounts, "transactionPricePerShare"))) or None
                    ),
                    acquired_disposed=_value_of(
                        _find_local(amounts, "transactionAcquiredDisposedCode")
                    ),
                    is_derivative=is_deriv,
                )
            )
    return f4


def _value_of(elem: Optional[ET.Element]) -> str:
    """Form 4 wraps most leaf fields in a ``<value>`` child (footnotes hang
    off the parent). Return the value child's text, falling back to the
    element's own text."""
    if elem is None:
        return ""
    v = _find_local(elem, "value")
    if v is not None and v.text:
        return v.text.strip()
    return (elem.text or "").strip()


# ---------------------------------------------------------------------------
# 13D / 13G cover-page (best effort)
# ---------------------------------------------------------------------------


# Cover pages label the field "PERCENT OF CLASS REPRESENTED BY AMOUNT IN
# ROW (11)" — the "(11)" is a row cross-reference, not the value. Skip
# intervening chars non-greedily and require the real figure to be followed
# by a literal '%' so we don't capture the row number.
_PCT_RE = re.compile(r"PERCENT\s+OF\s+CLASS[^%]{0,80}?(\d{1,3}(?:\.\d+)?)\s*%", re.I)
_CUSIP_RE = re.compile(r"\bCUSIP\s*(?:NO\.?|NUMBER)?\s*[:#]?\s*([0-9A-Z]{6,9})", re.I)


def parse_sc13_coverpage(text: str) -> Sc13Signal:
    """Pull percent-of-class and CUSIP from a 13D/13G text cover page.

    These filings are free-text/HTML with no reliable schema, so this is a
    heuristic. Returns whatever it can; callers treat the *existence* of the
    filing (from the submissions index) as the primary signal and these
    fields as enrichment."""
    sig = Sc13Signal()
    if not text:
        return sig
    m = _PCT_RE.search(text)
    if m:
        try:
            sig.percent_of_class = float(m.group(1))
        except ValueError:
            pass
    m = _CUSIP_RE.search(text)
    if m:
        sig.cusip = m.group(1).upper()
    return sig
