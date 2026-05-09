"""Financial Modeling Prep (Ultimate tier) provider.

Implements `FundamentalsProvider`. FMP Ultimate gives us:

* Statements: `/api/v3/income-statement/{symbol}`,
              `/api/v3/balance-sheet-statement/{symbol}`,
              `/api/v3/cash-flow-statement/{symbol}`
* Ratios:     `/api/v3/ratios/{symbol}`
* TTM data:   `/api/v3/income-statement-as-reported/{symbol}`
* Owner Earnings (custom): we compute from CFO − maintenance capex, since
  FMP exposes line-item capex breakdowns under `/api/v4/owner_earnings`.

The Fundamentals analyst in the upstream repo expects a *string* return —
a markdown table or section headers it can parse into prose. We render
that here so the existing `agent_utils.get_fundamentals` keeps working
when the vendor swaps to FMP.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import AuthError, ProviderError
from .cache import cached
from .http import AsyncHttpClient

logger = logging.getLogger(__name__)


class FmpProvider:
    name = "fmp"

    def __init__(self, api_key: Optional[str] = None) -> None:
        api_key = api_key or os.getenv("FMP_API_KEY")
        if not api_key:
            raise AuthError("fmp")
        self._api_key = api_key
        self._http = AsyncHttpClient(
            provider_name="fmp",
            base_url="https://financialmodelingprep.com",
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # FundamentalsProvider — string-returning, agent-friendly
    # ------------------------------------------------------------------

    async def get_fundamentals(self, symbol: str, period: str = "annual") -> str:
        ratios = await self._ratios(symbol, period)
        if not ratios:
            return f"No fundamentals data available for {symbol}."
        latest = ratios[0]
        return _md_kv_block(
            f"Fundamentals — {symbol} ({period}, latest period: {latest.get('date')})",
            [
                ("Gross margin",        _pct(latest.get("grossProfitMargin"))),
                ("Operating margin",    _pct(latest.get("operatingProfitMargin"))),
                ("Net margin",          _pct(latest.get("netProfitMargin"))),
                ("ROE",                 _pct(latest.get("returnOnEquity"))),
                ("ROIC",                _pct(latest.get("returnOnCapitalEmployed"))),
                ("Current ratio",       _num(latest.get("currentRatio"))),
                ("Debt / Equity",       _num(latest.get("debtEquityRatio"))),
                ("Interest coverage",   _num(latest.get("interestCoverage"))),
                ("FCF yield",           _pct(latest.get("freeCashFlowYield"))),
                ("P/E",                 _num(latest.get("priceEarningsRatio"))),
                ("EV/EBITDA",           _num(latest.get("enterpriseValueMultiple"))),
            ],
        )

    async def get_income_statement(self, symbol: str, period: str = "annual") -> str:
        rows = await self._statement(symbol, "income-statement", period)
        return _md_statement_table(
            f"Income Statement — {symbol} ({period}, last 4 periods)",
            rows[:4],
            [
                ("Revenue",            "revenue"),
                ("Gross profit",       "grossProfit"),
                ("Operating income",   "operatingIncome"),
                ("Net income",         "netIncome"),
                ("EPS (diluted)",      "epsdiluted"),
                ("R&D expense",        "researchAndDevelopmentExpenses"),
                ("EBITDA",             "ebitda"),
            ],
        )

    async def get_balance_sheet(self, symbol: str, period: str = "annual") -> str:
        rows = await self._statement(symbol, "balance-sheet-statement", period)
        return _md_statement_table(
            f"Balance Sheet — {symbol} ({period}, last 4 periods)",
            rows[:4],
            [
                ("Cash & equivalents",    "cashAndCashEquivalents"),
                ("Short-term investments","shortTermInvestments"),
                ("Total current assets",  "totalCurrentAssets"),
                ("Total assets",          "totalAssets"),
                ("Total current liab.",   "totalCurrentLiabilities"),
                ("Long-term debt",        "longTermDebt"),
                ("Total liabilities",     "totalLiabilities"),
                ("Total equity",          "totalStockholdersEquity"),
            ],
        )

    async def get_cashflow(self, symbol: str, period: str = "annual") -> str:
        rows = await self._statement(symbol, "cash-flow-statement", period)
        return _md_statement_table(
            f"Cash Flow — {symbol} ({period}, last 4 periods)",
            rows[:4],
            [
                ("Operating CF",            "operatingCashFlow"),
                ("Investing CF",            "netCashUsedForInvestingActivites"),
                ("Financing CF",            "netCashUsedProvidedByFinancingActivities"),
                ("Capex",                   "capitalExpenditure"),
                ("Free cash flow",          "freeCashFlow"),
                ("Stock-based comp.",       "stockBasedCompensation"),
                ("Dividends paid",          "dividendsPaid"),
                ("Buybacks (net)",          "commonStockRepurchased"),
            ],
        )

    # ------------------------------------------------------------------
    # Insider trading (Form 4) — used by the momentum-exhaustion module
    # ------------------------------------------------------------------

    async def get_insider_sell_pressure(
        self, symbol: str, *, lookback_days: int = 180,
    ) -> dict[str, Any]:
        """Aggregate Form 4 transactions into an "insider selling acceleration"
        signal. We compare the dollar value of officer / director / 10%-owner
        sales in the last 30 days to the prior 30-180 day baseline.

        A meaningful signal looks like: 30-day sales of $5M when the
        baseline monthly average is $500K — that's 10x acceleration on a
        non-trivial base, the kind of cluster that often precedes
        distribution.

        Returns a dict callers can read directly. ``accelerating`` is the
        boolean trip flag the exhaustion module reads.

        Filters:
          * Only treats sales (S, S-Sale, D / disposed) as sells; awards,
            grants, and option exercises are excluded.
          * Filters to officer / director / 10% owners — these are the
            people whose selling actually carries informational value.
            Random employee 10b5-1 sales add noise; we drop them.
        """
        rows = await self._insider_trading_raw(symbol)
        if not rows:
            return {
                "sells_30d_usd": 0.0,
                "sells_baseline_monthly_usd": 0.0,
                "ratio": 0.0,
                "n_sellers_30d": 0,
                "accelerating": False,
                "detail": "no insider trades in lookback",
            }

        from datetime import date, timedelta
        today = date.today()
        cutoff_30d = today - timedelta(days=30)
        cutoff_180d = today - timedelta(days=lookback_days)

        sells_30d_usd: float = 0.0
        sells_baseline_usd: float = 0.0
        sellers_30d: set[str] = set()

        for r in rows:
            tx_date = _parse_date(r.get("transactionDate") or r.get("filingDate"))
            if tx_date is None or tx_date < cutoff_180d:
                continue
            if not _is_insider_sale(r):
                continue
            owner_type = (r.get("typeOfOwner") or "").lower()
            # Filter to officer / director / 10%-owner — these are the
            # informationally-loaded sellers. Exclude employee-grant
            # automatic dispositions and random rank-and-file.
            if not any(k in owner_type for k in ("officer", "director", "10")):
                continue
            qty = float(r.get("securitiesTransacted") or 0)
            price = float(r.get("price") or 0)
            usd = qty * price
            if usd <= 0:
                continue
            if tx_date >= cutoff_30d:
                sells_30d_usd += usd
                sellers_30d.add(str(r.get("reportingName") or r.get("reportingCik") or "?"))
            else:
                sells_baseline_usd += usd

        # Normalize the 30-180-day baseline to a per-month rate
        baseline_months = max(1.0, (lookback_days - 30) / 30.0)
        baseline_monthly = sells_baseline_usd / baseline_months
        ratio = (sells_30d_usd / baseline_monthly) if baseline_monthly > 0 else 0.0

        # Trip when the 30d total is at least 2x the baseline AND the 30d
        # total is materially large ($500K floor) AND there's more than
        # one seller (single-seller spikes are often planned 10b5-1).
        accelerating = (
            ratio >= 2.0
            and sells_30d_usd >= 500_000
            and len(sellers_30d) >= 2
        )

        return {
            "sells_30d_usd": round(sells_30d_usd, 2),
            "sells_baseline_monthly_usd": round(baseline_monthly, 2),
            "ratio": round(ratio, 2),
            "n_sellers_30d": len(sellers_30d),
            "accelerating": accelerating,
            "detail": (
                f"30d=${sells_30d_usd/1e6:.2f}M from {len(sellers_30d)} "
                f"insider(s) vs baseline ${baseline_monthly/1e6:.2f}M/mo"
            ),
        }

    # ------------------------------------------------------------------
    # Analyst grades + price targets — feeds the 7th momentum-exhaustion
    # signal ("upgrade after a major run") and the bear research case.
    # ------------------------------------------------------------------

    @cached(ttl_s=4 * 3600, namespace="fmp.grades")
    async def get_analyst_grade_changes(self, symbol: str) -> list[dict[str, Any]]:
        """Per-symbol analyst rating changes (upgrades / downgrades / new
        coverage initiations) with prior + new grade and the firm.

        Returns FMP raw rows; caller filters by date window.
        """
        body = await self._http.get_json(
            "/stable/grades", params={"symbol": symbol, "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body if isinstance(body, list) else []

    @cached(ttl_s=4 * 3600, namespace="fmp.price_target_summary")
    async def get_price_target_summary(self, symbol: str) -> dict[str, Any]:
        """Rolling counts + average price targets across last month, quarter,
        and year. Used to detect price-target inflation (analysts chasing
        the stock up after the run)."""
        body = await self._http.get_json(
            "/stable/price-target-summary",
            params={"symbol": symbol, "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body[0] if isinstance(body, list) and body else {}

    @cached(ttl_s=4 * 3600, namespace="fmp.price_target_consensus")
    async def get_price_target_consensus(self, symbol: str) -> dict[str, Any]:
        """Current consensus price target (high / low / median / mean)."""
        body = await self._http.get_json(
            "/stable/price-target-consensus",
            params={"symbol": symbol, "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body[0] if isinstance(body, list) and body else {}

    # ------------------------------------------------------------------
    # 13F institutional ownership — quarterly, feeds exit-pressure as
    # a "smart money flow" sub-score
    # ------------------------------------------------------------------

    async def get_institutional_position_summary(
        self, symbol: str, *, year: int, quarter: int,
    ) -> dict[str, Any]:
        """Quarterly snapshot of institutional ownership for one symbol.

        Returns ``{}`` when the period isn't available. The endpoint
        requires both year and quarter — caller picks the most recent
        completed quarter.
        """
        body = await self._http.get_json(
            "/stable/institutional-ownership/symbol-positions-summary",
            params={"symbol": symbol, "year": year, "quarter": quarter,
                    "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body[0] if isinstance(body, list) and body else {}

    # ------------------------------------------------------------------
    # Earnings call transcripts — feeds the thesis-break detector
    # ------------------------------------------------------------------

    @cached(ttl_s=30 * 24 * 3600, namespace="fmp.transcript")
    async def get_earnings_transcript(
        self, symbol: str, *, year: int, quarter: int,
    ) -> Optional[dict[str, Any]]:
        """Full earnings call transcript content for a quarter.

        Path is ``/stable/earning-call-transcript`` (not ``earnings`` —
        FMP uses the singular form). Cached 30 days because transcripts
        don't change after the call.
        """
        body = await self._http.get_json(
            "/stable/earning-call-transcript",
            params={"symbol": symbol, "year": year, "quarter": quarter,
                    "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        if isinstance(body, list) and body:
            return body[0]
        return None

    # ------------------------------------------------------------------
    # SEC filings — 8-K watcher feeds the thesis-break detector
    # ------------------------------------------------------------------

    async def get_recent_8ks(
        self, *, from_date: str, to_date: Optional[str] = None,
        page: int = 0, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Universe-wide 8-K sweep over a date range.

        Returns the FMP raw shape; callers filter to symbols of interest.
        Date strings are 'YYYY-MM-DD'. Endpoint:
            /stable/sec-filings-8k?from=...&to=...&page=N&limit=M
        Each row carries ``symbol`` / ``filingDate`` / ``acceptedDate``
        / ``hasFinancials`` / ``link`` / ``finalLink``.

        Not cached at this layer because the watch module wants a fresh
        sweep on each tick (with its own dedup logic).
        """
        params = {
            "from": from_date, "page": page, "limit": limit,
            "apikey": self._api_key,
        }
        if to_date:
            params["to"] = to_date
        body = await self._http.get_json("/stable/sec-filings-8k", params=params)
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body if isinstance(body, list) else []

    async def get_filings_by_symbol(
        self, symbol: str, *, from_date: str, to_date: Optional[str] = None,
        page: int = 0, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Per-symbol filing search across all form types — used when a
        sweep flags a symbol and we want the full filing history for the
        thesis-break detector to look back at past 8-Ks.

        FMP requires both ``from`` and ``to`` on this endpoint; we
        default ``to`` to today when the caller doesn't provide it.
        """
        if not to_date:
            from datetime import date
            to_date = date.today().isoformat()
        params = {
            "symbol": symbol, "from": from_date, "to": to_date,
            "page": page, "limit": limit,
            "apikey": self._api_key,
        }
        body = await self._http.get_json(
            "/stable/sec-filings-search/symbol", params=params,
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body if isinstance(body, list) else []

    @cached(ttl_s=4 * 3600, namespace="fmp.insider")
    async def _insider_trading_raw(self, symbol: str) -> list[dict[str, Any]]:
        """Form 4 rows from FMP stable. 4-hour cache — insider filings are
        not high-frequency.

        Path is ``/stable/insider-trading/search`` (the per-symbol search
        endpoint). The bare ``/stable/insider-trading`` path 404s; FMP only
        exposes the per-symbol shape behind the ``/search`` suffix.
        """
        body = await self._http.get_json(
            "/stable/insider-trading/search",
            params={"symbol": symbol, "page": 0, "limit": 100,
                    "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body if isinstance(body, list) else []

    async def get_owner_earnings(self, symbol: str, period: str = "annual") -> str:
        body = await self._http.get_json(
            "/stable/owner-earnings",
            params={"symbol": symbol, "apikey": self._api_key},
        )
        rows = body if isinstance(body, list) else body.get("data") or []
        if not rows:
            return f"No owner-earnings data for {symbol}."
        latest = rows[0]
        return _md_kv_block(
            f"Owner Earnings — {symbol} ({latest.get('date')})",
            [
                ("Owner earnings",  _money(latest.get("ownersEarnings"))),
                ("Per share",       _num(latest.get("ownersEarningsPerShare"))),
                ("vs. Net income",  _money(latest.get("netIncome"))),
                ("vs. FCF",         _money(latest.get("freeCashFlow"))),
                ("Maint. capex",    _money(latest.get("maintenanceCapex"))),
            ],
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @cached(ttl_s=24 * 3600, namespace="fmp.statement")
    async def _statement(self, symbol: str, kind: str, period: str) -> list[dict[str, Any]]:
        # FMP migrated from /api/v3/<kind>/<symbol> (legacy, deprecated 2025-08-31)
        # to /stable/<kind>?symbol=<symbol>. The new shape returns the same JSON
        # payload, so downstream md formatters don't need to change.
        body = await self._http.get_json(
            f"/stable/{kind}",
            params={"symbol": symbol, "period": period, "limit": 8, "apikey": self._api_key},
        )
        if isinstance(body, dict) and "Error Message" in body:
            raise ProviderError("fmp", body["Error Message"])
        return body if isinstance(body, list) else []

    @cached(ttl_s=24 * 3600, namespace="fmp.ratios")
    async def _ratios(self, symbol: str, period: str) -> list[dict[str, Any]]:
        body = await self._http.get_json(
            f"/stable/ratios",
            params={"symbol": symbol, "period": period, "limit": 4, "apikey": self._api_key},
        )
        return body if isinstance(body, list) else []


# ---------------------------------------------------------------------------
# Markdown formatters — kept here so the agent-facing string output is
# deterministic and easy to regenerate when we tweak rubric.
# ---------------------------------------------------------------------------


def _md_kv_block(header: str, kvs: list[tuple[str, str]]) -> str:
    width = max(len(k) for k, _ in kvs) + 2
    lines = [f"## {header}", ""]
    for k, v in kvs:
        lines.append(f"- **{k:<{width}}** {v}")
    return "\n".join(lines)


def _md_statement_table(header: str, rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return f"## {header}\n\n_No data available._"
    dates = [str(r.get("date") or r.get("calendarYear") or "?") for r in rows]
    out = [f"## {header}", "", "| Line item | " + " | ".join(dates) + " |",
           "| --- | " + " | ".join(["---"] * len(dates)) + " |"]
    for label, key in cols:
        cells = [_money(r.get(key)) for r in rows]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _parse_date(s: Any) -> Optional[Any]:
    """Parse FMP-style date strings ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS')
    into a date. Returns None on any failure — caller filters."""
    if not s:
        return None
    from datetime import date, datetime
    try:
        s = str(s).split(" ")[0]
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _is_insider_sale(row: dict[str, Any]) -> bool:
    """True if this Form 4 row represents an open-market sale.

    FMP's transactionType field can be ``S``, ``S-Sale``, ``Sale``, etc.
    Older rows also use ``acquistionOrDisposition`` = ``D`` (disposed).
    Excludes awards / grants / option exercises which carry no signal.
    """
    tt = str(row.get("transactionType") or "").upper()
    aod = str(row.get("acquistionOrDisposition") or row.get("acquisitionOrDisposition") or "").upper()
    if tt.startswith("S") or "SALE" in tt:
        return True
    # Exclude clearly non-sale dispositions (gifts, conversions etc.)
    excluded_codes = ("A", "M", "F", "G", "I", "J", "K", "U", "W", "X", "Z")
    if aod == "D" and not any(c in tt for c in excluded_codes):
        return True
    return False


def _money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(n) >= 1e9:
        return f"${n / 1e9:,.2f}B"
    if abs(n) >= 1e6:
        return f"${n / 1e6:,.2f}M"
    if abs(n) >= 1e3:
        return f"${n / 1e3:,.2f}K"
    return f"${n:,.2f}"


def _num(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:,.2f}%"
    except (TypeError, ValueError):
        return str(v)
