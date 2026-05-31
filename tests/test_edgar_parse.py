"""Unit tests for the EDGAR filing parsers.

Covers the two correctness traps explicitly: the 2023-01-03 thousands→whole-
dollars value cutover, and namespace-blind XML parsing. Plus Coatue-style
sub-manager dedup and Form 4 ticker/transaction extraction.
"""

from datetime import date

import pytest

from tradingagents.dataflows.edgar_parse import (
    Holding13F,
    dedupe_holdings,
    normalize_13f_value,
    parse_13f_infotable,
    parse_form4,
    parse_sc13_coverpage,
    parse_submissions,
)

pytestmark = pytest.mark.unit


# --- 13F infotable: default namespace, whole-dollar era ---------------------

INFOTABLE_NS = """<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>67066G104</cusip>
    <value>1500000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>1200000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority><Sole>1200000</Sole><Shared>0</Shared><None>0</None></votingAuthority>
  </infoTable>
  <infoTable>
    <nameOfIssuer>NVIDIA CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>67066G104</cusip>
    <value>50000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>40000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <putCall>Put</putCall>
    <votingAuthority><Sole>0</Sole><Shared>0</Shared><None>40000</None></votingAuthority>
  </infoTable>
</informationTable>
"""

# Same data but with an explicit ns1: prefix (some filing agents emit this).
INFOTABLE_PREFIXED = """<?xml version="1.0"?>
<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <ns1:infoTable>
    <ns1:nameOfIssuer>APPLE INC</ns1:nameOfIssuer>
    <ns1:cusip>037833100</ns1:cusip>
    <ns1:value>2000</ns1:value>
    <ns1:shrsOrPrnAmt>
      <ns1:sshPrnamt>5000</ns1:sshPrnamt>
      <ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
    </ns1:shrsOrPrnAmt>
  </ns1:infoTable>
</ns1:informationTable>
"""


class TestValueNormalization:
    def test_post_cutover_is_whole_dollars(self):
        assert normalize_13f_value(1_500_000_000, filed_at=date(2024, 5, 15)) == 1_500_000_000

    def test_pre_cutover_multiplies_by_thousand(self):
        # A 2021 filing reporting "1500000" means $1.5B.
        assert normalize_13f_value(1_500_000, filed_at=date(2021, 8, 14)) == 1_500_000_000

    def test_cutover_boundary(self):
        # 2023-01-01 onward = whole dollars.
        assert normalize_13f_value(1000, filed_at=date(2023, 1, 1)) == 1000
        assert normalize_13f_value(1000, filed_at=date(2022, 12, 31)) == 1_000_000

    def test_unknown_filed_at_defaults_to_thousands(self):
        assert normalize_13f_value(1000, filed_at=None) == 1_000_000


class Test13FInfotable:
    def test_parses_default_namespace_and_normalises(self):
        hs = parse_13f_infotable(INFOTABLE_NS, filed_at=date(2024, 5, 15))
        assert len(hs) == 2
        long_row = next(h for h in hs if h.put_call is None)
        assert long_row.issuer_name == "NVIDIA CORP"
        assert long_row.cusip == "67066G104"
        assert long_row.value_usd == 1_500_000_000  # already whole dollars
        assert long_row.shares == 1_200_000
        assert long_row.put_call_flag == ""

    def test_option_row_kept_distinct(self):
        hs = parse_13f_infotable(INFOTABLE_NS, filed_at=date(2024, 5, 15))
        put_row = next(h for h in hs if h.put_call)
        assert put_row.put_call_flag == "P"

    def test_namespace_prefixed_parses_with_thousands(self):
        # Pre-cutover prefixed filing: value 2000 (thousands) → $2,000,000.
        hs = parse_13f_infotable(INFOTABLE_PREFIXED, filed_at=date(2021, 2, 12))
        assert len(hs) == 1
        assert hs[0].issuer_name == "APPLE INC"
        assert hs[0].value_usd == 2_000_000

    def test_empty_input(self):
        assert parse_13f_infotable("", filed_at=date(2024, 1, 1)) == []


class TestDedup:
    def test_sub_manager_rows_summed(self):
        # Coatue case: same CUSIP across two sub-advisers.
        rows = [
            Holding13F("FOO", "111111111", 100.0, 10, "SH"),
            Holding13F("FOO", "111111111", 250.0, 25, "SH"),
            Holding13F("FOO", "111111111", 30.0, 3, "SH", put_call="Call"),
        ]
        merged = dedupe_holdings(rows)
        by_key = {(h.cusip, h.put_call_flag): h for h in merged}
        assert by_key[("111111111", "")].value_usd == 350.0
        assert by_key[("111111111", "")].shares == 35
        # Option row stays separate.
        assert by_key[("111111111", "C")].value_usd == 30.0

    def test_does_not_mutate_input(self):
        rows = [Holding13F("FOO", "1", 100.0, 10, "SH"),
                Holding13F("FOO", "1", 100.0, 10, "SH")]
        dedupe_holdings(rows)
        assert rows[0].value_usd == 100.0  # original untouched


# --- Form 4 -----------------------------------------------------------------

FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>COOK TIMOTHY D</rptOwnerName></reportingOwnerId>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-04-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>10000</value></transactionShares>
        <transactionPricePerShare><value>170.50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2024-04-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>4000</value></transactionShares>
        <transactionPricePerShare><value>171.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


class TestForm4:
    def test_extracts_issuer_ticker_and_owner(self):
        f4 = parse_form4(FORM4)
        assert f4.issuer_ticker == "AAPL"
        assert f4.reporting_owner == "COOK TIMOTHY D"
        assert len(f4.transactions) == 2

    def test_net_open_market(self):
        f4 = parse_form4(FORM4)
        # +10000 buy, -4000 sale = +6000 net.
        assert f4.net_open_market_shares == 6000
        buy = f4.transactions[0]
        assert buy.code == "P" and buy.price_per_share == 170.50


# --- submissions index ------------------------------------------------------

SUBMISSIONS = {
    "cik": "1541617",
    "name": "ALTIMETER CAPITAL MANAGEMENT, LP",
    "filings": {
        "recent": {
            "accessionNumber": ["0001541617-24-000012", "0001541617-23-000031"],
            "form": ["13F-HR", "SC 13D"],
            "filingDate": ["2024-05-15", "2023-11-02"],
            "reportDate": ["2024-03-31", ""],
            "primaryDocument": ["primary_doc.xml", "sc13d.htm"],
        }
    },
}


class TestSubmissions:
    def test_parses_parallel_arrays(self):
        refs = parse_submissions(SUBMISSIONS)
        assert len(refs) == 2
        f13 = refs[0]
        assert f13.form == "13F-HR"
        assert f13.filed_at == date(2024, 5, 15)
        assert f13.period_end == date(2024, 3, 31)
        assert f13.accession_nodashes == "000154161724000012"
        # Missing reportDate tolerated.
        assert refs[1].period_end is None


class TestSc13:
    def test_coverpage_extracts_pct_and_cusip(self):
        text = "CUSIP No. 67066G104\n... PERCENT OF CLASS REPRESENTED BY AMOUNT IN ROW (11): 6.5%"
        sig = parse_sc13_coverpage(text)
        assert sig.cusip == "67066G104"
        assert sig.percent_of_class == 6.5
