"""
Tests for src/universe.py.

All tests are network-free: they monkeypatch _fetch_html to return a
hand-crafted HTML fixture that mirrors the Wikipedia page structure.

Fixture membership (500 tickers total):
  Current members: AAPL, MSFT, GOOGL, AMZN, NVDA + TICK001..TICK495

Fixture changes (most recent first):
  2022-11-21  NVDA added,  PCOR removed
  2017-06-19  AMZN added,  YHOO removed
  2013-09-09  GOOGL added, RMBS removed

So for members_as_of("2015-01-01"):
  Changes after 2015-01-01: 2022-11-21 and 2017-06-19
    → remove NVDA (added after 2015)
    → add back PCOR (removed after 2015)
    → remove AMZN (added after 2015)
    → add back YHOO (removed after 2015)
  Change on/before 2015-01-01: 2013-09-09 (not reversed)
    → GOOGL stays in (added before 2015)
    → RMBS stays out (removed before 2015)
"""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

import src.universe as universe


# --------------------------------------------------------------------------- #
# Fixture HTML                                                                 #
# --------------------------------------------------------------------------- #

_NAMED_CURRENT = [
    ("AAPL", "Apple Inc.", "Information Technology", "0000320193"),
    ("MSFT", "Microsoft Corporation", "Information Technology", "0000789019"),
    ("GOOGL", "Alphabet Inc.", "Communication Services", "0001652044"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "0001018724"),
    ("NVDA", "NVIDIA Corporation", "Information Technology", "0001045810"),
]

_N_GENERATED = 495  # TICK001..TICK495 → total current members = 500


def _build_fixture_html() -> str:
    """Build a minimal HTML page that matches the Wikipedia S&P 500 structure."""
    # --- constituents table ---
    member_rows = []
    for ticker, security, sector, cik in _NAMED_CURRENT:
        member_rows.append(
            f"<tr><td>{ticker}</td><td>{security}</td><td>{sector}</td>"
            f"<td>Sub</td><td>HQ</td><td>2000-01-01</td><td>{cik}</td><td>1900</td></tr>"
        )
    for i in range(1, _N_GENERATED + 1):
        ticker = f"TICK{i:03d}"
        member_rows.append(
            f"<tr><td>{ticker}</td><td>Company {i}</td><td>Sector</td>"
            f"<td>Sub</td><td>HQ</td><td>2000-01-01</td><td>{i:010d}</td><td>1990</td></tr>"
        )

    constituents_table = textwrap.dedent(f"""
        <table id="constituents" class="wikitable sortable">
          <thead>
            <tr>
              <th>Symbol</th><th>Security</th><th>GICS Sector</th>
              <th>GICS Sub-Industry</th><th>Headquarters Location</th>
              <th>Date added</th><th>CIK</th><th>Founded</th>
            </tr>
          </thead>
          <tbody>
            {''.join(member_rows)}
          </tbody>
        </table>
    """)

    # --- changes table (multi-level header, date cell uses rowspan=2 demo) ---
    changes_table = textwrap.dedent("""
        <table id="changes" class="wikitable">
          <thead>
            <tr>
              <th rowspan="2">Date</th>
              <th colspan="2">Added</th>
              <th colspan="2">Removed</th>
              <th rowspan="2">Reason</th>
            </tr>
            <tr>
              <th>Ticker</th><th>Security</th>
              <th>Ticker</th><th>Security</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>November 21, 2022</td>
              <td>NVDA</td><td>NVIDIA Corporation</td>
              <td>PCOR</td><td>Procore Technologies</td>
              <td>Market cap eligibility</td>
            </tr>
            <tr>
              <td>June 19, 2017</td>
              <td>AMZN</td><td>Amazon.com Inc.</td>
              <td>YHOO</td><td>Yahoo Inc.</td>
              <td>Market cap eligibility</td>
            </tr>
            <tr>
              <td>September 9, 2013</td>
              <td>GOOGL</td><td>Alphabet Inc.</td>
              <td>RMBS</td><td>Rambus Inc.</td>
              <td>Market cap eligibility</td>
            </tr>
          </tbody>
        </table>
    """)

    return f"<html><body>{constituents_table}{changes_table}</body></html>"


_FIXTURE_HTML = _build_fixture_html()


@pytest.fixture(autouse=True)
def patch_fetch_html(monkeypatch):
    """Replace network fetch with the in-memory fixture for every test."""
    monkeypatch.setattr(universe, "_fetch_html", lambda force_refresh=False: _FIXTURE_HTML)


# --------------------------------------------------------------------------- #
# members_as_of tests                                                          #
# --------------------------------------------------------------------------- #


class TestMembersAsOf:

    def test_today_returns_approximately_500_tickers(self):
        today = pd.Timestamp("2026-04-26")
        members = members_as_of_today = universe.members_as_of(today)
        assert 480 <= len(members) <= 520, (
            f"Expected ~500 members for today, got {len(members)}"
        )

    def test_today_returns_sorted_list_of_strings(self):
        today = pd.Timestamp("2026-04-26")
        members = universe.members_as_of(today)
        assert isinstance(members, list)
        assert all(isinstance(t, str) for t in members)
        assert members == sorted(members)

    def test_2015_excludes_companies_added_after_that_date(self):
        """NVDA (added 2022-11-21) and AMZN (added 2017-06-19) must not appear."""
        members = universe.members_as_of(pd.Timestamp("2015-01-01"))
        assert "NVDA" not in members, "NVDA was added 2022-11-21, should not be in 2015 list"
        assert "AMZN" not in members, "AMZN was added 2017-06-19, should not be in 2015 list"

    def test_2015_includes_companies_removed_after_that_date(self):
        """PCOR (removed 2022-11-21) and YHOO (removed 2017-06-19) must appear."""
        members = universe.members_as_of(pd.Timestamp("2015-01-01"))
        assert "PCOR" in members, "PCOR was removed 2022-11-21, should be in 2015 list"
        assert "YHOO" in members, "YHOO was removed 2017-06-19, should be in 2015 list"

    def test_2015_includes_company_added_before_that_date(self):
        """GOOGL was added 2013-09-09, so it should appear in the 2015 list."""
        members = universe.members_as_of(pd.Timestamp("2015-01-01"))
        assert "GOOGL" in members, "GOOGL was added 2013-09-09, should be in 2015 list"

    def test_2015_excludes_company_removed_before_that_date(self):
        """RMBS was removed 2013-09-09, before our query date, so it must NOT appear."""
        members = universe.members_as_of(pd.Timestamp("2015-01-01"))
        assert "RMBS" not in members, "RMBS was removed 2013-09-09, should not be in 2015 list"

    def test_very_early_date_restores_all_removed_tickers(self):
        """Query for 2000-01-01: all three changes are after this date, so NVDA,
        AMZN, GOOGL are removed and PCOR, YHOO, RMBS are added back."""
        members = universe.members_as_of(pd.Timestamp("2000-01-01"))
        assert "NVDA" not in members
        assert "AMZN" not in members
        assert "GOOGL" not in members
        assert "PCOR" in members
        assert "YHOO" in members
        assert "RMBS" in members

    def test_boundary_on_change_date_excludes_that_days_changes(self):
        """members_as_of uses strictly-after comparison: a change on 2022-11-21
        should NOT be reversed when querying exactly that date."""
        members = universe.members_as_of(pd.Timestamp("2022-11-21"))
        # The 2022-11-21 change is NOT after the query date, so it is not reversed.
        assert "NVDA" in members
        assert "PCOR" not in members


# --------------------------------------------------------------------------- #
# current_members / cik_map tests                                             #
# --------------------------------------------------------------------------- #


class TestCurrentMembers:

    def test_returns_dataframe_with_required_columns(self):
        df = universe.current_members()
        for col in ("ticker", "security", "sector", "cik"):
            assert col in df.columns, f"Missing column: {col}"

    def test_returns_approximately_500_rows(self):
        df = universe.current_members()
        assert 480 <= len(df) <= 520

    def test_known_tickers_present_with_correct_cik(self):
        df = universe.current_members().set_index("ticker")
        assert "AAPL" in df.index
        assert df.loc["AAPL", "cik"] == "0000320193"
        assert "MSFT" in df.index
        assert df.loc["MSFT", "cik"] == "0000789019"


class TestCikMap:

    def test_returns_dict(self):
        m = universe.cik_map()
        assert isinstance(m, dict)

    def test_known_ticker_cik(self):
        m = universe.cik_map()
        assert m["AAPL"] == "0000320193"
        assert m["MSFT"] == "0000789019"
