"""
Tests for src/edgar_pull.py.

Network policy
--------------
The AAPL fixture (tests/fixtures/edgar/CIK0000320193.json) must exist on disk
before running these tests. A helper at the bottom of this file can download it
once; the actual test functions never hit the network — they read from the file.

To download the fixture for the first time run:
    python -m pytest tests/test_edgar_pull.py --download-fixture -s

Or call _download_fixture() manually from a Python REPL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.edgar_pull import _extract_eps_facts, derive_quarterly_eps

_COLS = ["ticker", "cik", "period_end", "eps", "filed", "form", "fp", "accn"]

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "edgar"
AAPL_CIK = 320193
AAPL_FIXTURE = FIXTURE_DIR / "CIK0000320193.json"


# --------------------------------------------------------------------------- #
# Fixture download helper (run once; not a pytest test)                       #
# --------------------------------------------------------------------------- #


def _download_fixture() -> None:
    """Download the AAPL companyfacts JSON from SEC EDGAR and save it as a
    test fixture. Call this once before running the test suite.
    Requires network access."""
    import requests
    from src.edgar_pull import USER_AGENT, EDGAR_FACTS_URL

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    url = EDGAR_FACTS_URL.format(cik=AAPL_CIK)
    print(f"Fetching {url} ...")
    print(f"User-Agent: {USER_AGENT}")
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    AAPL_FIXTURE.write_text(resp.text, encoding="utf-8")
    size_kb = AAPL_FIXTURE.stat().st_size / 1024
    print(f"Saved {AAPL_FIXTURE} ({size_kb:.0f} KB)")


def pytest_addoption(parser):
    parser.addoption(
        "--download-fixture",
        action="store_true",
        default=False,
        help="Download the AAPL EDGAR fixture from the live SEC API.",
    )


def pytest_configure(config):
    if config.getoption("--download-fixture", default=False):
        _download_fixture()


# --------------------------------------------------------------------------- #
# Shared fixture                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def aapl_data() -> dict:
    if not AAPL_FIXTURE.exists():
        pytest.skip(
            f"AAPL fixture not found at {AAPL_FIXTURE}. "
            "Run: python -m pytest tests/test_edgar_pull.py --download-fixture -s"
        )
    with AAPL_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Tests against the AAPL fixture                                              #
# --------------------------------------------------------------------------- #


class TestExtractEpsFromAaplFixture:

    def test_returns_at_least_40_quarterly_observations(self, aapl_data):
        df = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        assert len(df) >= 40, f"Expected ≥40 observations, got {len(df)}"

    def test_all_eps_values_are_positive(self, aapl_data):
        """Apple has been profitable in every reported quarter in the dataset."""
        df = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        non_positive = df[df["eps"] <= 0]
        assert len(non_positive) == 0, (
            f"Expected all EPS > 0 for AAPL, found {len(non_positive)} non-positive rows:\n"
            f"{non_positive[['period_end', 'eps', 'form', 'fp']].to_string()}"
        )

    def test_output_has_all_required_columns(self, aapl_data):
        df = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        for col in _COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_period_end_dates_are_unique_after_deduplication(self, aapl_data):
        """Point-in-time dedup should yield one row per period_end."""
        df = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        dupes = df[df.duplicated(subset=["cik", "period_end"])]
        assert len(dupes) == 0, (
            f"Duplicate (cik, period_end) pairs after dedup:\n{dupes.to_string()}"
        )

    def test_period_end_is_datetime(self, aapl_data):
        df = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        assert pd.api.types.is_datetime64_any_dtype(df["period_end"])
        assert pd.api.types.is_datetime64_any_dtype(df["filed"])

    def test_ticker_and_cik_columns_populated(self, aapl_data):
        df = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        assert (df["ticker"] == "AAPL").all()
        assert (df["cik"] == AAPL_CIK).all()


# --------------------------------------------------------------------------- #
# Point-in-time correctness: amendment must be discarded                      #
# --------------------------------------------------------------------------- #


class TestPointInTimeDeduplication:

    def _make_synthetic_json(self, facts_usd_shares: list[dict]) -> dict:
        return {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {
                        "units": {"USD/shares": facts_usd_shares}
                    }
                }
            }
        }

    def test_earliest_filing_kept_not_amendment(self):
        """Two entries for the same period: original 10-Q filed at T,
        then a 10-Q/A filed at T+90 with a different (higher) EPS value.
        The parser must keep the original (lower) value."""
        data = self._make_synthetic_json([
            {
                "end": "2022-09-30",
                "val": 1.29,
                "filed": "2022-11-04",
                "form": "10-Q",
                "fp": "Q4",
                "accn": "0000000001",
            },
            {
                "end": "2022-09-30",
                "val": 1.35,          # amended value — must be discarded
                "filed": "2023-02-03",  # filed 91 days later
                "form": "10-Q/A",
                "fp": "Q4",
                "accn": "0000000002",
            },
        ])
        df = _extract_eps_facts(data, cik=999, ticker="TEST")
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        assert df.iloc[0]["eps"] == pytest.approx(1.29), (
            f"Expected original EPS 1.29, got {df.iloc[0]['eps']}"
        )
        assert df.iloc[0]["form"] == "10-Q"

    def test_multiple_periods_each_deduplicated_independently(self):
        """Three periods, each with an original and an amendment.
        All three originals should survive; all three amendments discarded."""
        periods = [
            # (period_end, orig_val, amend_val, orig_filed, amend_filed, fp)
            ("2021-03-31", 1.10, 1.15, "2021-05-01", "2021-08-01", "Q2"),
            ("2021-06-30", 1.20, 1.25, "2021-08-02", "2021-11-01", "Q3"),
            ("2021-09-30", 1.30, 1.35, "2021-11-02", "2022-02-01", "Q4"),
        ]
        facts = []
        for i, (end, orig_val, amend_val, orig_filed, amend_filed, fp) in enumerate(periods):
            facts.append({
                "end": end, "val": orig_val, "filed": orig_filed,
                "form": "10-Q", "fp": fp, "accn": f"orig-{i}",
            })
            facts.append({
                "end": end, "val": amend_val, "filed": amend_filed,
                "form": "10-Q/A", "fp": fp, "accn": f"amend-{i}",
            })

        data = self._make_synthetic_json(facts)
        df = _extract_eps_facts(data, cik=999, ticker="TEST")
        assert len(df) == 3
        assert list(df["eps"].round(2)) == [1.10, 1.20, 1.30]
        assert list(df["form"]) == ["10-Q", "10-Q", "10-Q"]


# --------------------------------------------------------------------------- #
# Fallback to EarningsPerShareBasic                                           #
# --------------------------------------------------------------------------- #


class TestEpsFallback:

    def test_falls_back_to_basic_when_diluted_absent(self):
        data = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareBasic": {
                        "units": {
                            "USD/shares": [
                                {
                                    "end": "2022-03-31", "val": 2.10,
                                    "filed": "2022-05-01",
                                    "form": "10-Q", "fp": "Q2",
                                    "accn": "0000000001",
                                }
                            ]
                        }
                    }
                }
            }
        }
        df = _extract_eps_facts(data, cik=999, ticker="TEST")
        assert len(df) == 1
        assert df.iloc[0]["eps"] == pytest.approx(2.10)

    def test_falls_back_to_basic_when_diluted_units_empty(self):
        data = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {"units": {}},
                    "EarningsPerShareBasic": {
                        "units": {
                            "USD/shares": [
                                {
                                    "end": "2022-03-31", "val": 3.00,
                                    "filed": "2022-05-01",
                                    "form": "10-Q", "fp": "Q2",
                                    "accn": "0000000001",
                                }
                            ]
                        }
                    },
                }
            }
        }
        df = _extract_eps_facts(data, cik=999, ticker="TEST")
        assert len(df) == 1
        assert df.iloc[0]["eps"] == pytest.approx(3.00)


# --------------------------------------------------------------------------- #
# Graceful handling of missing EPS fields                                     #
# --------------------------------------------------------------------------- #


class TestMissingEpsFields:

    def test_neither_eps_field_returns_empty_dataframe(self):
        data = {"facts": {"us-gaap": {"SomeOtherMetric": {}}}}
        df = _extract_eps_facts(data, cik=999, ticker="TEST")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        for col in _COLS:
            assert col in df.columns

    def test_empty_facts_returns_empty_dataframe(self):
        data = {"facts": {}}
        df = _extract_eps_facts(data, cik=999, ticker="TEST")
        assert len(df) == 0

    def test_completely_empty_json_returns_empty_dataframe(self):
        df = _extract_eps_facts({}, cik=999, ticker="TEST")
        assert len(df) == 0


# --------------------------------------------------------------------------- #
# derive_quarterly_eps                                                         #
# --------------------------------------------------------------------------- #


def _make_raw_panel(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal raw EDGAR panel DataFrame from a list of dicts."""
    records = []
    for r in rows:
        records.append({
            "ticker": r.get("ticker", "TEST"),
            "cik": r.get("cik", 999),
            "period_end": pd.Timestamp(r["period_end"]),
            "eps": float(r["eps"]),
            "filed": pd.Timestamp(r["filed"]),
            "form": r.get("form", "10-Q"),
            "fp": r["fp"],
            "accn": r.get("accn", ""),
        })
    return pd.DataFrame(records)


class TestDeriveQuarterlyEps:

    def test_complete_fiscal_year_derives_correct_q4(self):
        """Q1=$1.0, Q2=$1.2, Q3=$1.3, FY=$5.0 → derived Q4=$1.5."""
        df = _make_raw_panel([
            {"period_end": "2022-12-31", "eps": 1.0,  "filed": "2023-02-01", "fp": "Q1"},
            {"period_end": "2023-03-31", "eps": 1.2,  "filed": "2023-05-01", "fp": "Q2"},
            {"period_end": "2023-06-30", "eps": 1.3,  "filed": "2023-08-01", "fp": "Q3"},
            {"period_end": "2023-09-30", "eps": 5.0,  "filed": "2023-11-01", "fp": "FY",
             "form": "10-K"},
        ])
        result = derive_quarterly_eps(df)

        assert set(result["fp"]) == {"Q1", "Q2", "Q3", "Q4"}, (
            f"Expected fp values {{Q1,Q2,Q3,Q4}}, got {set(result['fp'])}"
        )
        q4 = result[result["fp"] == "Q4"]
        assert len(q4) == 1
        assert q4.iloc[0]["eps"] == pytest.approx(1.5)
        assert q4.iloc[0]["period_end"] == pd.Timestamp("2023-09-30")
        assert q4.iloc[0]["filed"] == pd.Timestamp("2023-11-01")
        assert q4.iloc[0]["form"] == "10-K"

    def test_no_fy_row_emits_quarterly_rows_only_no_q4(self):
        """If FY is absent, Q1/Q2/Q3 pass through and no Q4 is derived."""
        df = _make_raw_panel([
            {"period_end": "2022-12-31", "eps": 1.0, "filed": "2023-02-01", "fp": "Q1"},
            {"period_end": "2023-03-31", "eps": 1.2, "filed": "2023-05-01", "fp": "Q2"},
            {"period_end": "2023-06-30", "eps": 1.3, "filed": "2023-08-01", "fp": "Q3"},
        ])
        result = derive_quarterly_eps(df)

        assert len(result) == 3
        assert "Q4" not in result["fp"].values
        assert set(result["fp"]) == {"Q1", "Q2", "Q3"}

    def test_missing_one_quarter_skips_q4_derivation(self):
        """If Q2 is absent, FY is still dropped and Q4 is not derived."""
        df = _make_raw_panel([
            {"period_end": "2022-12-31", "eps": 1.0, "filed": "2023-02-01", "fp": "Q1"},
            # Q2 missing
            {"period_end": "2023-06-30", "eps": 1.3, "filed": "2023-08-01", "fp": "Q3"},
            {"period_end": "2023-09-30", "eps": 5.0, "filed": "2023-11-01", "fp": "FY",
             "form": "10-K"},
        ])
        result = derive_quarterly_eps(df)

        assert "Q4" not in result["fp"].values
        assert "FY" not in result["fp"].values
        assert len(result) == 2  # only Q1 and Q3

    def test_fy_rows_never_appear_in_output(self):
        """derive_quarterly_eps must always strip FY rows, even when Q4 can't be derived."""
        df = _make_raw_panel([
            {"period_end": "2023-09-30", "eps": 5.0, "filed": "2023-11-01", "fp": "FY",
             "form": "10-K"},
        ])
        result = derive_quarterly_eps(df)
        assert "FY" not in result["fp"].values

    def test_aapl_fy2023_derived_q4_in_expected_range(self, aapl_data):
        """AAPL FY2023 (period_end 2023-09-30): derived Q4 EPS should be ~$1.46."""
        raw = _extract_eps_facts(aapl_data, cik=AAPL_CIK, ticker="AAPL")
        quarterly = derive_quarterly_eps(raw)

        assert "FY" not in quarterly["fp"].values, "FY rows must be stripped"

        q4 = quarterly[
            (quarterly["ticker"] == "AAPL") &
            (quarterly["period_end"] == pd.Timestamp("2023-09-30")) &
            (quarterly["fp"] == "Q4")
        ]
        assert len(q4) == 1, (
            f"Expected exactly one derived Q4 row for AAPL FY2023, got {len(q4)}"
        )
        q4_eps = float(q4.iloc[0]["eps"])
        assert 1.45 <= q4_eps <= 1.50, (
            f"AAPL FY2023 derived Q4 EPS expected in [1.45, 1.50], got {q4_eps:.4f}"
        )
