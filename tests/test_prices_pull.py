"""
Tests for src/prices_pull.py.

Network policy
--------------
The AAPL 2020 fixture (tests/fixtures/prices/AAPL_2020.parquet) must exist
before the fixture-dependent tests run.  All other tests are fully offline and
use either the fixture or synthetic DataFrames.

To download the fixture once:
    python -c "from tests.test_prices_pull import _download_fixture; _download_fixture()"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prices_pull import (
    _build_price_df,
    _download_ticker_raw,
    pull_prices_for_tickers,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prices"
AAPL_FIXTURE = FIXTURE_DIR / "AAPL_2020.parquet"


# --------------------------------------------------------------------------- #
# Fixture download helper (run once; not a pytest test)                       #
# --------------------------------------------------------------------------- #


def _download_fixture() -> None:
    """Fetch AAPL 2020 price data from yfinance and save as a test fixture.
    Requires network.  Run once before the test suite."""
    import yfinance as yf

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    raw = yf.Ticker("AAPL").history(
        start="2020-01-01", end="2021-01-01",
        auto_adjust=False, actions=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty data for AAPL 2020")
    raw.to_parquet(AAPL_FIXTURE)
    print(f"Saved {AAPL_FIXTURE} ({AAPL_FIXTURE.stat().st_size / 1024:.0f} KB, {len(raw)} rows)")


# --------------------------------------------------------------------------- #
# Shared fixture                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def aapl_2020_raw() -> pd.DataFrame:
    """Load the AAPL 2020 raw Ticker.history() DataFrame from the parquet fixture."""
    if not AAPL_FIXTURE.exists():
        pytest.skip(
            f"AAPL 2020 fixture not found at {AAPL_FIXTURE}. "
            "Run: python -c \"from tests.test_prices_pull import _download_fixture; _download_fixture()\""
        )
    return pd.read_parquet(AAPL_FIXTURE)


@pytest.fixture(scope="module")
def aapl_2020_panel(aapl_2020_raw) -> pd.DataFrame:
    """Processed panel for AAPL 2020 (output of _build_price_df)."""
    return _build_price_df(aapl_2020_raw, "AAPL")


# --------------------------------------------------------------------------- #
# Column / schema tests                                                        #
# --------------------------------------------------------------------------- #


class TestPanelSchema:

    def test_required_columns_present(self, aapl_2020_panel):
        expected = {"ticker", "date", "open", "high", "low",
                    "close", "adj_close", "volume", "daily_return"}
        assert expected.issubset(set(aapl_2020_panel.columns))

    def test_date_column_is_datetime(self, aapl_2020_panel):
        assert pd.api.types.is_datetime64_any_dtype(aapl_2020_panel["date"])

    def test_date_column_is_timezone_naive(self, aapl_2020_panel):
        assert aapl_2020_panel["date"].dt.tz is None

    def test_ticker_column_is_aapl(self, aapl_2020_panel):
        assert (aapl_2020_panel["ticker"] == "AAPL").all()

    def test_full_year_row_count(self, aapl_2020_panel):
        # 2020 had 253 trading days
        assert 245 <= len(aapl_2020_panel) <= 260


# --------------------------------------------------------------------------- #
# daily_return correctness                                                     #
# --------------------------------------------------------------------------- #


class TestDailyReturn:

    def test_first_row_is_nan(self, aapl_2020_panel):
        first_return = aapl_2020_panel.sort_values("date").iloc[0]["daily_return"]
        assert pd.isna(first_return), f"Expected NaN for first row, got {first_return}"

    def test_return_formula_matches_adj_close(self, aapl_2020_panel):
        """Pick three consecutive rows and verify return = (p_t / p_{t-1}) - 1."""
        df = aapl_2020_panel.sort_values("date").reset_index(drop=True)
        for i in [1, 10, 50]:
            expected = df.loc[i, "adj_close"] / df.loc[i - 1, "adj_close"] - 1
            actual = df.loc[i, "daily_return"]
            assert actual == pytest.approx(expected, rel=1e-6), (
                f"Row {i}: expected return {expected:.6f}, got {actual:.6f}"
            )

    def test_returns_are_finite_after_first_row(self, aapl_2020_panel):
        df = aapl_2020_panel.sort_values("date").iloc[1:]
        assert df["daily_return"].notna().all()
        assert (df["daily_return"].abs() < 1.0).all(), "Suspiciously large daily return (>100%)"


# --------------------------------------------------------------------------- #
# adj_close used for returns (not raw close)                                  #
# --------------------------------------------------------------------------- #


class TestAdjCloseUsedForReturns:
    """
    Verifies that daily_return is computed from adj_close, not raw close.

    In yfinance 1.3.0, Ticker.history(auto_adjust=False) returns a Close that
    is already split-adjusted — the pre-split raw price is unavailable.  The
    meaningful distinction between Close and Adj Close is dividend ex-dates:
    on an ex-date, Close drops by the dividend amount (a spurious negative
    return), while Adj Close was retrospectively lowered for all prior days so
    the ex-date daily_return is ~0%.  Using adj_close is therefore correct.
    """

    def test_dividend_exdate_produces_zero_return_with_adj_close(self):
        """On a dividend ex-date the Close return is spuriously negative, but
        the Adj Close return correctly absorbs the dividend and reads ~0%.

        yfinance 1.3.0 note: Close is already split-adjusted (pre-split raw
        prices are unavailable); the real distinction between Close and
        Adj Close is therefore dividend ex-dates, not splits.

        Synthetic scenario (mirrors real yfinance behaviour):
          Day 0 (before ex-date): Close=100, Adj Close=99  <- backward-adjusted
          Day 1 (ex-date):        Close=99,  Adj Close=99  <- price drops by $1 div
          Return from Close:     99/100 - 1 = -1%  (wrong: treats div as a loss)
          Return from Adj Close: 99/99  - 1 =  0%  (correct: total return = 0%)
        """
        raw = pd.DataFrame(
            {
                "Open":      [100.0, 99.0],
                "High":      [101.0, 100.0],
                "Low":       [99.0,  98.0],
                "Close":     [100.0, 99.0],   # drops by $1 dividend on ex-date
                "Adj Close": [99.0,  99.0],   # day 0 backward-adjusted; no step on ex-date
                "Volume":    [1_000_000, 2_000_000],
            },
            index=pd.to_datetime(["2020-05-07", "2020-05-08"]),
        )
        raw.index.name = "Date"

        df = _build_price_df(raw, "TEST")
        exdate_return = df.sort_values("date").iloc[1]["daily_return"]

        assert exdate_return == pytest.approx(0.0, abs=1e-9), (
            f"Expected 0% return on dividend ex-date using adj_close, "
            f"got {exdate_return:.4f} — likely using raw close instead"
        )

    def test_raw_close_would_give_wrong_return_on_exdate(self):
        """Negative control: raw Close pct_change on the ex-date gives -1%,
        confirming the synthetic data distinguishes the two columns."""
        raw_close = pd.Series([100.0, 99.0])
        wrong_return = raw_close.pct_change().iloc[1]
        assert wrong_return == pytest.approx(-0.01, abs=1e-9)


# --------------------------------------------------------------------------- #
# Retry logic                                                                  #
# --------------------------------------------------------------------------- #


class TestRetryLogic:

    def _make_raw_df(self) -> pd.DataFrame:
        """Minimal raw DataFrame shaped like Ticker.history() output."""
        idx = pd.to_datetime(["2020-01-02", "2020-01-03"]).tz_localize("America/New_York")
        idx.name = "Date"
        return pd.DataFrame(
            {
                "Open":      [300.0, 301.0],
                "High":      [305.0, 306.0],
                "Low":       [299.0, 300.0],
                "Close":     [302.0, 303.0],
                "Adj Close": [302.0, 303.0],
                "Volume":    [10_000, 11_000],
            },
            index=idx,
        )

    def test_retries_on_transient_failure_and_returns_data(self):
        """Mock _download_ticker_raw to fail twice, succeed on third attempt."""
        good_df = self._make_raw_df()
        call_count = {"n": 0}

        def flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient network error")
            return good_df

        with patch("src.prices_pull._download_ticker_raw", side_effect=flaky), \
             patch("src.prices_pull.time.sleep"):   # skip actual sleep in tests
            panel, failed = pull_prices_for_tickers(
                ["FAKE"], start="2020-01-01", end="2020-01-05", force_refresh=True
            )

        assert call_count["n"] == 3, f"Expected 3 attempts, got {call_count['n']}"
        assert "FAKE" not in failed
        assert len(panel) == 2

    def test_all_retries_exhausted_adds_to_failed_list(self):
        """Mock _download_ticker_raw to always raise; ticker must land in failed."""
        with patch("src.prices_pull._download_ticker_raw",
                   side_effect=ConnectionError("always fails")), \
             patch("src.prices_pull.time.sleep"):
            panel, failed = pull_prices_for_tickers(
                ["FAIL"], start="2020-01-01", end="2020-01-05", force_refresh=True
            )

        assert "FAIL" in failed
        assert len(panel) == 0

    def test_one_failure_does_not_abort_other_tickers(self):
        """Two tickers: first always fails, second succeeds. Panel must have
        the second ticker's rows and only the first in failed_tickers."""
        good_df = self._make_raw_df()
        call_count = {"n": 0}

        def selective_fail(ticker, *args, **kwargs):
            if ticker == "BAD":
                raise ConnectionError("bad ticker")
            return good_df

        with patch("src.prices_pull._download_ticker_raw", side_effect=selective_fail), \
             patch("src.prices_pull.time.sleep"):
            panel, failed = pull_prices_for_tickers(
                ["BAD", "GOOD"], start="2020-01-01", end="2020-01-05", force_refresh=True
            )

        assert "BAD" in failed
        assert "GOOD" not in failed
        assert set(panel["ticker"].unique()) == {"GOOD"}
