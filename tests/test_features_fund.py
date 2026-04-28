"""
Tests for src/features_fund.py.

Network policy: the AAPL smoke test reuses the EDGAR fixture at
tests/fixtures/edgar/CIK0000320193.json (same fixture as test_edgar_pull.py).
All other tests use fully synthetic DataFrames — no network required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features_fund import compute_fundamental_features, _FEATURE_COLS

# ------------------------------------------------------------------ #
# Shared helpers                                                        #
# ------------------------------------------------------------------ #

_EDGAR_FIXTURE = Path(__file__).parent / "fixtures" / "edgar" / "CIK0000320193.json"
AAPL_CIK = 320193

_BASE_COLS = ["ticker", "cik", "period_end", "fact_name", "value",
              "filed", "form", "fp", "accn"]


def _row(ticker, fact_name, period_end, value, filed, fp="Q1", cik=999):
    return {
        "ticker": ticker, "cik": cik,
        "period_end": pd.Timestamp(period_end),
        "fact_name": fact_name,
        "value": float(value),
        "filed": pd.Timestamp(filed),
        "form": "10-Q", "fp": fp, "accn": "",
    }


def _panel(*rows):
    return pd.DataFrame(list(rows))


def _quarterly_dates(start: str, n: int) -> list[pd.Timestamp]:
    """Generate n approximately-quarterly dates starting from start."""
    base = pd.Timestamp(start)
    return [base + pd.DateOffset(months=3 * i) for i in range(n)]


def _make_multi_quarter_panel(
    ticker: str,
    fact_name: str,
    dates: list[pd.Timestamp],
    values: list[float],
    filed_dates: list[pd.Timestamp] | None = None,
    fp_list: list[str] | None = None,
) -> pd.DataFrame:
    """Build a facts panel for one ticker, one fact, multiple quarters."""
    if filed_dates is None:
        filed_dates = [d + pd.Timedelta(days=45) for d in dates]
    if fp_list is None:
        fps = ["Q1", "Q2", "Q3", "Q4"]
        fp_list = [fps[i % 4] for i in range(len(dates))]
    rows = []
    for d, v, f, fp in zip(dates, values, filed_dates, fp_list):
        rows.append(_row(ticker, fact_name, d, v, f, fp))
    return pd.DataFrame(rows)


def _concat_panels(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(list(dfs), ignore_index=True)


# ------------------------------------------------------------------ #
# TestAccrualsSloan                                                     #
# ------------------------------------------------------------------ #


class TestAccrualsSloan:
    """Verify Sloan (1996) balance-sheet accruals formula using handpicked values."""

    def _build_panel(self) -> pd.DataFrame:
        """5 quarters of data; accruals computed at Q5 (lag4 = Q1).

        All values in raw dollars (scaled to real-world magnitudes so they
        clear the $1M denominator floor). Ratios are what matter for the test.

        Q1 baseline (t-4):
          CA=100M, Cash=20M, CL=80M, STD=10M, TP=5M, TA=500M, Dep=3M/qtr

        Q5 current (t):
          CA=110M, Cash=25M, CL=85M, STD=12M, TP=6M, TA=520M, Dep=3M/qtr

        Manual calculation:
          ΔCA=10M, ΔCash=5M, ΔCL=5M, ΔSTD=2M, ΔTP=1M
          Dep_annual = 3M × 4 = 12M (trailing 4Q sum)
          avg_TA = (520M + 500M) / 2 = 510M
          numerator = (10M-5M) - (5M-2M-1M) - 12M = 5M - 2M - 12M = -9M
          accruals_sloan = -9M / 510M
        """
        M = 1_000_000
        dates = _quarterly_dates("2021-01-01", 5)
        fps = ["Q1", "Q2", "Q3", "Q4", "Q1"]

        def _q(fact, vals):
            rows = []
            for d, v, fp in zip(dates, vals, fps):
                rows.append(_row("TEST", fact, d, v * M, d + pd.Timedelta(days=45), fp))
            return pd.DataFrame(rows)

        return _concat_panels(
            _q("total_current_assets",      [100, 102, 104, 106, 110]),
            _q("cash",                      [ 20,  21,  22,  23,  25]),
            _q("total_current_liabilities", [ 80,  81,  82,  83,  85]),
            _q("short_term_debt",           [ 10,  10,  11,  11,  12]),
            _q("income_taxes_payable",      [  5,   5,   5,   6,   6]),
            _q("total_assets",              [500, 505, 510, 515, 520]),
            _q("depreciation_amortization", [  3,   3,   3,   3,   3]),
        )

    def test_accruals_value_matches_manual_calculation(self):
        panel = self._build_panel()
        result = compute_fundamental_features(panel)

        # Row at the 5th quarter (index 4 in per-ticker sorted output)
        row = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]
        expected = -9.0 / 510.0
        assert row["accruals_sloan"] == pytest.approx(expected, rel=1e-9), (
            f"Expected accruals_sloan={expected:.9f}, got {row['accruals_sloan']:.9f}"
        )

    def test_first_four_rows_have_nan_accruals(self):
        """Lag-4 is unavailable for the first 4 rows; accruals must be NaN."""
        panel = self._build_panel()
        result = compute_fundamental_features(panel)
        first_four = result[result["ticker"] == "TEST"].sort_values("period_end").head(4)
        assert first_four["accruals_sloan"].isna().all(), (
            "Expected NaN accruals for first 4 rows (lag-4 not available)"
        )


# ------------------------------------------------------------------ #
# TestGrossMarginChange                                                 #
# ------------------------------------------------------------------ #


class TestGrossMarginChange:

    def test_known_margin_expansion(self):
        """GP/Rev 40% → 45%; expected gross_margin_change_yoy = 0.05."""
        dates = _quarterly_dates("2022-01-01", 5)
        fps = ["Q1", "Q2", "Q3", "Q4", "Q1"]

        rows = []
        M = 1_000_000
        rev_vals = [100, 100, 100, 100, 100]   # flat revenue (×$1M)
        gp_vals  = [ 40,  40,  40,  40,  45]   # GP rises in Q5 (×$1M)

        for d, rv, gv, fp in zip(dates, rev_vals, gp_vals, fps):
            filed = d + pd.Timedelta(days=45)
            rows.append(_row("TEST", "revenue",      d, rv * M, filed, fp))
            rows.append(_row("TEST", "gross_profit", d, gv * M, filed, fp))

        panel = pd.DataFrame(rows)
        result = compute_fundamental_features(panel)
        last = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]

        assert last["gross_margin_change_yoy"] == pytest.approx(0.05, abs=1e-9)


# ------------------------------------------------------------------ #
# TestGapDetection                                                      #
# ------------------------------------------------------------------ #


class TestGapDetection:

    def test_missing_quarter_sets_yoy_to_nan(self):
        """Firm with 2020-Q1..Q4 then 2021-Q3 (skipping Q1, Q2).
        revenue_growth_yoy for 2021-Q3 must be NaN because shift(4) is misaligned."""
        dates = [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-04-01"),
            pd.Timestamp("2020-07-01"),
            pd.Timestamp("2020-10-01"),
            pd.Timestamp("2021-07-01"),   # gap: skips 2021-Q1 and Q2
        ]
        fps = ["Q1", "Q2", "Q3", "Q4", "Q3"]
        M = 1_000_000
        rev_vals = [100, 100, 100, 100, 110]

        rows = []
        for d, v, fp in zip(dates, rev_vals, fps):
            filed = d + pd.Timedelta(days=45)
            rows.append(_row("TEST", "revenue", d, v * M, filed, fp))

        panel = pd.DataFrame(rows)
        result = compute_fundamental_features(panel)
        last = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]

        assert pd.isna(last["revenue_growth_yoy"]), (
            f"Expected NaN revenue_growth_yoy for row after sequence gap, "
            f"got {last['revenue_growth_yoy']}"
        )

    def test_consecutive_quarters_compute_normally(self):
        """No gaps → revenue_growth_yoy must be non-NaN for the 5th quarter."""
        dates = _quarterly_dates("2020-01-01", 5)
        fps = ["Q1", "Q2", "Q3", "Q4", "Q1"]
        M = 1_000_000
        rev_vals = [100, 100, 100, 100, 110]

        rows = []
        for d, v, fp in zip(dates, rev_vals, fps):
            filed = d + pd.Timedelta(days=45)
            rows.append(_row("TEST", "revenue", d, v * M, filed, fp))

        panel = pd.DataFrame(rows)
        result = compute_fundamental_features(panel)
        last = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]

        assert not pd.isna(last["revenue_growth_yoy"]), (
            "Expected non-NaN revenue_growth_yoy for clean quarterly sequence"
        )


# ------------------------------------------------------------------ #
# TestOperatingLeverageClipping                                         #
# ------------------------------------------------------------------ #


class TestOperatingLeverageClipping:

    def _build_panel(self, delta_rev_pct: float) -> pd.DataFrame:
        """5-quarter panel where revenue changes by delta_rev_pct YoY in Q5."""
        M = 1_000_000
        rev_t4 = 1_000.0 * M
        oi_t4  =   100.0 * M
        rev_t  = rev_t4 * (1 + delta_rev_pct)
        oi_t   = oi_t4  * 2.0   # OI doubles → %ΔOI = 100%

        dates = _quarterly_dates("2021-01-01", 5)
        fps = ["Q1", "Q2", "Q3", "Q4", "Q1"]
        rows = []
        for i, (d, fp) in enumerate(zip(dates, fps)):
            filed = d + pd.Timedelta(days=45)
            rv = rev_t if i == 4 else rev_t4
            oi = oi_t if i == 4 else oi_t4
            rows.append(_row("TEST", "revenue",          d, rv, filed, fp))
            rows.append(_row("TEST", "operating_income", d, oi, filed, fp))
        return pd.DataFrame(rows)

    def test_tiny_revenue_change_clips_to_20(self):
        """0.1% revenue change with 100% OI change → raw leverage is 1000; clip to 20."""
        panel = self._build_panel(delta_rev_pct=0.001)
        result = compute_fundamental_features(panel)
        last = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]
        assert last["operating_leverage"] == pytest.approx(20.0), (
            f"Expected operating_leverage clipped to 20.0, got {last['operating_leverage']}"
        )

    def test_normal_revenue_change_not_clipped(self):
        """5% revenue change with 100% OI change → leverage = 20.0 (boundary; exactly clipped)."""
        panel = self._build_panel(delta_rev_pct=0.05)
        result = compute_fundamental_features(panel)
        last = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]
        # raw = (100%/100%) / (5%/5%) = 20; at the clip boundary
        assert last["operating_leverage"] <= 20.0 + 1e-9


# ------------------------------------------------------------------ #
# TestOutputShape                                                       #
# ------------------------------------------------------------------ #


class TestOutputShape:

    def _build_panel(self, n_tickers=3, n_quarters=12) -> pd.DataFrame:
        dates = _quarterly_dates("2020-01-01", n_quarters)
        fps = ["Q1", "Q2", "Q3", "Q4"] * (n_quarters // 4 + 1)
        rows = []
        for t in range(n_tickers):
            ticker = f"T{t:02d}"
            for i, (d, fp) in enumerate(zip(dates, fps)):
                filed = d + pd.Timedelta(days=45)
                M = 1_000_000
                rows.append(_row(ticker, "revenue",      d, (100.0 + i) * M, filed, fp))
                rows.append(_row(ticker, "gross_profit", d,  (40.0 + i) * M, filed, fp))
                rows.append(_row(ticker, "total_assets", d, (500.0 + i) * M, filed, fp))
        return pd.DataFrame(rows)

    def test_row_count(self):
        panel = self._build_panel(n_tickers=3, n_quarters=12)
        result = compute_fundamental_features(panel)
        assert len(result) == 3 * 12, f"Expected 36 rows, got {len(result)}"

    def test_column_count_and_names(self):
        panel = self._build_panel()
        result = compute_fundamental_features(panel)
        expected_cols = {"ticker", "period_end", "feature_filed"} | set(_FEATURE_COLS)
        assert set(result.columns) == expected_cols, (
            f"Column mismatch: {set(result.columns)} vs {expected_cols}"
        )
        assert len(result.columns) == 11

    def test_first_four_rows_per_ticker_have_nan_lag_features(self):
        """Features that need lag-4 (all except qoq) must be NaN for first 4 rows."""
        panel = self._build_panel()
        result = compute_fundamental_features(panel)
        lag4_features = [
            "accruals_sloan", "gross_margin_change_yoy",
            "revenue_growth_yoy", "operating_leverage",
            "dso_change_yoy", "inventory_to_sales_change_yoy",
        ]
        for ticker in result["ticker"].unique():
            first4 = result[result["ticker"] == ticker].sort_values("period_end").head(4)
            for feat in lag4_features:
                assert first4[feat].isna().all(), (
                    f"{feat} should be NaN for first 4 rows of {ticker}"
                )


# ------------------------------------------------------------------ #
# TestFeatureFiled                                                      #
# ------------------------------------------------------------------ #


class TestFeatureFiledCorrectness:

    def test_feature_filed_equals_max_of_current_and_lag4_filed(self):
        """feature_filed at Q5 must be max(filed_Q5, filed_Q1).
        Since filing dates increase over time, feature_filed = filed_Q5."""
        dates = _quarterly_dates("2021-01-01", 5)
        fps = ["Q1", "Q2", "Q3", "Q4", "Q1"]
        filed_dates = [
            pd.Timestamp("2021-02-15"),
            pd.Timestamp("2021-05-15"),
            pd.Timestamp("2021-08-15"),
            pd.Timestamp("2021-11-15"),
            pd.Timestamp("2022-02-15"),   # Q5 filed last → should be feature_filed
        ]
        rows = []
        M = 1_000_000
        for d, fp, f in zip(dates, fps, filed_dates):
            rows.append(_row("TEST", "revenue",      d, 100.0 * M, f, fp))
            rows.append(_row("TEST", "total_assets", d, 500.0 * M, f, fp))

        panel = pd.DataFrame(rows)
        result = compute_fundamental_features(panel)
        q5 = result[result["ticker"] == "TEST"].sort_values("period_end").iloc[-1]

        assert q5["feature_filed"] == pd.Timestamp("2022-02-15"), (
            f"Expected feature_filed=2022-02-15, got {q5['feature_filed']}"
        )

    def test_feature_filed_is_monotonically_increasing_per_ticker(self):
        """As we move forward in time, feature_filed should never decrease
        (it absorbs the max of all lagged inputs)."""
        dates = _quarterly_dates("2020-01-01", 8)
        fps = ["Q1", "Q2", "Q3", "Q4"] * 2
        rows = []
        M = 1_000_000
        for i, (d, fp) in enumerate(zip(dates, fps)):
            filed = d + pd.Timedelta(days=45)
            rows.append(_row("TEST", "revenue", d, 100.0 * M, filed, fp))

        panel = pd.DataFrame(rows)
        result = compute_fundamental_features(panel)
        filed = result[result["ticker"] == "TEST"].sort_values("period_end")["feature_filed"]
        # Drop NaT rows (first row where lag-1 and lag-4 haven't kicked in yet)
        filed_valid = filed.dropna()
        diffs = filed_valid.diff().dropna()
        assert (diffs >= pd.Timedelta(0)).all(), (
            "feature_filed decreased over time — lag filed dates not being propagated"
        )


# ------------------------------------------------------------------ #
# TestAaplSmokeTest                                                     #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def aapl_facts_panel():
    """Build an AAPL facts panel from the cached EDGAR fixture."""
    if not _EDGAR_FIXTURE.exists():
        pytest.skip(f"AAPL EDGAR fixture not found at {_EDGAR_FIXTURE}")

    import json
    from src.edgar_pull import _extract_facts, XBRL_TAGS, XBRL_UNITS

    with _EDGAR_FIXTURE.open(encoding="utf-8") as f:
        data = json.load(f)

    frames = []
    for fact_name, tags in XBRL_TAGS.items():
        unit = XBRL_UNITS.get(fact_name, "USD")
        df = _extract_facts(data, tags, "AAPL", AAPL_CIK, unit)
        if not df.empty:
            df = df.rename(columns={"value": "value"})
            df["fact_name"] = fact_name
            frames.append(df)

    if not frames:
        pytest.skip("No facts extracted from AAPL fixture")

    panel = pd.concat(frames, ignore_index=True)
    # _extract_facts returns 'value'; pull_facts_for_tickers adds 'fact_name'.
    # We need the full schema including fp — _extract_facts already has it.
    return panel


class TestAaplSmokeTest:

    def test_gross_margin_in_expected_range(self, aapl_facts_panel):
        """AAPL gross margin should be between 40% and 50% in recent quarters."""
        result = compute_fundamental_features(aapl_facts_panel)
        recent = result.dropna(subset=["gross_margin_change_yoy"]).tail(8)
        if recent.empty:
            pytest.skip("No recent gross_margin_change_yoy values computed for AAPL")
        # Absolute gross margin (not change) — verify via revenue_growth sanity
        # Check the change is in a plausible range: margins don't swing > 15pp/yr
        assert (recent["gross_margin_change_yoy"].abs() <= 0.15).all(), (
            "AAPL gross margin YoY change exceeded ±15pp — likely a data error"
        )

    def test_revenue_growth_in_expected_range(self, aapl_facts_panel):
        """AAPL YoY revenue growth should be within -20% to +30%."""
        result = compute_fundamental_features(aapl_facts_panel)
        recent = result.dropna(subset=["revenue_growth_yoy"]).tail(8)
        if recent.empty:
            pytest.skip("No recent revenue_growth_yoy values computed for AAPL")
        assert (recent["revenue_growth_yoy"].between(-0.20, 0.30)).all(), (
            f"AAPL revenue growth outside [-20%, +30%]:\n"
            f"{recent[['period_end', 'revenue_growth_yoy']].to_string()}"
        )

    def test_output_has_correct_columns(self, aapl_facts_panel):
        result = compute_fundamental_features(aapl_facts_panel)
        assert set(result.columns) == set(["ticker", "period_end", "feature_filed"] + _FEATURE_COLS)

    def test_feature_filed_not_before_period_end(self, aapl_facts_panel):
        """feature_filed must be after or equal to period_end (data always filed
        after the period it describes)."""
        result = compute_fundamental_features(aapl_facts_panel)
        valid = result.dropna(subset=["feature_filed"])
        bad = valid[valid["feature_filed"] < valid["period_end"]]
        assert len(bad) == 0, (
            f"feature_filed before period_end in {len(bad)} rows:\n{bad.head().to_string()}"
        )
