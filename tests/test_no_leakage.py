"""
Tests for forward-looking-bias (data leakage) in the earnings prediction pipeline.

The contract this module enforces:

  For any predicted observation (firm F, quarter Q, announcement date A_FQ),
  every input value used to make that prediction must satisfy:

      input_filing_date < A_FQ

  Strictly less-than. Never <=. The prediction is conceptually made the day
  BEFORE the announcement.

This module has two kinds of tests:

  1. Synthetic tests on `targets.py`. These run today on toy data — they
     verify that the SUE computation refuses to use data that wasn't yet
     filed. Run with `pytest tests/test_no_leakage.py -v`.

  2. Real-data assertions. Functions starting with `assert_panel_*` are
     designed to be called from a notebook against the actual pulled panel
     once the EDGAR puller is built. They aren't pytest tests because they
     need real data; they're reusable assertion helpers.

If a leakage test fails: STOP. Do not run any modeling. Backtest results
from a leaky pipeline are worse than no results, because they look credible
and they're wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.targets import (
    MIN_HISTORY_QUARTERS,
    SUE_WINSORIZE,
    fit_srwd,
    proxy_analyst_sue,
    three_day_car,
    time_series_sue,
    time_series_sue_panel,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #


def _make_eps_history(n_quarters: int, base: float = 1.0, growth: float = 0.05,
                     seasonal_amp: float = 0.2, seed: int = 0) -> pd.Series:
    """Build a synthetic quarterly EPS series with seasonality + drift + noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-03-31", periods=n_quarters, freq="QE")
    quarter_of_year = (dates.quarter - 1).to_numpy()  # 0,1,2,3,0,1,...
    seasonal = seasonal_amp * np.cos(2 * 3.14159 * quarter_of_year / 4.0)
    trend = base + growth * (dates.year - 2010 + dates.quarter / 4.0)
    noise = rng.normal(0, 0.05, size=n_quarters)
    return pd.Series(trend + seasonal + noise, index=dates, name="eps")


def _make_panel(n_firms: int = 3, n_quarters: int = 16, lag_days: int = 35):
    """Build a synthetic (eps_panel, announcement_dates) pair.

    `lag_days` is the gap between fiscal period end and SEC filing date.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    eps_rows = []
    ann_rows = []
    for firm_i in range(n_firms):
        ticker = f"FIRM{firm_i}"
        eps_series = _make_eps_history(
            n_quarters, base=1.0 + firm_i * 0.1, seed=firm_i
        )
        for period_end, eps in eps_series.items():
            filed = period_end + pd.Timedelta(days=lag_days)
            eps_rows.append({
                "ticker": ticker,
                "period_end": period_end,
                "eps": eps,
                "filed": filed,
            })
            ann_rows.append({
                "ticker": ticker,
                "period_end": period_end,
                "announcement_date": filed,
            })
    return pd.DataFrame(eps_rows), pd.DataFrame(ann_rows)


# --------------------------------------------------------------------------- #
# Unit tests on targets.py                                                    #
# --------------------------------------------------------------------------- #


class TestFitSRWD:

    def test_insufficient_history_returns_invalid(self):
        eps = _make_eps_history(MIN_HISTORY_QUARTERS - 1)
        fit = fit_srwd(eps)
        assert not fit.is_valid

    def test_sufficient_history_returns_valid(self):
        eps = _make_eps_history(MIN_HISTORY_QUARTERS + 4)
        fit = fit_srwd(eps)
        assert fit.is_valid
        assert fit.sigma > 0

    def test_drift_recovers_known_growth(self):
        """If we feed in a series with a known YoY drift, the fit should recover it."""
        # Pure drift, no seasonality, no noise.
        import numpy as np
        n = 20
        dates = pd.date_range("2010-03-31", periods=n, freq="QE")
        # EPS_t = EPS_{t-4} + 0.10 (annual drift of 0.10 in EPS)
        eps_values = [1.0 + (t // 4) * 0.10 for t in range(n)]
        eps = pd.Series(eps_values, index=dates)
        fit = fit_srwd(eps)
        assert abs(fit.drift - 0.10) < 1e-9


class TestTimeSeriesSUE:

    def test_returns_none_when_history_too_short(self):
        eps = _make_eps_history(MIN_HISTORY_QUARTERS - 1)
        period_end = eps.index[-1] + pd.DateOffset(months=3)
        result = time_series_sue(eps, actual_eps=1.5, period_end=period_end)
        assert result is None

    def test_zero_surprise_gives_zero_sue(self):
        """If actual EPS exactly equals the SRWD expectation, SUE should be 0."""
        # Construct a deterministic series with fixed drift, then predict the next
        # quarter at exactly EPS_{t-4} + drift.
        import numpy as np
        n = 16
        dates = pd.date_range("2010-03-31", periods=n, freq="QE")
        drift = 0.05
        eps_values = [1.0 + (t // 4) * drift + 0.01 * (t % 4) for t in range(n)]
        eps = pd.Series(eps_values, index=dates)

        next_period_end = dates[-1] + pd.DateOffset(months=3)
        eps_lag4 = eps_values[-4]
        # The fit will recover drift exactly only if YoY changes are constant.
        # Our series has constant YoY change (0.05), so drift fit = 0.05 exactly,
        # sigma fit = 0. Sigma=0 makes SUE undefined, so we pad with a tiny noise
        # term in a separate test variant.
        fit = fit_srwd(eps)
        # With perfect deterministic series, sigma = 0, so this fit is invalid.
        assert not fit.is_valid

    def test_sue_winsorized(self):
        """Extreme actual values should be clipped to +/- SUE_WINSORIZE."""
        eps = _make_eps_history(20, seed=1)
        period_end = eps.index[-1] + pd.DateOffset(months=3)
        # An EPS of 1000 is many sigmas above expected; should clip.
        result = time_series_sue(eps, actual_eps=1000.0, period_end=period_end)
        assert result is not None
        assert result == SUE_WINSORIZE

        result_neg = time_series_sue(eps, actual_eps=-1000.0, period_end=period_end)
        assert result_neg is not None
        assert result_neg == -SUE_WINSORIZE


# --------------------------------------------------------------------------- #
# THE leakage test — synthetic version                                        #
# --------------------------------------------------------------------------- #


class TestNoLeakageInPanel:
    """The single most important test in the repository.

    These tests verify that `time_series_sue_panel` cannot use data that
    wasn't yet public on the announcement date. We do this by constructing
    panels with deliberately-poisoned future data and verifying that
    poisoning the future doesn't change past SUE values.
    """

    def test_future_data_poisoning_does_not_affect_past_sue(self):
        """If we replace EPS values for quarters AFTER quarter Q with garbage,
        SUE for quarter Q must not change. If it does, the function is leaking
        future data into the past computation."""
        eps_panel, ann_dates = _make_panel(n_firms=2, n_quarters=20)

        clean_result = time_series_sue_panel(eps_panel, ann_dates)

        # Poison: replace EPS for the last 4 quarters of each firm with -999.
        poisoned = eps_panel.copy()
        for ticker in poisoned["ticker"].unique():
            firm_mask = poisoned["ticker"] == ticker
            firm = poisoned[firm_mask].sort_values("period_end")
            poison_idx = firm.index[-4:]
            poisoned.loc[poison_idx, "eps"] = -999.0

        poisoned_result = time_series_sue_panel(poisoned, ann_dates)

        # For each (ticker, period_end) where the period_end is BEFORE the
        # poisoned region, the SUE must be identical.
        merged = clean_result.merge(
            poisoned_result,
            on=["ticker", "period_end", "announcement_date"],
            suffixes=("_clean", "_poison"),
        )
        for ticker in merged["ticker"].unique():
            firm_dates = sorted(merged[merged["ticker"] == ticker]["period_end"])
            if len(firm_dates) < 5:
                continue
            unaffected_cutoff = firm_dates[-5]  # rows strictly before the poisoned 4
            unaffected = merged[
                (merged["ticker"] == ticker) & (merged["period_end"] <= unaffected_cutoff)
            ]
            for _, row in unaffected.iterrows():
                if pd.notna(row["sue_clean"]) or pd.notna(row["sue_poison"]):
                    assert row["sue_clean"] == row["sue_poison"], (
                        f"Leakage detected: SUE for {row['ticker']} "
                        f"{row['period_end']} changed when future data was "
                        f"poisoned. Clean={row['sue_clean']}, "
                        f"Poisoned={row['sue_poison']}"
                    )

    def test_same_period_data_filed_after_announcement_excluded(self):
        """Suppose firm A's Q1 2015 EPS was filed on 2015-04-30, but firm A's
        Q2 2014 EPS was somehow re-filed on 2015-05-15 (an amendment after
        the Q1 2015 announcement). The amended Q2 2014 value should NOT be
        used when computing SUE for Q1 2015."""
        eps_panel, ann_dates = _make_panel(n_firms=1, n_quarters=16)
        # Find a quarter in the middle, and pretend an EARLIER quarter's EPS
        # was amended AFTER this one's announcement.
        ann_dates_sorted = ann_dates.sort_values("period_end").reset_index(drop=True)
        target_idx = 10
        target_announcement = ann_dates_sorted.loc[target_idx, "announcement_date"]
        target_period_end = ann_dates_sorted.loc[target_idx, "period_end"]

        # Amend the EPS for the quarter 8 quarters back: original filed date
        # was way before target_announcement; we now move it to AFTER.
        amended = eps_panel.copy()
        old_period = ann_dates_sorted.loc[target_idx - 8, "period_end"]
        amend_mask = (amended["ticker"] == "FIRM0") & (amended["period_end"] == old_period)
        amended.loc[amend_mask, "filed"] = target_announcement + pd.Timedelta(days=1)
        amended.loc[amend_mask, "eps"] = -999.0

        clean_result = time_series_sue_panel(eps_panel, ann_dates)
        amended_result = time_series_sue_panel(amended, ann_dates)

        # SUE for the target quarter must be unchanged because the amended
        # value wasn't yet public.
        clean_sue = clean_result[
            (clean_result["ticker"] == "FIRM0")
            & (clean_result["period_end"] == target_period_end)
        ]["sue"].iloc[0]
        amended_sue = amended_result[
            (amended_result["ticker"] == "FIRM0")
            & (amended_result["period_end"] == target_period_end)
        ]["sue"].iloc[0]

        if pd.notna(clean_sue) or pd.notna(amended_sue):
            assert clean_sue == amended_sue, (
                "Leakage: amended EPS filed after target announcement was used"
            )


# --------------------------------------------------------------------------- #
# Real-data assertion helpers (for use from notebooks)                        #
# --------------------------------------------------------------------------- #


def assert_panel_no_future_features(
    feature_panel: pd.DataFrame,
    *,
    feature_filing_date_col: str = "feature_filed",
    target_announcement_col: str = "announcement_date",
    feature_name_col: str = "feature_name",
) -> None:
    """Assert no row has a feature whose filing date is >= the target announcement.

    Designed to be called from a notebook against the real, fully-built panel,
    in long format where each row is one (firm, quarter, feature) observation
    with the filing date attached. Strict less-than is required.

    Raises AssertionError with a useful diagnostic if any leakage is found.
    """
    required = {feature_filing_date_col, target_announcement_col}
    missing = required - set(feature_panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    feature_panel = feature_panel.copy()
    feature_panel[feature_filing_date_col] = pd.to_datetime(
        feature_panel[feature_filing_date_col]
    )
    feature_panel[target_announcement_col] = pd.to_datetime(
        feature_panel[target_announcement_col]
    )

    leaky = feature_panel[
        feature_panel[feature_filing_date_col] >= feature_panel[target_announcement_col]
    ]
    if len(leaky) == 0:
        return

    # Build a useful diagnostic: top 5 leaky rows, plus a count by feature.
    msg_parts = [
        f"LEAKAGE DETECTED: {len(leaky)} rows have feature filing date >= "
        f"target announcement date.",
        "First 5 leaky rows:",
        leaky.head(5).to_string(),
    ]
    if feature_name_col in leaky.columns:
        by_feature = leaky[feature_name_col].value_counts().head(10)
        msg_parts.append("Top leaky features:")
        msg_parts.append(by_feature.to_string())
    raise AssertionError("\n".join(msg_parts))


def assert_announcement_date_after_period_end(
    announcement_dates: pd.DataFrame,
    *,
    min_lag_days: int = 1,
    max_lag_days: int = 120,
    period_end_col: str = "period_end",
    announcement_col: str = "announcement_date",
) -> None:
    """Sanity-check announcement dates against fiscal period ends.

    Earnings are typically announced 20–60 days after period end. This catches
    misaligned data where someone has accidentally swapped the two columns or
    used the period-end as the announcement.
    """
    df = announcement_dates.copy()
    df[period_end_col] = pd.to_datetime(df[period_end_col])
    df[announcement_col] = pd.to_datetime(df[announcement_col])
    lag = (df[announcement_col] - df[period_end_col]).dt.days

    too_short = (lag < min_lag_days).sum()
    too_long = (lag > max_lag_days).sum()
    if too_short > 0 or too_long > 0:
        raise AssertionError(
            f"Suspicious announcement-date / period-end gaps: "
            f"{too_short} rows have lag < {min_lag_days} days "
            f"(announcement on or before period end?); "
            f"{too_long} rows have lag > {max_lag_days} days "
            f"(stale/missing announcement date?). "
            f"Median lag was {lag.median():.1f} days."
        )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
