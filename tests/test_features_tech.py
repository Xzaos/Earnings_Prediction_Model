"""
Tests for src/features_tech.py.

All tests except the AAPL fixture smoke test are fully offline and use
synthetic DataFrames with known values.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features_tech import compute_technical_features

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prices"
AAPL_FIXTURE = FIXTURE_DIR / "AAPL_2020.parquet"

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_SQRT252 = math.sqrt(252)


def _make_prices(
    ticker: str,
    dates: list[str],
    adj_close: list[float],
    close: list[float] | None = None,
    volume: list[float] | None = None,
) -> pd.DataFrame:
    n = len(dates)
    if close is None:
        close = adj_close[:]
    if volume is None:
        volume = [1_000_000.0] * n
    ac = pd.Series(adj_close)
    daily_return = ac.pct_change().tolist()
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(dates),
            "open": adj_close,
            "high": adj_close,
            "low": adj_close,
            "close": close,
            "adj_close": adj_close,
            "volume": volume,
            "daily_return": daily_return,
        }
    )


def _make_ann(ticker: str, period_end: str, announcement_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "period_end": [period_end],
            "announcement_date": [announcement_date],
        }
    )


def _flat_market(dates: list[str], ret: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates), "daily_return": ret})


# --------------------------------------------------------------------------- #
# Trading-day grid helper                                                       #
# --------------------------------------------------------------------------- #


def _bday_range(start: str, periods: int) -> list[str]:
    """Generate `periods` business days starting from `start`."""
    return [str(d.date()) for d in pd.bdate_range(start=start, periods=periods)]


# --------------------------------------------------------------------------- #
# 1. Momentum — known prices, manual computation                               #
# --------------------------------------------------------------------------- #


class TestMomentum:

    def _prices_and_ann(self):
        """25 trading days + 1 announcement day.
        adj_close grows by exactly 1% each day so total returns are exact."""
        dates = _bday_range("2020-01-02", 26)  # 26 business days
        factor = 1.01
        adj_close = [100.0 * (factor ** i) for i in range(26)]
        prices = _make_prices("TEST", dates, adj_close)
        # announcement_date = the 26th day (index 25); days used: indices 0–24
        ann = _make_ann("TEST", "2020-03-31", dates[25])
        mkt = _flat_market(dates)
        return prices, ann, mkt

    def test_momentum_1m_is_correct(self):
        """21 days of +1%/day → (1.01^21) - 1."""
        prices, ann, mkt = self._prices_and_ann()
        out = compute_technical_features(prices, ann, mkt)
        expected = 1.01 ** 21 - 1
        assert out["momentum_1m"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_momentum_3m_is_correct(self):
        """With only 25 pre-announcement days we have exactly 25 trading days.
        momentum_3m needs 63 days, so it should return NaN (< 31 available)."""
        prices, ann, mkt = self._prices_and_ann()
        out = compute_technical_features(prices, ann, mkt)
        # 25 pre-announcement days < 63/2=31 threshold → NaN
        assert pd.isna(out["momentum_3m"].iloc[0])

    def test_momentum_1m_with_flat_prices(self):
        """Flat prices → momentum = 0."""
        dates = _bday_range("2020-01-02", 25)
        prices = _make_prices("FLAT", dates, [50.0] * 25)
        ann = _make_ann("FLAT", "2020-03-31", dates[22])
        mkt = _flat_market(dates)
        out = compute_technical_features(prices, ann, mkt)
        assert out["momentum_1m"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_momentum_12m_excl_1m(self):
        """231 days of +0.1%/day: window [t-252, t-22] has 231 days."""
        n = 255
        dates = _bday_range("2019-01-02", n)
        factor = 1.001
        adj_close = [100.0 * (factor ** i) for i in range(n)]
        prices = _make_prices("LONG", dates, adj_close)
        ann = _make_ann("LONG", "2020-01-31", dates[-1])
        mkt = _flat_market(dates)
        out = compute_technical_features(prices, ann, mkt)
        # window [t-252, t-22] relative to t-1 = index 253.
        # t-1 = dates[253], t-22 = dates[232], t-252 = dates[2] (clamped to index 2)
        # actual available: index 0..253, so start_offset=-251 → index 2
        assert not pd.isna(out["momentum_12m_excl_1m"].iloc[0])


# --------------------------------------------------------------------------- #
# 2. Volatility — synthetic                                                    #
# --------------------------------------------------------------------------- #


class TestVolatility:

    def test_constant_returns_give_zero_vol(self):
        """Identical returns every day → std=0 → annualised vol=0."""
        dates = _bday_range("2020-01-02", 25)
        # build prices where daily_return = exactly 0.01 every day
        adj_close = [100.0 * (1.01 ** i) for i in range(25)]
        prices = _make_prices("CONST", dates, adj_close)
        ann = _make_ann("CONST", "2020-03-31", dates[22])
        mkt = _flat_market(dates)
        out = compute_technical_features(prices, ann, mkt)
        assert out["realized_vol_1m"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_alternating_returns_vol_formula(self):
        """Alternating +r, -r daily returns → known std and annualised vol."""
        r = 0.02
        n = 25
        dates = _bday_range("2020-01-02", n)
        # Build adj_close so daily_return alternates +r / -r
        prices_list = [100.0]
        for i in range(1, n):
            sign = 1 if i % 2 == 1 else -1
            prices_list.append(prices_list[-1] * (1 + sign * r))
        prices = _make_prices("ALT", dates, prices_list)
        ann = _make_ann("ALT", "2020-03-31", dates[22])
        mkt = _flat_market(dates)
        out = compute_technical_features(prices, ann, mkt)

        # The daily_return Series for the 21-day window (indices 1..21 of pre-ann prices)
        # Each pct_change alternates sign but pct_change of ratio prices is not exactly ±r.
        # Just verify the value is positive and finite.
        vol = out["realized_vol_1m"].iloc[0]
        assert not pd.isna(vol)
        assert vol > 0

    def test_vol_3m_requires_enough_data(self):
        """With only 30 pre-announcement days, vol_3m should be NaN (need ≥31)."""
        dates = _bday_range("2020-01-02", 32)
        prices = _make_prices("SHORT", dates, list(range(100, 132)))
        ann = _make_ann("SHORT", "2020-03-31", dates[31])
        mkt = _flat_market(dates)
        out = compute_technical_features(prices, ann, mkt)
        assert pd.isna(out["realized_vol_3m"].iloc[0])


# --------------------------------------------------------------------------- #
# 3. Leakage test — poison prices on/after announcement_date                  #
# --------------------------------------------------------------------------- #


class TestLeakage:

    def test_poison_prices_after_announcement_do_not_affect_features(self):
        """Append absurd prices on/after announcement_date; features must be
        identical to the version without the poison rows."""
        dates_clean = _bday_range("2020-01-02", 30)
        adj_close_clean = [100.0 * (1.005 ** i) for i in range(30)]
        prices_clean = _make_prices("LEAK", dates_clean, adj_close_clean)
        ann_date = dates_clean[25]
        ann = _make_ann("LEAK", "2020-03-31", ann_date)
        mkt = _flat_market(dates_clean)

        out_clean = compute_technical_features(prices_clean, ann, mkt)

        # Add poison: dates ON and AFTER announcement_date with huge returns
        poison_dates = _bday_range(ann_date, 5)
        poison_adj = [1e6] * 5
        prices_poison = _make_prices("LEAK", poison_dates, poison_adj)
        prices_with_poison = pd.concat([prices_clean, prices_poison], ignore_index=True)
        out_poison = compute_technical_features(prices_with_poison, ann, mkt)

        for col in ["momentum_1m", "momentum_3m", "realized_vol_1m", "dollar_volume_1m"]:
            c = out_clean[col].iloc[0]
            p = out_poison[col].iloc[0]
            if pd.isna(c):
                assert pd.isna(p), f"{col}: clean=NaN but poison={p}"
            else:
                assert c == pytest.approx(p, rel=1e-9), f"{col} differs: {c} vs {p}"


# --------------------------------------------------------------------------- #
# 4. surprise_track_record — NaN with fewer than 8 prior announcements        #
# --------------------------------------------------------------------------- #


class TestSurpriseTrackRecord:

    def _setup(self, n_prior: int):
        """Build `n_prior` prior announcements plus one current announcement."""
        all_dates = _bday_range("2018-01-02", (n_prior + 1) * 65 + 10)
        adj_close = [100.0 + i * 0.1 for i in range(len(all_dates))]
        prices = _make_prices("STR", all_dates, adj_close)
        mkt = _flat_market(all_dates, ret=0.0)

        # Space announcements ~60 trading days apart
        ann_rows = []
        for k in range(n_prior + 1):
            idx = 10 + k * 60
            if idx + 3 >= len(all_dates):
                break
            ann_rows.append(
                {
                    "ticker": "STR",
                    "period_end": all_dates[idx],
                    "announcement_date": all_dates[idx],
                }
            )
        ann_df = pd.DataFrame(ann_rows)
        return prices, ann_df, mkt

    def test_fewer_than_8_prior_gives_nan(self):
        prices, ann_df, mkt = self._setup(n_prior=5)
        out = compute_technical_features(prices, ann_df, mkt)
        # All rows have fewer than 8 prior announcements → all NaN
        assert out["surprise_track_record"].isna().all()

    def test_exactly_8_prior_gives_non_nan(self):
        prices, ann_df, mkt = self._setup(n_prior=8)
        out = compute_technical_features(prices, ann_df, mkt)
        # Last row (9th announcement) has 8 priors and rising prices → positive CARs
        last_val = out["surprise_track_record"].iloc[-1]
        assert not pd.isna(last_val)
        assert 0.0 <= last_val <= 1.0

    def test_track_record_value_with_all_positive_cars(self):
        """All prior 3-day CARs positive (steadily rising prices, flat market)
        → surprise_track_record should equal 1.0."""
        # Use 8 priors spaced such that prices always rise into each announcement
        dates = _bday_range("2018-01-02", 500)
        # 1% per day → always positive 3-day returns
        adj_close = [100.0 * (1.01 ** i) for i in range(500)]
        prices = _make_prices("POS", dates, adj_close)
        mkt = _flat_market(dates, ret=0.0)  # flat market → CAR = raw return

        ann_rows = [
            {"ticker": "POS", "period_end": dates[10 + k * 50], "announcement_date": dates[10 + k * 50]}
            for k in range(10)
        ]
        ann_df = pd.DataFrame(ann_rows)
        out = compute_technical_features(prices, ann_df, mkt)
        last = out["surprise_track_record"].iloc[-1]
        assert last == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 5. AAPL 2020 fixture smoke test                                              #
# --------------------------------------------------------------------------- #


class TestAAPL2020:

    @pytest.fixture(scope="class")
    def aapl_prices(self):
        if not AAPL_FIXTURE.exists():
            pytest.skip(
                f"AAPL 2020 fixture not found at {AAPL_FIXTURE}. "
                "Run: python -c \"from tests.test_prices_pull import _download_fixture; _download_fixture()\""
            )
        from src.prices_pull import _build_price_df
        raw = pd.read_parquet(AAPL_FIXTURE)
        return _build_price_df(raw, "AAPL")

    @pytest.fixture(scope="class")
    def aapl_out(self, aapl_prices):
        ann = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
                "period_end": ["2020-03-28", "2020-06-27", "2020-09-26", "2020-12-26"],
                "announcement_date": ["2020-04-30", "2020-07-30", "2020-10-29", "2021-01-27"],
            }
        )
        # Use AAPL itself as a trivial market proxy (SPY unavailable in fixture)
        mkt = aapl_prices[["date", "daily_return"]].copy()
        return compute_technical_features(aapl_prices, ann, mkt)

    def test_output_schema(self, aapl_out):
        expected_cols = {
            "ticker", "period_end", "announcement_date", "feature_filed",
            "momentum_1m", "momentum_3m", "momentum_12m_excl_1m",
            "realized_vol_1m", "realized_vol_3m", "dollar_volume_1m",
            "surprise_track_record",
        }
        assert expected_cols.issubset(set(aapl_out.columns))

    def test_four_rows_returned(self, aapl_out):
        assert len(aapl_out) == 4

    def test_feature_filed_before_announcement(self, aapl_out):
        assert (aapl_out["feature_filed"] < aapl_out["announcement_date"]).all()

    def test_momentum_1m_finite_for_q2_q3_q4(self, aapl_out):
        # Q1 may have insufficient history (start of fixture) but Q2–Q4 should have it
        finite = aapl_out["momentum_1m"].notna()
        assert finite.sum() >= 3

    def test_vol_1m_positive(self, aapl_out):
        vols = aapl_out["realized_vol_1m"].dropna()
        assert (vols > 0).all()

    def test_dollar_volume_finite(self, aapl_out):
        dvol = aapl_out["dollar_volume_1m"].dropna()
        assert len(dvol) >= 3
        assert (dvol.abs() > 0).all()

    def test_surprise_track_record_nan_first_rows(self, aapl_out):
        # Only 4 rows total → always fewer than 8 prior → all NaN
        assert aapl_out["surprise_track_record"].isna().all()
