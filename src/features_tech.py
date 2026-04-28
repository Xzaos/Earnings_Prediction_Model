"""
Price and volume feature engineering for earnings prediction.

All features are computed point-in-time: for each (ticker, period_end) row the
announcement_date is the as-of cutoff.  Only prices strictly before
announcement_date enter any calculation.  The `feature_filed` output column is
set to the last trading day before announcement_date (i.e., the latest date on
which the price data could have been observed).

Features
--------
momentum_1m          : total return over [t-21, t-1] trading days before announcement
momentum_3m          : total return over [t-63, t-1]
momentum_12m_excl_1m : total return over [t-252, t-22]  (excludes the most recent month)
realized_vol_1m      : annualised std of daily_return over the 21 days ending t-1
realized_vol_3m      : annualised std of daily_return over the 63 days ending t-1
dollar_volume_1m     : log(mean(close × volume)) over the 21 days ending t-1
surprise_track_record: fraction of the 8 most recent prior announcements whose
                       3-day CAR (days 0, +1, +2 relative to announcement) was > 0.
                       NaN when fewer than 8 prior rows exist or price coverage is
                       insufficient for any of the 8 CARs.

Inputs
------
prices_panel       : long-format DataFrame from pull_prices_for_tickers.
                     Required columns: ticker, date (datetime64), adj_close,
                     close, volume, daily_return.
announcement_dates : DataFrame with columns: ticker, period_end, announcement_date.
market_returns     : DataFrame with columns: date (datetime64), daily_return
                     (market benchmark, e.g. SPY).  Used to compute CARs.

Output
------
DataFrame with columns: ticker, period_end, announcement_date, feature_filed,
momentum_1m, momentum_3m, momentum_12m_excl_1m, realized_vol_1m,
realized_vol_3m, dollar_volume_1m, surprise_track_record.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SQRT252 = math.sqrt(252)

_OUTPUT_COLS = [
    "ticker",
    "period_end",
    "announcement_date",
    "feature_filed",
    "momentum_1m",
    "momentum_3m",
    "momentum_12m_excl_1m",
    "realized_vol_1m",
    "realized_vol_3m",
    "dollar_volume_1m",
    "surprise_track_record",
]


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _sorted_trading_dates(prices_ticker: pd.DataFrame) -> pd.Series:
    """Sorted unique trading dates for one ticker as a pd.Series of Timestamps."""
    return pd.Series(sorted(prices_ticker["date"].unique()))


def _offset_date(trading_dates: pd.Series, ref_date: pd.Timestamp, offset: int) -> pd.Timestamp | None:
    """Return the trading date at `offset` positions relative to the last date
    strictly before `ref_date`.  offset=0 → last pre-announcement day (t-1);
    offset=-20 → 21 days back from t-1; etc.  Returns None if out of range."""
    idx_arr = trading_dates[trading_dates < ref_date]
    if idx_arr.empty:
        return None
    base_pos = len(idx_arr) - 1  # position of t-1 in trading_dates
    target_pos = base_pos + offset
    if target_pos < 0 or target_pos >= len(trading_dates):
        return None
    return trading_dates.iloc[target_pos]


def _window_returns(
    prices_ticker: pd.DataFrame,
    trading_dates: pd.Series,
    announcement_date: pd.Timestamp,
    lookback: int,
) -> pd.Series | None:
    """Return the `daily_return` series for the `lookback` trading days ending
    at t-1 (inclusive).  Returns None if insufficient data."""
    end_date = _offset_date(trading_dates, announcement_date, 0)
    start_date = _offset_date(trading_dates, announcement_date, -(lookback - 1))
    if end_date is None or start_date is None:
        return None
    mask = (prices_ticker["date"] >= start_date) & (prices_ticker["date"] <= end_date)
    sub = prices_ticker.loc[mask, "daily_return"].dropna()
    if len(sub) < lookback // 2:  # require at least half the window
        return None
    return sub


def _total_return(daily_returns: pd.Series) -> float:
    """Compound return from a series of daily returns."""
    return float(np.prod(1.0 + daily_returns) - 1.0)


def _compute_momentum(
    prices_ticker: pd.DataFrame,
    trading_dates: pd.Series,
    announcement_date: pd.Timestamp,
    start_offset: int,
    end_offset: int,
) -> float:
    """Total return from start_offset to end_offset (both relative to t-1, inclusive).
    start_offset <= end_offset <= 0."""
    end_date = _offset_date(trading_dates, announcement_date, end_offset)
    start_date = _offset_date(trading_dates, announcement_date, start_offset)
    if end_date is None or start_date is None:
        return float("nan")
    mask = (prices_ticker["date"] >= start_date) & (prices_ticker["date"] <= end_date)
    returns = prices_ticker.loc[mask, "daily_return"].dropna()
    n_expected = end_offset - start_offset + 1
    if len(returns) < n_expected // 2:
        return float("nan")
    return _total_return(returns)


def _compute_vol(
    prices_ticker: pd.DataFrame,
    trading_dates: pd.Series,
    announcement_date: pd.Timestamp,
    lookback: int,
) -> float:
    """Annualised realised volatility over `lookback` trading days ending t-1."""
    rets = _window_returns(prices_ticker, trading_dates, announcement_date, lookback)
    if rets is None or len(rets) < 2:
        return float("nan")
    return float(rets.std(ddof=1) * _SQRT252)


def _compute_dollar_volume(
    prices_ticker: pd.DataFrame,
    trading_dates: pd.Series,
    announcement_date: pd.Timestamp,
    lookback: int,
) -> float:
    """log(mean(close × volume)) over `lookback` trading days ending t-1."""
    end_date = _offset_date(trading_dates, announcement_date, 0)
    start_date = _offset_date(trading_dates, announcement_date, -(lookback - 1))
    if end_date is None or start_date is None:
        return float("nan")
    mask = (prices_ticker["date"] >= start_date) & (prices_ticker["date"] <= end_date)
    sub = prices_ticker.loc[mask].copy()
    if sub.empty:
        return float("nan")
    dv = (sub["close"] * sub["volume"]).replace(0, float("nan"))
    mean_dv = dv.mean()
    if pd.isna(mean_dv) or mean_dv <= 0:
        return float("nan")
    return float(math.log(mean_dv))


def _compute_car_3day(
    prices_ticker: pd.DataFrame,
    market_map: dict[pd.Timestamp, float],
    announcement_date: pd.Timestamp,
) -> float | None:
    """3-day CAR (days 0, +1, +2 relative to announcement_date).
    Returns None if price coverage is insufficient."""
    mask = prices_ticker["date"] >= announcement_date
    sub = prices_ticker.loc[mask].nsmallest(3, "date")
    if len(sub) < 3:
        return None
    days = sub.sort_values("date")
    car = 0.0
    for _, row in days.iterrows():
        mkt = market_map.get(row["date"])
        if mkt is None or pd.isna(mkt):
            return None
        car += float(row["daily_return"]) - float(mkt)
    return car


def _compute_surprise_track_record(
    prior_rows: pd.DataFrame,
    prices_ticker: pd.DataFrame,
    market_map: dict[pd.Timestamp, float],
) -> float:
    """Fraction of 8 prior announcements with positive 3-day CAR.
    NaN if fewer than 8 prior rows or any CAR is uncomputable."""
    if len(prior_rows) < 8:
        return float("nan")
    last8 = prior_rows.iloc[-8:]
    cars = []
    for _, row in last8.iterrows():
        ann = pd.Timestamp(row["announcement_date"])
        car = _compute_car_3day(prices_ticker, market_map, ann)
        if car is None:
            return float("nan")
        cars.append(car)
    return float(np.mean([c > 0 for c in cars]))


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def compute_technical_features(
    prices_panel: pd.DataFrame,
    announcement_dates: pd.DataFrame,
    market_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute price/volume features for every (ticker, period_end) row.

    Parameters
    ----------
    prices_panel : pd.DataFrame
        Long-format daily prices.  Columns: ticker, date, close, volume,
        adj_close, daily_return.
    announcement_dates : pd.DataFrame
        Columns: ticker, period_end, announcement_date.
    market_returns : pd.DataFrame
        Columns: date, daily_return (benchmark, e.g. SPY).

    Returns
    -------
    pd.DataFrame
        Columns: ticker, period_end, announcement_date, feature_filed,
        momentum_1m, momentum_3m, momentum_12m_excl_1m,
        realized_vol_1m, realized_vol_3m, dollar_volume_1m,
        surprise_track_record.
    """
    prices_panel = prices_panel.copy()
    prices_panel["date"] = pd.to_datetime(prices_panel["date"])

    announcement_dates = announcement_dates.copy()
    announcement_dates["announcement_date"] = pd.to_datetime(announcement_dates["announcement_date"])
    announcement_dates["period_end"] = pd.to_datetime(announcement_dates["period_end"])
    announcement_dates = announcement_dates.sort_values(["ticker", "announcement_date"]).reset_index(drop=True)

    market_returns = market_returns.copy()
    market_returns["date"] = pd.to_datetime(market_returns["date"])
    market_map: dict[pd.Timestamp, float] = dict(
        zip(market_returns["date"], market_returns["daily_return"])
    )

    records: list[dict] = []

    for ticker, ticker_anns in announcement_dates.groupby("ticker", sort=False):
        prices_t = prices_panel[prices_panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        if prices_t.empty:
            logger.warning("No prices for ticker %s — skipping", ticker)
            for _, ann_row in ticker_anns.iterrows():
                records.append(_nan_record(ticker, ann_row))
            continue

        trading_dates = _sorted_trading_dates(prices_t)
        ticker_anns_sorted = ticker_anns.sort_values("announcement_date").reset_index(drop=True)

        for i, ann_row in ticker_anns_sorted.iterrows():
            ann_date = ann_row["announcement_date"]
            period_end = ann_row["period_end"]

            # feature_filed = last trading day before announcement_date
            feature_filed = _offset_date(trading_dates, ann_date, 0)

            # momentum features
            mom_1m = _compute_momentum(prices_t, trading_dates, ann_date, -20, 0)
            mom_3m = _compute_momentum(prices_t, trading_dates, ann_date, -62, 0)
            mom_12m = _compute_momentum(prices_t, trading_dates, ann_date, -251, -21)

            # vol features
            vol_1m = _compute_vol(prices_t, trading_dates, ann_date, 21)
            vol_3m = _compute_vol(prices_t, trading_dates, ann_date, 63)

            # dollar volume
            dvol = _compute_dollar_volume(prices_t, trading_dates, ann_date, 21)

            # surprise track record: prior announcements strictly before this one
            prior = ticker_anns_sorted[ticker_anns_sorted["announcement_date"] < ann_date]
            str_val = _compute_surprise_track_record(prior, prices_t, market_map)

            records.append(
                {
                    "ticker": ticker,
                    "period_end": period_end,
                    "announcement_date": ann_date,
                    "feature_filed": feature_filed,
                    "momentum_1m": mom_1m,
                    "momentum_3m": mom_3m,
                    "momentum_12m_excl_1m": mom_12m,
                    "realized_vol_1m": vol_1m,
                    "realized_vol_3m": vol_3m,
                    "dollar_volume_1m": dvol,
                    "surprise_track_record": str_val,
                }
            )

    if not records:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    out = pd.DataFrame(records)
    out["feature_filed"] = pd.to_datetime(out["feature_filed"])
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["announcement_date"] = pd.to_datetime(out["announcement_date"])
    return out[_OUTPUT_COLS].reset_index(drop=True)


def _nan_record(ticker: str, ann_row: pd.Series) -> dict:
    return {
        "ticker": ticker,
        "period_end": ann_row["period_end"],
        "announcement_date": ann_row["announcement_date"],
        "feature_filed": pd.NaT,
        "momentum_1m": float("nan"),
        "momentum_3m": float("nan"),
        "momentum_12m_excl_1m": float("nan"),
        "realized_vol_1m": float("nan"),
        "realized_vol_3m": float("nan"),
        "dollar_volume_1m": float("nan"),
        "surprise_track_record": float("nan"),
    }
