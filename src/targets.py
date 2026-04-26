"""
Target variable construction for the earnings prediction engine.

Three target formulations are supported:

1. `time_series_sue` — Foster-Olsen-Shevlin (1984) standardized unexpected
   earnings, computed from a firm's own past EPS history. This is the primary
   target. No analyst data required.

2. `proxy_analyst_sue` — A surprise-relative-to-consensus measure that uses
   yfinance's shallow consensus history. Secondary, sanity-check target.

3. `three_day_car` — Cumulative abnormal return in a [-1, +1] window around the
   announcement, with abnormal return defined as stock return minus market
   return (CRSP-VW substitute via SPY/^GSPC). This sidesteps the estimate-quality
   problem entirely — the market's reaction is the label.

All functions are pure (no I/O). Inputs are pandas objects; outputs are pandas
objects with the same index. Real data fetching lives in `edgar_pull.py` and
`prices_pull.py`.

Conventions
-----------
- All EPS series are quarterly, indexed by the *fiscal period end date*.
- All "announcement date" references mean the SEC filing date (`filed` field
  in EDGAR companyfacts), NOT the period end date.
- Returns are simple returns (P_t / P_{t-1} - 1), not log returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Foster-Olsen-Shevlin time-series SUE                                     #
# --------------------------------------------------------------------------- #


@dataclass
class SRWDFit:
    """Result of fitting a seasonal random walk with drift to one firm's EPS.

    The model is:
        EPS_t = EPS_{t-4} + delta + epsilon_t

    where `delta` is the drift (estimated as the mean of the first differences
    of the year-over-year change), and `sigma` is the std of residuals.
    """

    drift: float
    sigma: float
    n_obs: int  # number of (EPS_t, EPS_{t-4}) pairs used in the fit

    @property
    def is_valid(self) -> bool:
        """A fit is valid only if we have enough history and a meaningfully
        positive sigma. We reject near-zero sigmas (below 1e-8) because they
        produce explosive SUE values from floating-point noise — a perfectly
        deterministic EPS series would otherwise yield SUE = (residual)/(1e-16)
        which is meaningless."""
        return (
            self.n_obs >= MIN_HISTORY_QUARTERS
            and np.isfinite(self.sigma)
            and self.sigma > 1e-8
        )


# Minimum number of past quarters needed before we'll publish a SUE value.
# FOS 1984 used at least 5 years (20 quarters); we relax to 8 because we want
# to use names that IPO'd or first appeared in our universe more recently. This
# is a tunable knob — bump it back up to 20 for a closer FOS replication.
MIN_HISTORY_QUARTERS = 8

# Cap on absolute SUE values. The denominator (residual std) can be very small
# for firms with stable, predictable EPS, producing extreme tail values that
# wreck downstream models. We winsorize at +/- 8, which is conservative.
SUE_WINSORIZE = 8.0


def fit_srwd(eps_history: pd.Series) -> SRWDFit:
    """Fit a seasonal random walk with drift to a single firm's quarterly EPS history.

    Parameters
    ----------
    eps_history : pd.Series
        Quarterly EPS values, indexed by fiscal period end date, sorted ascending.
        Must contain *only* historical data — i.e., observations strictly before
        the quarter we are about to predict. The caller is responsible for
        slicing this correctly; this function does not look at the index for
        leakage protection.

    Returns
    -------
    SRWDFit
        Drift, residual sigma, number of usable observations. Check `.is_valid`
        before using.
    """
    if not isinstance(eps_history, pd.Series):
        raise TypeError("eps_history must be a pandas Series")

    eps = eps_history.dropna().sort_index()
    if len(eps) < MIN_HISTORY_QUARTERS:
        return SRWDFit(drift=np.nan, sigma=np.nan, n_obs=len(eps))

    # Year-over-year change at lag 4.
    yoy_change = eps - eps.shift(4)
    yoy_change = yoy_change.dropna()
    if len(yoy_change) < 2:
        return SRWDFit(drift=np.nan, sigma=np.nan, n_obs=len(yoy_change))

    drift = float(yoy_change.mean())
    # Residuals from the SRWD model are yoy_change - drift; their std is sigma.
    residuals = yoy_change - drift
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else np.nan

    return SRWDFit(drift=drift, sigma=sigma, n_obs=int(len(yoy_change)))


def time_series_sue(
    eps_history: pd.Series,
    actual_eps: float,
    period_end: pd.Timestamp,
) -> Optional[float]:
    """Compute Foster-Olsen-Shevlin SUE for a single (firm, quarter) observation.

    Parameters
    ----------
    eps_history : pd.Series
        The firm's quarterly EPS history, indexed by fiscal period end. **Must
        only contain quarters whose announcement (filing) date is strictly
        before the announcement date of the quarter being predicted.** Caller
        is responsible for that slicing — this function does not enforce it
        because it doesn't have access to filing dates here.
    actual_eps : float
        The realized EPS for the quarter being predicted.
    period_end : pd.Timestamp
        The fiscal period end date for the quarter being predicted. Used to
        find EPS_{t-4} in the history.

    Returns
    -------
    float or None
        The SUE value, winsorized at +/- SUE_WINSORIZE. None if there isn't
        enough history or the SRWD fit is invalid.

    Notes
    -----
    The expected EPS under SRWD is `EPS_{t-4} + drift`. SUE is then
    `(actual - expected) / sigma`. If we can't find an EPS value approximately
    4 quarters back (within a 30-day tolerance to handle fiscal-year shifts),
    we return None rather than guessing.
    """
    fit = fit_srwd(eps_history)
    if not fit.is_valid:
        return None

    # Find EPS approximately 4 quarters before period_end.
    target_lag = period_end - pd.DateOffset(months=12)
    tolerance = pd.Timedelta(days=30)
    candidate_dates = eps_history.index[
        (eps_history.index >= target_lag - tolerance)
        & (eps_history.index <= target_lag + tolerance)
    ]
    if len(candidate_dates) == 0:
        return None
    # If multiple, take the one closest to the target.
    closest_date = min(candidate_dates, key=lambda d: abs(d - target_lag))
    eps_lag4 = eps_history.loc[closest_date]
    if pd.isna(eps_lag4):
        return None

    expected_eps = eps_lag4 + fit.drift
    sue_raw = (actual_eps - expected_eps) / fit.sigma
    return float(np.clip(sue_raw, -SUE_WINSORIZE, SUE_WINSORIZE))


def time_series_sue_panel(
    eps_panel: pd.DataFrame,
    announcement_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Compute time-series SUE for every (firm, quarter) in a panel.

    This is the production-style entry point that walks the panel firm by firm,
    quarter by quarter, building a strictly-historical EPS series for each
    target observation. **This is where the leakage discipline gets enforced.**

    Parameters
    ----------
    eps_panel : pd.DataFrame
        Long-format EPS panel with columns:
          - `ticker` (str)
          - `period_end` (datetime64[ns]): fiscal period end date
          - `eps` (float): reported EPS for that period
          - `filed` (datetime64[ns]): the SEC filing date when this EPS first
            became public. **Required**; this is what makes leakage protection
            possible.
    announcement_dates : pd.DataFrame
        Same index keys as eps_panel, columns:
          - `ticker`
          - `period_end`
          - `announcement_date`: the date we are computing SUE *for*. In
            practice this is the same as `filed` for that (ticker, period_end).

    Returns
    -------
    pd.DataFrame
        Columns: ticker, period_end, announcement_date, sue.
        Rows where SUE could not be computed (insufficient history, no lag-4
        observation, etc.) are still returned, with sue = NaN, so the panel
        shape is preserved for downstream joins.
    """
    required_cols = {"ticker", "period_end", "eps", "filed"}
    missing = required_cols - set(eps_panel.columns)
    if missing:
        raise ValueError(f"eps_panel missing columns: {missing}")

    eps_panel = eps_panel.copy()
    eps_panel["period_end"] = pd.to_datetime(eps_panel["period_end"])
    eps_panel["filed"] = pd.to_datetime(eps_panel["filed"])

    announcement_dates = announcement_dates.copy()
    announcement_dates["period_end"] = pd.to_datetime(announcement_dates["period_end"])
    announcement_dates["announcement_date"] = pd.to_datetime(
        announcement_dates["announcement_date"]
    )

    out_rows = []
    for (ticker, period_end, announcement_date) in announcement_dates[
        ["ticker", "period_end", "announcement_date"]
    ].itertuples(index=False, name=None):
        firm_eps = eps_panel[eps_panel["ticker"] == ticker]
        # Keep only EPS observations that were FILED strictly before the
        # announcement we're predicting. This is the leakage guard.
        firm_eps_known = firm_eps[firm_eps["filed"] < announcement_date]
        # And only quarters strictly before the one we're predicting.
        firm_eps_known = firm_eps_known[firm_eps_known["period_end"] < period_end]

        history_series = (
            firm_eps_known.set_index("period_end")["eps"].sort_index()
        )

        # Find the actual EPS for the quarter being predicted (this is the
        # label, so it's allowed — we're computing what the surprise WAS, after
        # the fact, for training purposes).
        actual_row = firm_eps[firm_eps["period_end"] == period_end]
        if len(actual_row) == 0:
            sue = np.nan
        else:
            actual_eps = float(actual_row["eps"].iloc[0])
            result = time_series_sue(history_series, actual_eps, period_end)
            sue = result if result is not None else np.nan

        out_rows.append(
            {
                "ticker": ticker,
                "period_end": period_end,
                "announcement_date": announcement_date,
                "sue": sue,
            }
        )

    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# 2. Proxy analyst-consensus SUE                                              #
# --------------------------------------------------------------------------- #


def proxy_analyst_sue(
    actual_eps: float,
    consensus_eps: float,
    floor: float = 0.01,
) -> Optional[float]:
    """Surprise-relative-to-consensus, scaled by abs(consensus).

    Parameters
    ----------
    actual_eps : float
    consensus_eps : float
        The consensus EPS estimate that was active immediately before the
        announcement. Caller must enforce the timestamp discipline; this
        function takes consensus at face value.
    floor : float
        Minimum value used in the denominator to avoid division-by-near-zero
        explosions when consensus is very close to zero. The denominator is
        `max(|consensus|, floor)`.

    Returns
    -------
    float or None
        The proxy SUE, or None if either input is NaN.
    """
    if pd.isna(actual_eps) or pd.isna(consensus_eps):
        return None
    denom = max(abs(consensus_eps), floor)
    return float((actual_eps - consensus_eps) / denom)


# --------------------------------------------------------------------------- #
# 3. Three-day cumulative abnormal return                                     #
# --------------------------------------------------------------------------- #


def three_day_car(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    announcement_date: pd.Timestamp,
) -> Optional[float]:
    """Compound the [-1, 0, +1] trading-day excess return around the announcement.

    Parameters
    ----------
    stock_returns : pd.Series
        Daily simple returns for the stock, indexed by trading date.
    market_returns : pd.Series
        Daily simple returns for the market benchmark (e.g., SPY or ^GSPC),
        indexed by trading date.
    announcement_date : pd.Timestamp
        The earnings announcement date. If the announcement happened
        after-market-close, the [-1, 0, +1] window straddles the close
        correctly because day 0 is the announcement day itself and day +1 is
        the following trading day where the news is priced in.

    Returns
    -------
    float or None
        The 3-day CAR, or None if any of the three days is missing returns
        for either series.

    Notes
    -----
    A more rigorous abnormal-return measure would use a market model
    (alpha + beta * R_m) estimated over a pre-announcement window. For v1 we
    use the simpler "stock minus market" definition; upgrade is straightforward
    later.
    """
    # Find the announcement day in the stock_returns index, or the next trading
    # day if announcement_date itself isn't a trading day.
    trading_days = stock_returns.index
    on_or_after = trading_days[trading_days >= announcement_date]
    if len(on_or_after) == 0:
        return None
    day_0 = on_or_after[0]

    # Find day -1 and day +1 by integer offset in the trading-days index.
    day_0_idx = trading_days.get_loc(day_0)
    if day_0_idx == 0 or day_0_idx >= len(trading_days) - 1:
        # Not enough flanking days.
        return None
    day_minus1 = trading_days[day_0_idx - 1]
    day_plus1 = trading_days[day_0_idx + 1]

    window_days = [day_minus1, day_0, day_plus1]

    try:
        stock_window = stock_returns.loc[window_days]
        market_window = market_returns.loc[window_days]
    except KeyError:
        return None

    if stock_window.isna().any() or market_window.isna().any():
        return None

    abnormal = stock_window - market_window
    car = float((1.0 + abnormal).prod() - 1.0)
    return car
