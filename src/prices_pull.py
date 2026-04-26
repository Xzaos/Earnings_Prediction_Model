"""
Daily price data puller using yfinance.

Fetches OHLCV + adjusted-close for every ticker in the universe, plus a
market benchmark (SPY by default), caching each ticker to a per-ticker
Parquet file.  Returns a long-format panel with a pre-computed daily_return
column based on adj_close.

Cache policy
------------
Each ticker's data lives at data/raw/prices/{ticker}.parquet.  If the file
exists and is less than 7 days old, the network call is skipped.  Pass
force_refresh=True to bypass the cache.

Retry policy
------------
yfinance is flaky.  Every download is attempted up to 3 times with
exponential backoff (1 s, 2 s, 4 s).  Tickers that fail all retries are
collected in a `failed_tickers` list returned alongside the panel; they do
NOT crash the run.

Returns convention
------------------
daily_return is computed as (adj_close_t / adj_close_{t-1}) - 1 using
adj_close so that dividend ex-dates do not inject spurious negative returns.
The first row per ticker is NaN.

yfinance 1.3.0 column behaviour (auto_adjust=False)
----------------------------------------------------
In yfinance >= 1.x, Ticker.history(auto_adjust=False) returns:
  - Close     : split-adjusted but NOT dividend-adjusted.
  - Adj Close : split-adjusted AND dividend-adjusted (backward-looking
                cumulative factor from all dividends through the fetch date).

Concretely, the Close column already reflects historical splits — there is no
"raw" pre-split price in the returned data.  The distinction between Close and
Adj Close is therefore purely dividend-driven: on quarterly dividend ex-dates
the adj_close return absorbs the dividend yield (~0.2pp for AAPL), while
the Close return shows an equal-sized spurious price drop.  Using adj_close
for daily_return is the correct choice.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

_REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = _REPO_ROOT / "data" / "raw" / "prices"
CACHE_MAX_AGE_DAYS = 7

_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


# --------------------------------------------------------------------------- #
# Cache helpers                                                                #
# --------------------------------------------------------------------------- #


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker}.parquet"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return age.days < CACHE_MAX_AGE_DAYS


# --------------------------------------------------------------------------- #
# Download helpers (isolated for mocking in tests)                            #
# --------------------------------------------------------------------------- #


def _download_ticker_raw(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Single attempt — no retry. Returns raw Ticker.history() DataFrame."""
    return yf.Ticker(ticker).history(
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
    )


def _fetch_with_retry(ticker: str, start: str, end: str, max_attempts: int = 3) -> pd.DataFrame:
    """Download with exponential backoff. Raises on exhausted retries."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(max_attempts):
        try:
            df = _download_ticker_raw(ticker, start, end)
            if df is None or df.empty:
                raise ValueError(f"yfinance returned empty data for {ticker}")
            return df
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)   # 1 s, 2 s, (4 s …)
    raise last_exc


# --------------------------------------------------------------------------- #
# Data transformation                                                          #
# --------------------------------------------------------------------------- #


def _build_price_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalise a raw Ticker.history() frame into the canonical output format.

    - Strips timezone from the DatetimeIndex and keeps date-only precision.
    - Renames columns to snake_case.
    - Adds daily_return = (adj_close / adj_close.shift(1)) - 1.
    """
    df = raw[list(_RENAME.keys())].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.rename(columns=_RENAME)
    df.insert(0, "ticker", ticker)
    df = df.reset_index()   # date becomes a column

    df = df.sort_values("date").reset_index(drop=True)
    df["daily_return"] = df["adj_close"].pct_change()   # NaN for first row

    return df


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def pull_prices_for_tickers(
    tickers: list[str],
    start: str = "2009-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Download daily OHLCV + adj_close for every ticker.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to fetch.
    start : str
        ISO date string for the start of the history window.
    end : str, optional
        ISO date string for the end of the window. Defaults to today.
    force_refresh : bool
        Re-download even if the per-ticker cache is fresh.

    Returns
    -------
    (price_panel, failed_tickers)
        price_panel : pd.DataFrame
            Long-format panel. Columns: ticker, date, open, high, low,
            close, adj_close, volume, daily_return.
        failed_tickers : list[str]
            Tickers that failed all retries. Excluded from price_panel.
    """
    if end is None:
        end = datetime.date.today().isoformat()

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        if i > 0 and i % 50 == 0:
            print(f"  [{i}/{len(tickers)}] pulling price data...")

        path = _cache_path(ticker)

        if not force_refresh and _is_cache_fresh(path):
            frames.append(pd.read_parquet(path))
            continue

        try:
            raw = _fetch_with_retry(ticker, start, end)
        except Exception as exc:
            print(f"  WARNING: failed to fetch {ticker}: {exc}")
            failed.append(ticker)
            continue

        df = _build_price_df(raw, ticker)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        frames.append(df)

    if not frames:
        _COLS = ["ticker", "date", "open", "high", "low",
                 "close", "adj_close", "volume", "daily_return"]
        return pd.DataFrame(columns=_COLS), failed

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel, failed


def pull_market_benchmark(
    symbol: str = "SPY",
    start: str = "2009-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download daily prices for the market benchmark.

    Uses the same cache / retry logic as pull_prices_for_tickers.
    SPY is the default; '^GSPC' works too but has more historical gaps.

    Returns
    -------
    pd.DataFrame
        Same schema as the panel returned by pull_prices_for_tickers,
        with a single ticker value equal to `symbol`.
    """
    panel, failed = pull_prices_for_tickers(
        [symbol], start=start, end=end, force_refresh=force_refresh
    )
    if failed:
        raise RuntimeError(f"Failed to fetch market benchmark '{symbol}'")
    return panel
