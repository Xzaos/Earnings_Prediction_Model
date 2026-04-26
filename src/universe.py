"""
Point-in-time S&P 500 universe construction.

Scrapes the Wikipedia list of S&P 500 companies and reconstructs historical
membership by reversing add/remove changes from the current state back to any
target date. CIK numbers (for EDGAR pulls) are included in the current-members
table.

Cache strategy: raw HTML is stored at data/raw/sp500_wiki.html and reused for
up to 30 days before re-fetching. Pass force_refresh=True to bypass the cache.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_REPO_ROOT = Path(__file__).parent.parent
CACHE_PATH = _REPO_ROOT / "data" / "raw" / "sp500_wiki.html"
CACHE_MAX_AGE_DAYS = 30


# --------------------------------------------------------------------------- #
# HTML fetch / cache                                                           #
# --------------------------------------------------------------------------- #


def _fetch_html(force_refresh: bool = False) -> str:
    """Return Wikipedia page HTML, using the on-disk cache when it is fresh."""
    if not force_refresh and CACHE_PATH.exists():
        mtime = datetime.datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
        age = datetime.datetime.now() - mtime
        if age.days < CACHE_MAX_AGE_DAYS:
            return CACHE_PATH.read_text(encoding="utf-8")

    resp = requests.get(
        WIKI_URL,
        headers={"User-Agent": "earnings-prediction-research/1.0"},
        timeout=30,
    )
    resp.raise_for_status()
    html = resp.text
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(html, encoding="utf-8")
    return html


# --------------------------------------------------------------------------- #
# HTML table utilities                                                         #
# --------------------------------------------------------------------------- #


def _expand_table(table) -> list[list[str]]:
    """Flatten an HTML table with rowspan/colspan into a 2-D list of strings.

    Wikipedia's changes table uses rowspan on date cells when multiple changes
    share the same date. Without this expansion those rows would be mis-indexed.
    """
    grid: dict[tuple[int, int], str] = {}

    for row_idx, tr in enumerate(table.find_all("tr")):
        col_idx = 0
        for cell in tr.find_all(["td", "th"]):
            while (row_idx, col_idx) in grid:
                col_idx += 1
            text = cell.get_text(separator=" ", strip=True)
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for r in range(rowspan):
                for c in range(colspan):
                    grid[(row_idx + r, col_idx + c)] = text
            col_idx += colspan

    if not grid:
        return []

    max_row = max(r for r, _ in grid) + 1
    max_col = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(max_col)] for r in range(max_row)]


# --------------------------------------------------------------------------- #
# Table parsers                                                                #
# --------------------------------------------------------------------------- #


def _parse_current_members(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse the current-members table (id='constituents').

    Returns
    -------
    pd.DataFrame
        Columns: ticker, security, sector, cik, date_added.
    """
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise ValueError("Could not find 'constituents' table on the Wikipedia page.")

    rows = _expand_table(table)
    if len(rows) < 2:
        return pd.DataFrame(columns=["ticker", "security", "sector", "cik", "date_added"])

    headers = [h.lower().strip() for h in rows[0]]

    def _col(*fragments: str) -> int | None:
        for i, h in enumerate(headers):
            if any(f in h for f in fragments):
                return i
        return None

    ticker_col = _col("symbol", "ticker")
    security_col = _col("security", "name")
    sector_col = _col("gics sector", "sector")
    cik_col = _col("cik")
    date_added_col = _col("date added", "date")

    if ticker_col is None:
        raise ValueError(f"Could not find ticker column in headers: {headers}")

    out = []
    for row in rows[1:]:
        if len(row) <= ticker_col:
            continue
        ticker = row[ticker_col].strip()
        if not ticker:
            continue
        out.append({
            "ticker": ticker,
            "security": row[security_col].strip() if security_col is not None else "",
            "sector": row[sector_col].strip() if sector_col is not None else "",
            "cik": row[cik_col].strip() if cik_col is not None else "",
            "date_added": row[date_added_col].strip() if date_added_col is not None else "",
        })

    return pd.DataFrame(out)


def _parse_changes(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse the historical changes table (id='changes').

    Returns
    -------
    pd.DataFrame
        Columns: date (pd.Timestamp), added_ticker (str), removed_ticker (str).
        Rows where a ticker is absent (cell is blank) have an empty string.
    """
    table = soup.find("table", {"id": "changes"})
    if table is None:
        raise ValueError("Could not find 'changes' table on the Wikipedia page.")

    all_rows = _expand_table(table)

    # Determine the number of header rows to skip.
    # The changes table has a 2-row header: first row has "Date", "Added",
    # "Removed", "Reason" with colspan/rowspan; second row has "Ticker",
    # "Security", "Ticker", "Security". After expansion row 0 and row 1 are
    # headers. We skip rows until we find a parseable date in column 0.
    data_start = 0
    for i, row in enumerate(all_rows):
        try:
            pd.to_datetime(row[0])
            data_start = i
            break
        except Exception:
            continue

    out = []
    for row in all_rows[data_start:]:
        if len(row) < 4:
            continue
        date_str = row[0].strip()
        if not date_str:
            continue
        try:
            date = pd.to_datetime(date_str)
        except Exception:
            continue
        added_ticker = row[1].strip()
        removed_ticker = row[3].strip()
        out.append({
            "date": date,
            "added_ticker": added_ticker,
            "removed_ticker": removed_ticker,
        })

    if not out:
        return pd.DataFrame(columns=["date", "added_ticker", "removed_ticker"])
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def current_members(force_refresh: bool = False) -> pd.DataFrame:
    """Return the current S&P 500 member list with CIK numbers.

    Parameters
    ----------
    force_refresh : bool
        Re-fetch Wikipedia even if the local cache is fresh.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, security, sector, cik, date_added.
        One row per current S&P 500 constituent.
    """
    html = _fetch_html(force_refresh)
    soup = BeautifulSoup(html, "lxml")
    return _parse_current_members(soup)


def cik_map(force_refresh: bool = False) -> dict[str, str]:
    """Return a {ticker: cik} mapping for current S&P 500 members.

    CIK values come from the Wikipedia constituents table and are strings
    (zero-padded to 10 digits by Wikipedia, e.g. '0000320193' for AAPL).
    They are needed for EDGAR API calls in edgar_pull.py.
    """
    df = current_members(force_refresh)
    return dict(zip(df["ticker"], df["cik"]))


def members_as_of(date: pd.Timestamp, force_refresh: bool = False) -> list[str]:
    """Return S&P 500 tickers that were members as of the given date.

    Algorithm: start from the current membership, then reverse every add/remove
    change that occurred *after* `date`. A ticker that was added after `date` is
    removed from the reconstructed set; a ticker that was removed after `date` is
    added back.

    Parameters
    ----------
    date : pd.Timestamp
        Point-in-time date. Membership as of end-of-day on this date.
    force_refresh : bool
        Re-fetch Wikipedia even if the local cache is fresh.

    Returns
    -------
    list[str]
        Sorted list of ticker symbols that were S&P 500 members as of `date`.
    """
    html = _fetch_html(force_refresh)
    soup = BeautifulSoup(html, "lxml")

    members_df = _parse_current_members(soup)
    current: set[str] = set(members_df["ticker"].tolist())

    changes_df = _parse_changes(soup)
    if changes_df.empty:
        return sorted(current)

    # Only undo changes that happened *after* the query date.
    after = changes_df[changes_df["date"] > date].sort_values("date", ascending=False)

    for _, row in after.iterrows():
        added = row["added_ticker"].strip()
        removed = row["removed_ticker"].strip()
        if added:
            current.discard(added)
        if removed:
            current.add(removed)

    return sorted(current)
