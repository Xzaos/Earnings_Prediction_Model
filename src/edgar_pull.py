"""
SEC EDGAR companyfacts puller.

Fetches quarterly fundamental facts for S&P 500 tickers using the EDGAR
companyfacts API, with caching, rate limiting, and strict point-in-time
correctness (earliest filing per period, never amendments).

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING — FY ROWS CONTAIN CUMULATIVE ANNUAL EPS, NOT Q4-ONLY EPS
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Rows with fp == 'FY' come from 10-K annual filings. Their `value` (and `eps`
in the backward-compat view) is the full-year cumulative figure, NOT the Q4
standalone figure. Annual filers (including Apple, Microsoft, most S&P 500
companies) do not file a separate 10-Q for Q4; the 10-K is the only source.

Q4 EPS MUST be derived in feature engineering as:
    Q4_EPS = FY_EPS - Q1_EPS - Q2_EPS - Q3_EPS

Downstream code that consumes the DataFrame returned by pull_eps_for_tickers
MUST either:
  (a) Filter to fp in {Q1, Q2, Q3} and forgo Q4 entirely, OR
  (b) Call derive_quarterly_eps() (defined in this module) which performs
      the subtraction and returns a clean quarterly-only panel.

Passing FY rows directly into a model or feature pipeline as if they were
quarterly observations will silently inject annualised values that are
approximately 4× larger than true quarterly EPS — a major data error.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Point-in-time rule
------------------
EDGAR companyfacts returns all versions of every fact, including amendments.
Using an amended value (e.g. 10-Q/A filed 90 days after the original 10-Q)
for a prediction made at time T would introduce silent look-ahead bias if the
amendment wasn't yet filed at T. We therefore keep only the EARLIEST filing
for each (cik, period_end) pair — i.e., the value that was first made public.

Duration filter
---------------
EDGAR stores both standalone (~90d) and cumulative YTD (~180d for H1, ~270d
for 9-month) income-statement facts for Q2 and Q3, all sharing the same
period_end and fp label. Only the standalone (~90d) entry reflects the quarter
in isolation; YTD entries would corrupt the FY-Q1-Q2-Q3 subtraction used in
derive_quarterly_eps. We keep Q1/Q2/Q3/Q4 with duration 60–130 days and FY
with duration 340–400 days.

Balance sheet items (Assets, Liabilities, etc.) are point-in-time "instant"
snapshots: EDGAR stores only an end date, no start date. For these facts the
duration filter is skipped entirely (controlled by presence of the 'start'
field in the raw fact dict).
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Constants — edit USER_AGENT before first run                                #
# --------------------------------------------------------------------------- #

# SEC requires a descriptive User-Agent or returns 403.
# Format: "<Name> <ProjectName> <ContactEmail>"
USER_AGENT = "Ritwik EarningsPredictionEngine ritwikmishra13539@gmail.com"

EDGAR_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_REPO_ROOT = Path(__file__).parent.parent
CACHE_DIR = _REPO_ROOT / "data" / "raw" / "edgar"
CACHE_MAX_AGE_DAYS = 7

# Minimum gap between requests — 10 req/s max as per SEC fair-use policy.
_MIN_REQUEST_INTERVAL = 1.0 / 10.0  # seconds

_last_request_time: float = 0.0

# --------------------------------------------------------------------------- #
# XBRL tag and unit mappings                                                  #
# --------------------------------------------------------------------------- #

# Maps a canonical fact name to an ordered list of XBRL tag candidates.
# _extract_facts tries each tag in order and stops at the first non-empty one.
# This handles tag renames across GAAP taxonomy versions (e.g., ASC 606 revenue).
XBRL_TAGS: dict[str, list[str]] = {
    "eps_diluted":             ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "revenue":                 ["SalesRevenueNet",
                                "RevenueFromContractWithCustomerExcludingAssessedTax",
                                "Revenues"],
    "cost_of_revenue":        ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit":           ["GrossProfit"],
    "operating_income":       ["OperatingIncomeLoss"],
    "net_income":             ["NetIncomeLoss"],
    "total_assets":           ["Assets"],
    "total_current_assets":   ["AssetsCurrent"],
    "cash":                   ["CashAndCashEquivalentsAtCarryingValue", "Cash"],
    "total_current_liabilities": ["LiabilitiesCurrent"],
    "short_term_debt":        ["DebtCurrent", "ShortTermBorrowings"],
    "income_taxes_payable":   ["AccruedIncomeTaxesCurrent", "TaxesPayableCurrent"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization",
                                  "DepreciationAndAmortization",
                                  "Depreciation"],
    "accounts_receivable":    ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory":              ["InventoryNet"],
    "capex":                  ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

# Maps a canonical fact name to the EDGAR unit string used to look up facts
# within the "units" dict of a companyfacts entry.
# Facts not listed here default to "USD" via XBRL_UNITS.get(fact_name, "USD").
XBRL_UNITS: dict[str, str] = {
    "eps_diluted": "USD/shares",
    # share-count facts (e.g. shares_outstanding) would use "shares" here
}


# --------------------------------------------------------------------------- #
# Network / cache helpers                                                      #
# --------------------------------------------------------------------------- #


def _cache_path(cik: int) -> Path:
    return CACHE_DIR / f"CIK{cik:010d}.json"


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return age.days < CACHE_MAX_AGE_DAYS


def _fetch_companyfacts(cik: int, force_refresh: bool = False) -> dict:
    """Return the companyfacts JSON for one CIK, using disk cache when fresh."""
    global _last_request_time

    path = _cache_path(cik)

    if not force_refresh and _is_cache_fresh(path):
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    # Rate-limit: sleep if needed to stay under 10 req/s.
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    url = EDGAR_FACTS_URL.format(cik=cik)
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_request_time = time.monotonic()
    resp.raise_for_status()

    data = resp.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


# --------------------------------------------------------------------------- #
# Fact extraction                                                              #
# --------------------------------------------------------------------------- #

# Quarterly period codes. "FY" annual filings are kept because they represent
# Q4 for calendar-year filers; feature engineering handles the labelling.
_QUARTERLY_FP = {"Q1", "Q2", "Q3", "Q4", "FY"}

# Forms we accept. Exclude 10-K/A, 10-Q/A etc. — the point-in-time rule
# (keep earliest filing) handles amendments automatically, but filtering forms
# here provides a useful belt-and-suspenders guard.
_ACCEPTED_FORMS = {"10-Q", "10-K", "10-K/A", "10-Q/A", "20-F", "20-F/A"}


def _extract_facts(
    data: dict,
    tag_candidates: list[str],
    ticker: str,
    cik: int,
    value_unit: str,
) -> pd.DataFrame:
    """Extract one fact from a companyfacts JSON blob.

    Tries each tag in tag_candidates in order; stops at the first non-empty
    one. Applies the duration filter for income-statement facts (those with a
    'start' field) and skips it for balance-sheet instant facts (no 'start').
    Deduplicates by (cik, period_end), keeping the earliest-filed value.

    Returns a DataFrame with columns:
        ticker, cik, period_end, value, filed, form, fp, accn
    Returns an empty DataFrame (correct columns) when no data is found.
    """
    _COLS = ["ticker", "cik", "period_end", "value", "filed", "form", "fp", "accn"]

    us_gaap = data.get("facts", {}).get("us-gaap", {})

    raw_facts: list | None = None
    for tag in tag_candidates:
        entry = us_gaap.get(tag, {})
        units = entry.get("units", {})
        facts = units.get(value_unit) or (next(iter(units.values()), None) if units else None)
        if facts:
            raw_facts = facts
            break

    if not raw_facts:
        return pd.DataFrame(columns=_COLS)

    rows = []
    for fact in raw_facts:
        fp = fact.get("fp", "")
        if fp not in _QUARTERLY_FP:
            continue

        # XBRL distinguishes durations (income statement, cash flow — has both 'start' and 'end')
        # from instants (balance sheet — has only 'end'). The presence/absence of 'start' is the
        # structural signal. We only filter by duration when start is genuinely present.
        if "start" in fact:
            start_str = fact["start"]
            end_str = fact.get("end", "")
            if not start_str or not end_str:
                # Malformed fact — start key present but empty/null. Skip rather than guess.
                continue
            dur = (datetime.date.fromisoformat(end_str) - datetime.date.fromisoformat(start_str)).days
            if fp == "FY" and not (340 <= dur <= 400):
                continue
            if fp in {"Q1", "Q2", "Q3", "Q4"} and not (60 <= dur <= 130):
                continue
        # else: instant fact, no duration filter applies

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "period_end": pd.to_datetime(fact["end"]),
            "value": float(fact["val"]),
            "filed": pd.to_datetime(fact["filed"]),
            "form": fact.get("form", ""),
            "fp": fp,
            "accn": fact.get("accn", ""),
        })

    if not rows:
        return pd.DataFrame(columns=_COLS)

    df = pd.DataFrame(rows)

    # Point-in-time deduplication: for each (cik, period_end) keep only the
    # EARLIEST filing. This ensures we never use a restated/amended value that
    # wasn't yet public at prediction time.
    df = (
        df.sort_values("filed")
          .drop_duplicates(subset=["cik", "period_end"], keep="first")
          .sort_values(["period_end", "filed"])
          .reset_index(drop=True)
    )

    return df


def _extract_eps_facts(data: dict, cik: int, ticker: str) -> pd.DataFrame:
    """Backward-compatible wrapper around _extract_facts for EPS.

    Returns the same schema as before the refactor:
        ticker, cik, period_end, eps, filed, form, fp, accn
    """
    _EPS_COLS = ["ticker", "cik", "period_end", "eps", "filed", "form", "fp", "accn"]
    df = _extract_facts(
        data,
        tag_candidates=XBRL_TAGS["eps_diluted"],
        ticker=ticker,
        cik=cik,
        value_unit=XBRL_UNITS.get("eps_diluted", "USD"),
    )
    if df.empty:
        return pd.DataFrame(columns=_EPS_COLS)
    return df.rename(columns={"value": "eps"})


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def pull_facts_for_tickers(
    tickers: list[str],
    cik_lookup: dict[str, str],
    fact_names: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download one or more fundamental facts for every ticker.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to fetch. Must all have entries in `cik_lookup`.
    cik_lookup : dict[str, str]
        Mapping from ticker to CIK string (zero-padded or plain integer string).
    fact_names : list[str]
        Canonical fact names to extract. Each must be a key in XBRL_TAGS.
        Example: ["eps_diluted", "revenue", "total_assets"]
    force_refresh : bool
        Re-fetch from EDGAR even if the per-ticker cache is fresh.

    Returns
    -------
    pd.DataFrame
        Long-format panel. Columns:
            ticker, cik, period_end, fact_name, value, filed, form, fp, accn
        One row per (ticker, period_end, fact_name) after point-in-time
        deduplication. FY rows carry cumulative annual values (not Q4-only);
        pass through derive_quarterly_eps() for EPS before modelling.
    """
    _COLS = ["ticker", "cik", "period_end", "fact_name", "value", "filed", "form", "fp", "accn"]

    frames: list[pd.DataFrame] = []
    missing_cik: list[str] = []

    for i, ticker in enumerate(tickers):
        if i > 0 and i % 50 == 0:
            print(f"  [{i}/{len(tickers)}] pulling EDGAR data...")

        raw_cik = cik_lookup.get(ticker)
        if not raw_cik:
            missing_cik.append(ticker)
            continue

        cik = int(raw_cik)
        try:
            data = _fetch_companyfacts(cik, force_refresh=force_refresh)
        except requests.HTTPError as exc:
            print(f"  WARNING: HTTP {exc.response.status_code} for {ticker} (CIK {cik:010d})")
            continue
        except Exception as exc:
            print(f"  WARNING: Failed to fetch {ticker}: {exc}")
            continue

        for fact_name in fact_names:
            tag_candidates = XBRL_TAGS.get(fact_name, [fact_name])
            value_unit = XBRL_UNITS.get(fact_name, "USD")
            df = _extract_facts(data, tag_candidates, ticker=ticker, cik=cik, value_unit=value_unit)
            if not df.empty:
                df.insert(df.columns.get_loc("value"), "fact_name", fact_name)
                frames.append(df)

    if missing_cik:
        print(f"  WARNING: No CIK found for {len(missing_cik)} tickers: {missing_cik[:10]}...")

    if not frames:
        return pd.DataFrame(columns=_COLS)

    out = pd.concat(frames, ignore_index=True)
    out["period_end"] = pd.to_datetime(out["period_end"])
    out["filed"] = pd.to_datetime(out["filed"])
    return out


def pull_eps_for_tickers(
    tickers: list[str],
    cik_lookup: dict[str, str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull quarterly EPS for every ticker, return a combined panel.

    Thin wrapper around pull_facts_for_tickers that reshapes the output to
    the original contract: columns ticker, cik, period_end, eps, filed, form,
    fp, accn (one row per (ticker, period_end) after point-in-time dedup).

    WARNING: rows where fp == 'FY' carry cumulative annual EPS, not Q4-only.
    Pass this DataFrame through derive_quarterly_eps() before any modelling
    or feature-engineering step that expects standalone quarterly figures.
    """
    _EPS_COLS = ["ticker", "cik", "period_end", "eps", "filed", "form", "fp", "accn"]
    df = pull_facts_for_tickers(tickers, cik_lookup, ["eps_diluted"], force_refresh=force_refresh)
    if df.empty:
        return pd.DataFrame(columns=_EPS_COLS)
    return df.rename(columns={"value": "eps"}).drop(columns=["fact_name"])


def derive_quarterly_eps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw EDGAR panel into a clean quarterly-only panel.

    Removes fp == 'FY' rows and replaces them with derived Q4 rows where
    all four fiscal-year components are available:

        Q4_EPS = FY_EPS - Q1_EPS - Q2_EPS - Q3_EPS

    The derived Q4 row inherits period_end, filed, form, and accn from the
    FY row so that point-in-time correctness is preserved (Q4 is not known
    until the 10-K is filed).

    Rules
    -----
    - For each (ticker, fiscal year), a Q4 is derived only when Q1, Q2, Q3,
      and FY rows are ALL present for that fiscal year.
    - Fiscal year membership is determined by whether a quarterly period_end
      falls within the 366 days immediately before the FY period_end.
    - If an explicit fp == 'Q4' row already exists in the input for a given
      (ticker, period_end), it is passed through unchanged and no derivation
      is attempted for that fiscal year.
    - If any of Q1/Q2/Q3/FY is absent, the available quarterly rows are
      emitted as-is; no Q4 row is produced for that fiscal year.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of pull_eps_for_tickers / _extract_eps_facts.

    Returns
    -------
    pd.DataFrame
        Same columns as the input. fp values are a subset of
        {Q1, Q2, Q3, Q4}; all FY rows are removed.
    """
    _WINDOW = pd.Timedelta(days=366)

    quarterly = df[df["fp"].isin({"Q1", "Q2", "Q3"})].copy()
    explicit_q4 = df[df["fp"] == "Q4"].copy()
    annual = df[df["fp"] == "FY"].copy()

    derived_rows: list[dict] = []

    for _, fy_row in annual.iterrows():
        ticker = fy_row["ticker"]
        fy_end = pd.Timestamp(fy_row["period_end"])

        # Skip derivation if an explicit Q4 already exists for this period.
        already_has_q4 = (
            (explicit_q4["ticker"] == ticker) &
            (explicit_q4["period_end"] == fy_end)
        ).any()
        if already_has_q4:
            continue

        # Collect the three interim quarters that belong to this fiscal year.
        mask = (
            (quarterly["ticker"] == ticker) &
            (quarterly["period_end"] > fy_end - _WINDOW) &
            (quarterly["period_end"] < fy_end)
        )
        q_rows = quarterly[mask]

        if set(q_rows["fp"]) != {"Q1", "Q2", "Q3"}:
            continue  # incomplete year — no derivation

        q1_eps = float(q_rows.loc[q_rows["fp"] == "Q1", "eps"].iloc[0])
        q2_eps = float(q_rows.loc[q_rows["fp"] == "Q2", "eps"].iloc[0])
        q3_eps = float(q_rows.loc[q_rows["fp"] == "Q3", "eps"].iloc[0])

        derived_rows.append({
            "ticker": ticker,
            "cik": fy_row["cik"],
            "period_end": fy_end,
            "eps": float(fy_row["eps"]) - q1_eps - q2_eps - q3_eps,
            "filed": fy_row["filed"],
            "form": fy_row["form"],
            "fp": "Q4",
            "accn": fy_row["accn"],
        })

    parts = [quarterly, explicit_q4]
    if derived_rows:
        parts.append(pd.DataFrame(derived_rows))

    non_empty = [p for p in parts if len(p) > 0]
    if not non_empty:
        return pd.DataFrame(columns=df.columns)

    result = pd.concat(non_empty, ignore_index=True)
    return result.sort_values(["ticker", "period_end"]).reset_index(drop=True)
