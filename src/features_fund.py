"""
Fundamental feature engineering from the EDGAR facts panel.

All features are computed point-in-time: inputs for a row at period_end Q
use only data filed on or before Q's own filing date. The feature_filed
column records the maximum filing date across all inputs (including lagged
values), enabling downstream leakage assertions.

Gap policy
----------
shift(4) assumes four consecutive quarterly rows per ticker. If consecutive
period_ends differ by outside 70–110 days, the alignment is broken. The
offending row and the next three rows (shift-window peers) have all lag-4
dependent features set to NaN, and a warning is logged.

Depreciation (Sloan)
--------------------
Annual depreciation for Sloan accruals = trailing 4-quarter sum when
quarterly data exists; falls back to the most recent FY figure (used
directly — it is already annual) joined by period_end proximity.

FY rows
-------
The pivot filters to fp ∈ {Q1, Q2, Q3, Q4}. FY rows are excluded from the
main computation; Q4 income-statement values will be NaN for firms that only
file a 10-K (v2 will generalise derive_quarterly_eps to all facts).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

_QUARTERLY_FP = {"Q1", "Q2", "Q3", "Q4"}

_FLOOR_ASSETS   = 1_000_000.0   # $1M — avg total assets denominator
_FLOOR_REVENUE  = 1_000_000.0   # $1M — single-quarter revenue denominators
_FLOOR_REV_4Q   = 4_000_000.0   # $4M — 4-quarter rolling revenue denominator
_OL_CLIP        = 20.0           # operating leverage symmetrical clip
_OL_REV_FLOOR   = 0.01           # 1% minimum |%ΔRevenue| denominator

_GAP_LO, _GAP_HI = 70, 110      # acceptable days between consecutive quarters

# Duration (income-statement / cash-flow) facts whose FY values are full-year
# cumulative, not standalone Q4. These are nulled for FY rows in the pivot so
# that shift(4) alignment works for companies whose Q4 is only in a 10-K filing
# (e.g. Apple, whose FY ends September and has no separate 10-Q for that quarter).
_INCOME_STMT_FACTS = frozenset({
    "revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income",
})
# capex and depreciation_amortization are NOT in _INCOME_STMT_FACTS because their
# FY rows carry a genuine annual figure (not an inflated sum of quarterly rows).
# However, AAPL (and many filers) report these as YTD-cumulative on 10-Qs:
#   Q1 ~90d (standalone), Q2 ~180d (H1 YTD), Q3 ~270d (9-month YTD), FY ~365d.
# The duration filter in _extract_facts already drops Q2/Q3 YTD entries (>130d),
# leaving only Q1 standalone rows. Standalone Q2/Q3 capex/dep require subtraction
# of prior-quarter YTD values — implement as derive_quarterly_durations() in v2.

_FEATURE_COLS = [
    "accruals_sloan",
    "gross_margin_change_yoy",
    "revenue_growth_yoy",
    "revenue_growth_qoq",
    "operating_leverage",
    "dso_change_yoy",
    "inventory_to_sales_change_yoy",
    "capex_to_sales",
]

_OUTPUT_COLS = ["ticker", "period_end", "feature_filed"] + _FEATURE_COLS


# ------------------------------------------------------------------ #
# Private helpers                                                      #
# ------------------------------------------------------------------ #


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if it exists, else a NaN Series aligned to df.index."""
    if col in df.columns:
        return df[col].copy()
    return pd.Series(np.nan, index=df.index, dtype=float)


def _rolling_max_dates(s: pd.Series, window: int) -> pd.Series:
    """Rolling max of a datetime Series within a pre-grouped context.

    Converts to nanosecond int (NaT → NaN via float cast), rolls, converts
    back. Must be called on a single-ticker slice that is already sorted.
    """
    ns = s.astype(np.int64).astype(float)
    ns[s.isna()] = np.nan
    rolled = ns.rolling(window, min_periods=1).max()
    return pd.to_datetime(rolled, unit="ns", errors="coerce")


def _mark_bad_lag4(df: pd.DataFrame) -> np.ndarray:
    """Return a boolean array (len = len(df)) marking rows whose shift(4)
    alignment is broken due to a gap in the quarterly sequence.

    A gap at position i (gap between row i-1 and row i outside 70–110 days)
    invalidates rows i through i+3 for lag-4 features.
    """
    bad = np.zeros(len(df), dtype=bool)
    pos = 0
    for ticker, grp in df.groupby("ticker", sort=False):
        n = len(grp)
        gaps = grp["period_end"].diff().dt.days.to_numpy()
        for i in range(1, n):
            if not np.isnan(gaps[i]) and not (_GAP_LO <= gaps[i] <= _GAP_HI):
                end = min(i + 4, n)
                bad[pos + i : pos + end] = True
                logger.warning(
                    "Ticker %s: gap of %.0f days before row %d "
                    "(period_end %s); lag-4 NaN'd for rows %d–%d.",
                    ticker, gaps[i], pos + i,
                    grp["period_end"].iloc[i].date(), pos + i, pos + end - 1,
                )
        pos += n
    return bad


def _pivot_quarterly(facts_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot long facts panel to wide, including FY rows to preserve shift(4) alignment.

    FY rows fill the 'Q4' slot for companies (e.g. Apple) that only file a 10-K for
    their fourth fiscal quarter. Income-statement facts at FY positions are nulled
    in compute_fundamental_features (they carry cumulative annual values, not Q4-only).
    Balance-sheet facts at FY positions are correct instant snapshots and are kept.

    Returns
    -------
    val_wide : DataFrame with one column per fact plus _fp (period type).
    filed_row : DataFrame with (ticker, period_end, _row_filed).
    """
    q = facts_panel[facts_panel["fp"].isin(_QUARTERLY_FP | {"FY"})].copy()

    val_wide = (
        q.pivot_table(
            index=["ticker", "period_end"],
            columns="fact_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    val_wide.columns.name = None

    # Track fp per row so compute_fundamental_features can null income-stmt facts
    # for FY rows before computing lags and features.
    fp_per_row = (
        q.groupby(["ticker", "period_end"])["fp"]
        .first()
        .reset_index()
        .rename(columns={"fp": "_fp"})
    )
    val_wide = val_wide.merge(fp_per_row, on=["ticker", "period_end"], how="left")

    filed_row = (
        q.groupby(["ticker", "period_end"])["filed"]
        .max()
        .reset_index()
        .rename(columns={"filed": "_row_filed"})
    )
    return val_wide, filed_row


def _join_fy_depreciation(val_wide: pd.DataFrame, facts_panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join the most recent FY depreciation onto each quarterly row.

    For row at period_end t, attaches the FY dep with the largest
    fy_period_end strictly less than t (point-in-time safe).
    """
    fy_dep = facts_panel[
        (facts_panel["fp"] == "FY") &
        (facts_panel["fact_name"] == "depreciation_amortization")
    ][["ticker", "period_end", "value", "filed"]].rename(
        columns={"period_end": "fy_period_end", "value": "dep_fy", "filed": "dep_fy_filed"}
    ).copy()

    if fy_dep.empty:
        val_wide["dep_fy"] = np.nan
        val_wide["dep_fy_filed"] = pd.NaT
        return val_wide

    parts = []
    for ticker, grp in val_wide.groupby("ticker"):
        fy_grp = fy_dep[fy_dep["ticker"] == ticker].sort_values("fy_period_end")
        grp = grp.sort_values("period_end").copy()
        if fy_grp.empty:
            grp["dep_fy"] = np.nan
            grp["dep_fy_filed"] = pd.NaT
            grp["fy_period_end"] = pd.NaT
        else:
            grp = pd.merge_asof(
                grp,
                fy_grp[["fy_period_end", "dep_fy", "dep_fy_filed"]],
                left_on="period_end",
                right_on="fy_period_end",
                direction="backward",
            )
            # Enforce strict inequality: drop matches where fy_period_end == period_end
            bad = grp["fy_period_end"] >= grp["period_end"]
            grp.loc[bad, ["dep_fy", "dep_fy_filed", "fy_period_end"]] = np.nan
        parts.append(grp)

    merged = pd.concat(parts, ignore_index=True)
    return merged.drop(columns=["fy_period_end"], errors="ignore")


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #


def compute_fundamental_features(
    facts_panel: pd.DataFrame,
    eps_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute fundamental features from the EDGAR long-format facts panel.

    Parameters
    ----------
    facts_panel : pd.DataFrame
        Output of pull_facts_for_tickers.
        Required columns: ticker, cik, period_end, fact_name, value, filed, form, fp, accn.
    eps_panel : pd.DataFrame, optional
        Output of derive_quarterly_eps. Unused in v1; reserved for future features.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame. Columns: ticker, period_end, feature_filed, + 8 feature columns.
        One row per (ticker, period_end). NaN where inputs are missing or denominators
        fall below their floor. feature_filed = max filing date across all inputs
        (current and lagged) that contributed to any feature in the row.
    """
    if facts_panel.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # ---- 1. Pivot (quarterly + FY rows) ----------------------------
    val_wide, filed_row = _pivot_quarterly(facts_panel)

    # ---- 2. Null income-statement facts for FY rows ----------------
    # FY income-statement values are full-year cumulative, not standalone Q4.
    # Balance-sheet facts at FY positions are correct and kept.
    is_fy = val_wide["_fp"] == "FY"
    if is_fy.any():
        for col in _INCOME_STMT_FACTS:
            if col in val_wide.columns:
                val_wide.loc[is_fy, col] = np.nan

    # ---- 3. FY depreciation fallback --------------------------------
    val_wide = _join_fy_depreciation(val_wide, facts_panel)

    # ---- 4. Sort and merge row-level filed date ---------------------
    val_wide = val_wide.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    filed_row = filed_row.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    val_wide = val_wide.merge(filed_row, on=["ticker", "period_end"], how="left")

    # Combine quarterly filed dates with FY dep filed date
    val_wide["_current_filed"] = val_wide[["_row_filed", "dep_fy_filed"]].max(axis=1)

    # ---- 5. Gap detection -------------------------------------------
    val_wide["_bad_lag4"] = _mark_bad_lag4(val_wide)

    # ---- 5. Lag columns (values) ------------------------------------
    _lag4_facts = [
        "revenue", "gross_profit", "operating_income",
        "total_current_assets", "cash", "total_current_liabilities",
        "short_term_debt", "income_taxes_payable", "total_assets",
        "accounts_receivable", "inventory",
    ]

    grp_v = val_wide.groupby("ticker")

    for col in _lag4_facts:
        val_wide[f"{col}_lag4"] = grp_v[col].shift(4) if col in val_wide.columns else np.nan

    val_wide["revenue_lag1"] = grp_v["revenue"].shift(1) if "revenue" in val_wide.columns else np.nan

    # ---- 6. Lag columns (filed dates) --------------------------------
    grp_f = val_wide.groupby("ticker")
    val_wide["_filed_lag4"] = grp_f["_current_filed"].shift(4)
    val_wide["_filed_lag1"] = grp_f["_current_filed"].shift(1)
    val_wide["_filed_roll4"] = grp_f["_current_filed"].transform(
        lambda s: _rolling_max_dates(s, 4)
    )

    # ---- 7. Rolling 4Q sums ----------------------------------------
    for col in ("depreciation_amortization", "capex", "revenue"):
        if col in val_wide.columns:
            val_wide[f"{col}_4q"] = grp_v[col].transform(
                lambda s: s.rolling(4, min_periods=4).sum()
            )
        else:
            val_wide[f"{col}_4q"] = np.nan

    # Resolve annual depreciation: prefer 4Q quarterly sum, else FY value
    val_wide["dep_annual"] = _safe(val_wide, "depreciation_amortization_4q").fillna(
        _safe(val_wide, "dep_fy")
    )

    # ---- 8. Apply bad_lag4 mask ------------------------------------
    _lag4_cols_to_null = [f"{c}_lag4" for c in _lag4_facts] + ["_filed_lag4"]
    bad = val_wide["_bad_lag4"].to_numpy()
    for col in _lag4_cols_to_null:
        if col in val_wide.columns:
            val_wide.loc[bad, col] = np.nan

    # Also null rolling-4Q features for bad-gap rows (window spans the gap)
    for col in ("depreciation_amortization_4q", "capex_4q", "revenue_4q", "dep_annual"):
        if col in val_wide.columns:
            val_wide.loc[bad, col] = np.nan

    # ---- 9. Compute features ----------------------------------------
    out = pd.DataFrame({"ticker": val_wide["ticker"], "period_end": val_wide["period_end"]})

    rev_t  = _safe(val_wide, "revenue")
    rev_t4 = _safe(val_wide, "revenue_lag4")
    rev_t1 = _safe(val_wide, "revenue_lag1")
    gp_t   = _safe(val_wide, "gross_profit")
    gp_t4  = _safe(val_wide, "gross_profit_lag4")
    oi_t   = _safe(val_wide, "operating_income")
    oi_t4  = _safe(val_wide, "operating_income_lag4")
    ta_t   = _safe(val_wide, "total_assets")
    ta_t4  = _safe(val_wide, "total_assets_lag4")
    ca_t   = _safe(val_wide, "total_current_assets")
    ca_t4  = _safe(val_wide, "total_current_assets_lag4")
    cash_t = _safe(val_wide, "cash")
    cash_t4= _safe(val_wide, "cash_lag4")
    cl_t   = _safe(val_wide, "total_current_liabilities")
    cl_t4  = _safe(val_wide, "total_current_liabilities_lag4")
    std_t  = _safe(val_wide, "short_term_debt")
    std_t4 = _safe(val_wide, "short_term_debt_lag4")
    tp_t   = _safe(val_wide, "income_taxes_payable")
    tp_t4  = _safe(val_wide, "income_taxes_payable_lag4")
    ar_t   = _safe(val_wide, "accounts_receivable")
    ar_t4  = _safe(val_wide, "accounts_receivable_lag4")
    inv_t  = _safe(val_wide, "inventory")
    inv_t4 = _safe(val_wide, "inventory_lag4")
    dep_a  = _safe(val_wide, "dep_annual")
    cap_4q = _safe(val_wide, "capex_4q")
    rev_4q = _safe(val_wide, "revenue_4q")

    # accruals_sloan
    avg_ta = (ta_t + ta_t4) / 2
    avg_ta_safe = avg_ta.where(avg_ta >= _FLOOR_ASSETS)
    numerator = (
        (ca_t - ca_t4) - (cash_t - cash_t4)
        - ((cl_t - cl_t4) - (std_t - std_t4) - (tp_t - tp_t4))
        - dep_a
    )
    out["accruals_sloan"] = numerator / avg_ta_safe

    # gross_margin_change_yoy
    rev_t_s  = rev_t.where(rev_t.abs()  >= _FLOOR_REVENUE)
    rev_t4_s = rev_t4.where(rev_t4.abs() >= _FLOOR_REVENUE)
    out["gross_margin_change_yoy"] = gp_t / rev_t_s - gp_t4 / rev_t4_s

    # revenue_growth_yoy
    out["revenue_growth_yoy"] = rev_t / rev_t4_s - 1

    # revenue_growth_qoq
    rev_t1_s = rev_t1.where(rev_t1.abs() >= _FLOOR_REVENUE)
    out["revenue_growth_qoq"] = rev_t / rev_t1_s - 1

    # operating_leverage
    delta_oi  = oi_t - oi_t4
    delta_rev = rev_t - rev_t4
    pct_oi  = delta_oi / oi_t4.where(oi_t4.abs() > 0)
    # Floor |%ΔRev| at OL_REV_FLOOR, preserving sign
    abs_delta_rev   = delta_rev.abs()
    min_delta_rev   = rev_t4.abs() * _OL_REV_FLOOR
    safe_abs_drev   = abs_delta_rev.where(abs_delta_rev >= min_delta_rev, min_delta_rev)
    sgn = np.sign(delta_rev).replace(0, 1.0)
    pct_rev = (safe_abs_drev * sgn) / rev_t4_s
    out["operating_leverage"] = (pct_oi / pct_rev).clip(-_OL_CLIP, _OL_CLIP)

    # dso_change_yoy
    dso_t  = (ar_t  / rev_t_s)  * 90
    dso_t4 = (ar_t4 / rev_t4_s) * 90
    out["dso_change_yoy"] = dso_t - dso_t4

    # inventory_to_sales_change_yoy
    out["inventory_to_sales_change_yoy"] = inv_t / rev_t_s - inv_t4 / rev_t4_s

    # capex_to_sales
    rev_4q_s = rev_4q.where(rev_4q.abs() >= _FLOOR_REV_4Q)
    out["capex_to_sales"] = cap_4q / rev_4q_s

    # ---- 10. feature_filed -----------------------------------------
    filed_inputs = val_wide[["_current_filed", "_filed_lag1", "_filed_lag4", "_filed_roll4"]]
    out["feature_filed"] = filed_inputs.max(axis=1)

    # ---- 11. Ensure all output columns present ----------------------
    for col in _OUTPUT_COLS:
        if col not in out.columns:
            out[col] = np.nan

    return out[_OUTPUT_COLS].reset_index(drop=True)
