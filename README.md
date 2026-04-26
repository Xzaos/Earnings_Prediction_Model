# Earnings Prediction Engine

ML model predicting earnings surprises for US equities, using only free data sources
(SEC EDGAR + yfinance). Research-focused: notebook + backtest, not production trading.

## What this project predicts

The headline target is a **time-series SUE** in the spirit of Foster, Olsen & Shevlin (1984):

```
SUE_t = (EPS_t - E[EPS_t]) / σ(ε_t)
```

where `E[EPS_t]` is the firm's expected EPS under a seasonal random walk with drift
(SRWD) model fit on the firm's own past 8–20 quarterly EPS values, and `σ(ε_t)` is the
standard deviation of that model's residuals. This is *not* the analyst-consensus SUE
from Livnat-Mendenhall — we don't have the analyst data to compute that one. See
`docs/target_choice.md` (or the relevant section below) for why this substitution is
defensible.

A secondary, **proxy analyst SUE** target uses whatever consensus history yfinance
exposes:

```
ProxySUE_t = (EPS_actual - EPS_consensus) / |EPS_consensus|
```

This is computable for ~2–4 years of recent history per ticker and serves as a
sanity-check signal, not the primary target.

## Why time-series SUE, not analyst SUE

The original project spec called for analyst-consensus SUE. We can't build that
from free sources:

- yfinance exposes a current consensus point estimate plus a shallow surprise
  history (typically 4–8 quarters). It does **not** expose per-analyst estimates,
  the cross-sectional standard deviation of estimates, or a clean revision history.
- SEC EDGAR has actuals but no estimates.
- The literal SUE denominator (cross-sectional σ across analysts) is therefore
  impossible to compute correctly.

Time-series SUE (Foster 1977; Foster, Olsen & Shevlin 1984) was the original SUE
formulation, predates analyst-based versions, and remains a documented predictor
of post-earnings drift. It uses only the firm's own EPS history, which we can
extract cleanly from EDGAR.

This is an honest substitution, not a workaround: the resulting signal is
"earnings surprise relative to the firm's own predictable pattern," which is
arguably closer to what fundamental investors actually care about than a deviation
from sell-side consensus.

## Universe

S&P 500 constituents, 2010-Q1 onward, with point-in-time membership (firms
included for the quarters they were actually in the index, dropped after removal).
Roughly 40 quarters × 500 names ≈ 20,000 firm-quarter observations once we
exclude observations without enough history for the SRWD model to be fit.

## The leakage discipline

This project's single biggest correctness risk is forward-looking bias.
Two specific traps:

**Trap 1: Period-end vs filing-date.** A Q3 2023 quarter (period ending Sep 30) is
typically reported in late October or early November via 10-Q. Computing features
"as-of" the period-end date uses information that wasn't yet public. **Every feature
must be timestamped by the EDGAR `filed` date**, and predictions for quarter Q's
announcement must use only data filed strictly before Q's announcement date.

**Trap 2: Restated fundamentals.** EDGAR `companyfacts` returns as-amended values, not
as-originally-reported. For most fundamentals (margins, growth) the difference is
small. For accruals it can be material. We accept this in v1 and document it as a
known limitation.

We enforce trap 1 with `tests/test_no_leakage.py`, which for a sample of
(firm, quarter) pairs asserts that every input feature's source-data filing date
is strictly less than the announcement date. This test must pass in CI; if it
fails, the model results are not trustworthy.

## Repo layout

```
earnings_prediction/
├── README.md                  ← you are here
├── src/
│   ├── universe.py            ← point-in-time S&P 500 membership
│   ├── edgar_pull.py          ← SEC companyfacts API + filing dates
│   ├── prices_pull.py         ← yfinance with retry/cache
│   ├── targets.py             ← FOS time-series SUE, proxy SUE, 3-day CAR
│   ├── features_fund.py       ← accruals, margin deltas, growth, etc.
│   ├── features_tech.py       ← momentum, vol, surprise track record
│   ├── panel.py               ← point-in-time join, leak checks
│   ├── models.py              ← ridge/lasso, lgbm, time-series CV
│   └── evaluate.py            ← IC, quintile spreads, decay
├── tests/
│   └── test_no_leakage.py     ← THE most important test in the repo
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_targets_construction.ipynb   ← starts here
│   ├── 03_features_eda.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_backtest.ipynb
└── data/
    ├── raw/        ← EDGAR JSONs, yfinance pulls (gitignored)
    ├── interim/    ← parsed quarterly EPS panels
    └── processed/  ← final feature matrix, targets
```

## Build order

1. **Targets first.** `src/targets.py` and `notebooks/02_targets_construction.ipynb`.
   Build SUE on a small sample (10 tickers, recent quarters) and eyeball it. Verify
   the SRWD residuals look roughly mean-zero and that the SUE distribution is
   roughly bell-shaped with fat tails. **This is the current scaffold's stopping
   point.**
2. **Universe.** `src/universe.py` — historical S&P 500 membership with add/remove dates.
3. **EDGAR puller.** `src/edgar_pull.py` — companyfacts API, rate-limited, cached.
4. **Leakage test.** `tests/test_no_leakage.py` runs against pulled data; should
   pass before any modeling.
5. **Features.** Fundamental first, then technical.
6. **Models.** Ridge → LightGBM → ensemble. Time-series CV only.
7. **Evaluation.** Quarterly IC, quintile spread, decay.

## Evaluation metrics

- **Information Coefficient (IC):** Spearman rank correlation between predicted
  SUE and realized SUE, computed cross-sectionally per quarter, then averaged.
  Target: long-run mean IC of 0.05–0.10 (commercially meaningful in this domain).
- **Quintile hit rate:** Form quintiles by predicted SUE; measure realized
  beat-rate and average realized SUE per quintile. Top minus bottom should be
  positive and monotone.
- **IC stability:** Standard deviation of quarterly IC. A model with mean IC 0.07
  and std 0.04 is much better than mean 0.07 std 0.15.
- **Sector neutrality check:** Compute IC within each GICS sector. A signal
  that only works in one sector is fragile.

## Decisions deferred

- **Options-implied features** (ATM straddle, earnings premium): require paid
  data. Out of scope for v1.
- **Textual features** from 10-Q MD&A: phase 2.
- **As-originally-reported fundamentals:** would need EDGAR's amendment-tracking;
  v1 accepts as-amended.
- **Universe expansion** beyond S&P 500: do this once v1 IC is established on
  the cleaner universe.

## Citations

- Foster, G. (1977). Quarterly accounting data: Time-series properties and
  predictive-ability results. *The Accounting Review*, 52(1), 1–21.
- Foster, G., Olsen, C., & Shevlin, T. (1984). Earnings releases, anomalies, and
  the behavior of security returns. *The Accounting Review*, 59(4), 574–603.
- Livnat, J., & Mendenhall, R. R. (2006). Comparing the post-earnings
  announcement drift for surprises calculated from analyst and time series
  forecasts. *Journal of Accounting Research*, 44(1), 177–205.
- Sloan, R. G. (1996). Do stock prices fully reflect information in accruals
  and cash flows about future earnings? *The Accounting Review*, 71(3), 289–315.
