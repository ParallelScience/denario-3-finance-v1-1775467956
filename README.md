# Cross-Sectional Dispersion of Realized Volatility as a Leading Indicator for VIX Regime Transitions

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-04-06
**Status:** Methods generated — awaiting results

## Idea

Cross-sectional standard deviation of 21-day rolling log-return volatility across 10 sector ETFs (SVD) as a leading indicator for VIX regime transitions. Hypothesis: SVD expansion precedes systemic VIX shifts even when market beta is stable.

## Methods (8 steps)

1. Data preprocessing — load close_prices.csv, filter 10 sector ETFs + VIX, compute log-returns
2. SVD construction — 21-day rolling realized vol per ETF → CSSD → Z-score normalize
3. VIX regime definition — HMM (3 states: Low/Moderate/High) with expanding window to avoid look-ahead bias
4. Stationarity & feature engineering — ADF tests, lagged SVD (1d/5d/21d), rolling avg pairwise correlation
5. Decoupling analysis — SVD × AvgCorrelation interaction term
6. Predictive regression — Δln(VIX)_{t+k} ~ SVD + controls, Newey-West SEs, k=5 and 21 days
7. Regime transition probability — logistic regression → AUROC + Precision-Recall, SVD window sensitivity
8. Robustness — leave-one-out cross-validation, PCA comparison

## Progress

| Step | Iteration 0 |
|------|------------|
| Setup | done |
| Idea | done |
| Methods | done |
| Results | |
| Evaluate | |
| Paper | |

