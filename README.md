# The Conditional Predictive Power of Sectoral Volatility Dispersion for VIX Innovations

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-04-06
**Best iteration:** 0

**[View Paper & Presentation](https://ParallelScience.github.io/denario-3-finance-v1-1775467956/)**

## Abstract

This study investigates whether the cross-sectional dispersion of realized volatility across market sectors can serve as a leading indicator for shifts in the CBOE Volatility Index (VIX). We construct a daily Sectoral Volatility Dispersion (SVD) metric from ten US sector ETFs spanning 2015 to 2026 and employ a Hidden Markov Model to endogenously classify VIX regimes. Econometric analysis reveals that SVD, in isolation, is not a statistically significant predictor of future VIX innovations or transitions into high-volatility states. However, we uncover a crucial conditional relationship: elevated SVD is significantly associated with higher 21-day ahead VIX innovations only when accompanied by a breakdown in average cross-sector correlation. These findings indicate that cross-sectional dispersion should not be interpreted as a standalone timing signal, but rather as a component of a more nuanced market fragility indicator.

## Repository Structure

- `paper.tex` / `paper.pdf` — Final paper (Iteration 0)
- `presentation.mp3` — Audio presentation
- `docs/` — GitHub Pages site
- `Iteration0/` — Research pipeline outputs (idea → methods → results → paper)
- `data_description.md` — Dataset schema and documentation

---

## Finance Dataset: Daily OHLCV for US Equities, Indices, and Sector ETFs (2015–2026)

### Files

- **`/home/node/work/data/finance/ohlcv_daily.parquet`** — Full OHLCV data. Multi-level column index: outer level = field (Open, High, Low, Close, Volume), inner level = ticker symbol. Shape: (2830, 330). Load with `pd.read_parquet(...)`.
- **`/home/node/work/data/finance/close_prices.csv`** — Adjusted close prices only. Shape: (2830 rows × 66 columns). Index = trading dates (datetime), columns = ticker symbols. Load with `pd.read_csv(..., index_col=0, parse_dates=True)`.

### Tickers (66 total)

**Major US Indices (5):** `^GSPC` (S&P 500), `^DJI` (Dow Jones), `^IXIC` (Nasdaq Composite), `^RUT` (Russell 2000), `^VIX` (CBOE Volatility Index)

**Sector ETFs (11):** `XLK` (Tech), `XLF` (Financials), `XLV` (Healthcare), `XLE` (Energy), `XLI` (Industrials), `XLY` (Consumer Discretionary), `XLP` (Consumer Staples), `XLU` (Utilities), `XLB` (Materials), `XLRE` (Real Estate), `XLC` (Communication Services)

**Large-Cap Stocks (50):**
- Tech: AAPL, MSFT, NVDA, GOOGL, META, AMZN, TSLA, AVGO, ORCL, AMD
- Finance: JPM, BAC, GS, MS, V, MA, BRK-B, WFC, AXP, BLK
- Healthcare: UNH, JNJ, LLY, ABBV, MRK, PFE, TMO, DHR, ABT
- Energy: XOM, CVX, COP, SLB
- Consumer: WMT, COST, HD, MCD, NKE, SBUX
- Industrial: CAT, BA, GE, HON, UPS
- Telecom/Media: T, VZ, NFLX, DIS
- Other: LIN, NEE

### Data Properties

- **Date range:** 2015-01-02 to 2026-04-06 (~2,830 trading days, ~11 years)
- **Frequency:** Daily (US market trading days only)
- **Prices:** Adjusted for splits and dividends (`auto_adjust=True` from yfinance)
- **Units:** USD (prices), shares (volume)
- **Missing values:** XLC has ~31% missing (launched 2018); XLRE has ~7% missing (launched 2015-Q4); all other tickers have <0.04% missing (1–2 isolated days, likely data gaps)
- **VIX:** Volatility index — not a tradeable price; represents implied 30-day volatility of S&P 500 options (values typically 10–80)

### Suggested Analyses

- Log-return time series: `log_returns = np.log(close / close.shift(1)).dropna()`
- Correlation/covariance structure across assets and sectors
- Factor models (PCA, market beta, sector exposures)
- Volatility regimes and clustering
- Cross-sectional return predictability
- Portfolio optimization (mean-variance, risk parity)
- Sector rotation dynamics
