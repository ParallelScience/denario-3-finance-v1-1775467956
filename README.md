# US Equity & Macro Finance Dataset (2015–2026)

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-04-06
**Status:** Project initialized

## Dataset

Daily adjusted OHLCV data for 66 US tickers: 5 major indices, 11 sector ETFs, and 50 large-cap stocks across tech, finance, healthcare, energy, consumer, industrial, and telecom sectors. Date range: 2015-01-02 to 2026-04-06 (~2,830 trading days).

## Progress

| Step | Iteration 0 |
|------|------------|
| Setup | done |
| Idea | |
| Methods | |
| Results | |
| Evaluate | |
| Paper | |

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
