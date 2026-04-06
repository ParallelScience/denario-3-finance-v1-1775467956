1. **Data Preprocessing and Harmonization**
   - Load the `close_prices.csv` dataset.
   - Filter for the 10 stable sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLB) and the `^VIX` index.
   - Calculate daily log-returns for the 10 selected sector ETFs.
   - Handle missing values by dropping rows where any of the 10 sector ETFs have missing data to ensure a balanced panel.

2. **Construction of Sectoral Volatility Dispersion (SVD)**
   - Calculate the 21-day rolling realized volatility for each sector ETF, annualizing the result by multiplying the standard deviation of log-returns by $\sqrt{252}$.
   - Compute the Cross-Sectional Standard Deviation (CSSD) of these 10 annualized realized volatility series for each day $t$ to define the SVD metric.
   - Apply Z-score normalization to the SVD series using a long-term rolling window to ensure stationarity and coefficient stability.

3. **VIX Regime Definition**
   - Transform the `^VIX` series into log-returns ($\ln(VIX_t / VIX_{t-1})$) to ensure stationarity.
   - Implement a Hidden Markov Model (HMM) with three states (Low, Moderate, High) to endogenously classify market regimes based on the `^VIX` series.
   - Use an expanding window approach for HMM parameter estimation to avoid look-ahead bias, ensuring regime labels at time $t$ are based only on information available up to that point.

4. **Stationarity and Feature Engineering**
   - Perform Augmented Dickey-Fuller (ADF) tests on the normalized SVD series.
   - Generate lagged features of the SVD ($SVD_{t-1}, SVD_{t-5}, SVD_{t-21}$) to capture the temporal lead-lag relationship.
   - Calculate the 21-day rolling average pairwise correlation of the 10 sector ETFs to serve as a control variable for market decoupling.

5. **Correlation and Decoupling Analysis**
   - Quantify the relationship between SVD and the average sector correlation.
   - Define "market decoupling" as periods of high SVD and low average correlation.
   - Prepare an interaction term ($SVD_t \times \text{AvgCorrelation}_t$) to test if the predictive power of SVD is conditional on the state of market integration.

6. **Predictive Modeling of VIX Innovations**
   - Estimate the regression: $\Delta \ln(VIX)_{t+k} = \alpha + \beta_1 SVD_t + \beta_2 \Delta \ln(VIX)_t + \beta_3 \text{MarketReturn}_t + \beta_4 (SVD_t \times \text{AvgCorrelation}_t) + \epsilon_t$.
   - Set lead horizons $k$ to 5 and 21 days to align with the SVD lookback window.
   - Use Newey-West standard errors to account for heteroskedasticity and autocorrelation.

7. **Regime Transition Probability Modeling**
   - Use a logistic regression model to predict the probability of transitioning into a "High" volatility regime (as identified by the HMM) using lagged SVD as the primary predictor.
   - Evaluate model performance using AUROC and Precision-Recall curves.
   - Conduct sensitivity analysis by varying the SVD lookback window (10, 21, 63 days) to optimize the detection horizon.

8. **Robustness Testing and PCA Comparison**
   - Perform "leave-one-out" cross-validation by systematically removing one sector ETF at a time to ensure the SVD signal is not driven by a single outlier.
   - Compare the SVD-based signal against the "explained variance ratio" of the first principal component (the market factor) derived from the same 10 sector ETFs.
   - Confirm that SVD provides unique predictive information regarding idiosyncratic dispersion that is not captured by the market factor.