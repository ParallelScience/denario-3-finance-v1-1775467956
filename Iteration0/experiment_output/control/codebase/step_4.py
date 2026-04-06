# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

def main():
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_data_step1.csv'), index_col=0, parse_dates=True)
    features_df = pd.read_csv(os.path.join(data_dir, 'engineered_features_step3.csv'), index_col=0, parse_dates=True)
    results_list = []
    for k in [5, 21]:
        y = np.log(df['^VIX_raw'].shift(-k) / df['^VIX_raw'])
        X = pd.DataFrame(index=df.index)
        X['SVD_t'] = features_df['SVD_normalized']
        X['VIX_log_return_t'] = features_df['VIX_log_return']
        X['MarketReturn_t'] = df['^GSPC']
        X['SVD_x_AvgCorr_t'] = features_df['SVD_normalized'] * features_df['Avg_Sector_Corr_21d']
        X = sm.add_constant(X)
        reg_data = pd.concat([y.rename('Target'), X], axis=1).dropna()
        model = sm.OLS(reg_data['Target'], reg_data[['const', 'SVD_t', 'VIX_log_return_t', 'MarketReturn_t', 'SVD_x_AvgCorr_t']])
        result = model.fit(cov_type='HAC', cov_kwds={'maxlags': k})
        print('\n--- OLS Regression Summary for k=' + str(k) + ' ---')
        print(result.summary())
        res_df = pd.DataFrame({'Coefficient': result.params, 'Std_Error': result.bse, 't_stat': result.tvalues, 'p_value': result.pvalues})
        res_df['k'] = k
        res_df['R_squared'] = result.rsquared
        results_list.append(res_df)
    all_ols_results = pd.concat(results_list)
    all_ols_results.to_csv(os.path.join(data_dir, 'ols_regression_results.csv'))
    print('\nSaved OLS regression results to: ' + os.path.join(data_dir, 'ols_regression_results.csv'))
    target = (features_df['VIX_regime'] == 'High').astype(int)
    universe = features_df['VIX_regime'].shift(1) != 'High'
    X_log = features_df[['SVD_t-1']].copy()
    X_log = sm.add_constant(X_log)
    log_data = pd.concat([target.rename('Target'), X_log, universe.rename('Universe')], axis=1).dropna()
    log_data = log_data[log_data['Universe']]
    log_model = sm.Logit(log_data['Target'], log_data[['const', 'SVD_t-1']])
    log_result = log_model.fit(disp=0)
    print('\n--- Logistic Regression Summary (Transition to High Regime) ---')
    print(log_result.summary())
    preds = log_result.predict(log_data[['const', 'SVD_t-1']])
    base_auroc = roc_auc_score(log_data['Target'], preds)
    print('\nBase Logistic Regression AUROC: ' + str(round(base_auroc, 4)))
    log_res_df = pd.DataFrame({'Coefficient': log_result.params, 'Std_Error': log_result.bse, 'z_stat': log_result.tvalues, 'p_value': log_result.pvalues})
    log_res_df.to_csv(os.path.join(data_dir, 'logistic_regression_results.csv'))
    print('Saved Logistic regression results to: ' + os.path.join(data_dir, 'logistic_regression_results.csv'))
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    auroc_scores = []
    for exclude_etf in sector_etfs:
        current_etfs = [etf for etf in sector_etfs if etf != exclude_etf]
        rolling_vol = df[current_etfs].rolling(window=21).std() * np.sqrt(252)
        svd_raw = rolling_vol.std(axis=1)
        svd_rolling_mean = svd_raw.rolling(window=252).mean()
        svd_rolling_std = svd_raw.rolling(window=252).std()
        svd_normalized = (svd_raw - svd_rolling_mean) / svd_rolling_std
        svd_t_minus_1 = svd_normalized.shift(1)
        X_cv = pd.DataFrame({'const': 1, 'SVD_t-1': svd_t_minus_1})
        cv_data = pd.concat([target.rename('Target'), X_cv, universe.rename('Universe')], axis=1).dropna()
        cv_data = cv_data[cv_data['Universe']]
        try:
            model_cv = sm.Logit(cv_data['Target'], cv_data[['const', 'SVD_t-1']])
            res_cv = model_cv.fit(disp=0)
            preds_cv = res_cv.predict(cv_data[['const', 'SVD_t-1']])
            score = roc_auc_score(cv_data['Target'], preds_cv)
            auroc_scores.append(score)
        except Exception as e:
            print('Error fitting model excluding ' + exclude_etf + ': ' + str(e))
            auroc_scores.append(np.nan)
    auroc_scores = np.array(auroc_scores)
    valid_scores = auroc_scores[~np.isnan(auroc_scores)]
    print('\n--- Leave-One-Out (Sector ETF) Cross-Validation AUROC ---')
    print('Mean: ' + str(round(valid_scores.mean(), 4)))
    print('Std:  ' + str(round(valid_scores.std(), 4)))
    print('Min:  ' + str(round(valid_scores.min(), 4)))
    print('Max:  ' + str(round(valid_scores.max(), 4)))
    metrics_df = pd.DataFrame({'Excluded_ETF': sector_etfs, 'AUROC': auroc_scores})
    metrics_df.to_csv(os.path.join(data_dir, 'loocv_auroc_metrics.csv'), index=False)
    print('Saved LOOCV AUROC metrics to: ' + os.path.join(data_dir, 'loocv_auroc_metrics.csv'))

if __name__ == '__main__':
    main()