# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
from hmmlearn import hmm
from statsmodels.tsa.stattools import adfuller
import os

def main():
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_data_step1.csv'), index_col=0, parse_dates=True)
    svd_df = pd.read_csv(os.path.join(data_dir, 'svd_series.csv'), index_col=0, parse_dates=True)
    vix_log_returns = np.log(df['^VIX_raw'] / df['^VIX_raw'].shift(1)).dropna()
    model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=1000, random_state=42)
    vix_log_returns_reshaped = vix_log_returns.values.reshape(-1, 1)
    model.fit(vix_log_returns_reshaped)
    hidden_states = model.predict(vix_log_returns_reshaped)
    vix_raw_aligned = df['^VIX_raw'].loc[vix_log_returns.index]
    state_means = {}
    for i in range(3):
        state_means[i] = vix_raw_aligned[hidden_states == i].mean()
    sorted_states = sorted(state_means, key=state_means.get)
    state_map = {sorted_states[0]: 'Low', sorted_states[1]: 'Moderate', sorted_states[2]: 'High'}
    mapped_states = [state_map[state] for state in hidden_states]
    print('HMM State Mapping (Mean VIX Level):')
    for state in sorted_states:
        print('State ' + str(state) + ' (' + state_map[state] + '): ' + str(state_means[state]))
    svd_raw_clean = svd_df['SVD_raw'].dropna()
    svd_norm_clean = svd_df['SVD_normalized'].dropna()
    adf_raw = adfuller(svd_raw_clean)
    adf_norm = adfuller(svd_norm_clean)
    print('\nADF Test on Raw SVD:')
    print('ADF Statistic: ' + str(adf_raw[0]))
    print('p-value: ' + str(adf_raw[1]))
    print('Lags Used: ' + str(adf_raw[2]))
    print('\nADF Test on Normalized SVD:')
    print('ADF Statistic: ' + str(adf_norm[0]))
    print('p-value: ' + str(adf_norm[1]))
    print('Lags Used: ' + str(adf_norm[2]))
    svd_df['SVD_t-1'] = svd_df['SVD_normalized'].shift(1)
    svd_df['SVD_t-5'] = svd_df['SVD_normalized'].shift(5)
    svd_df['SVD_t-21'] = svd_df['SVD_normalized'].shift(21)
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    returns_df = df[sector_etfs]
    pairs = [(sector_etfs[i], sector_etfs[j]) for i in range(len(sector_etfs)) for j in range(i + 1, len(sector_etfs))]
    corr_series = []
    for (a, b) in pairs:
        corr_series.append(returns_df[a].rolling(window=21).corr(returns_df[b]))
    avg_corr = pd.concat(corr_series, axis=1).mean(axis=1)
    features_df = pd.DataFrame(index=df.index)
    features_df['VIX_log_return'] = vix_log_returns
    features_df['VIX_regime'] = pd.Series(mapped_states, index=vix_log_returns.index)
    features_df['SVD_raw'] = svd_df['SVD_raw']
    features_df['SVD_normalized'] = svd_df['SVD_normalized']
    features_df['SVD_t-1'] = svd_df['SVD_t-1']
    features_df['SVD_t-5'] = svd_df['SVD_t-5']
    features_df['SVD_t-21'] = svd_df['SVD_t-21']
    features_df['Avg_Sector_Corr_21d'] = avg_corr
    output_path = os.path.join(data_dir, 'engineered_features_step3.csv')
    features_df.to_csv(output_path)
    print('\nSaved engineered features to: ' + output_path)

if __name__ == '__main__':
    main()