# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def construct_svd():
    input_path = 'data/cleaned_data_step1.csv'
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    rolling_vol = df[sector_etfs].rolling(window=21).std() * np.sqrt(252)
    svd_raw = rolling_vol.std(axis=1)
    svd_rolling_mean = svd_raw.rolling(window=252).mean()
    svd_rolling_std = svd_raw.rolling(window=252).std()
    svd_normalized = (svd_raw - svd_rolling_mean) / svd_rolling_std
    svd_df = pd.DataFrame({'SVD_raw': svd_raw, 'SVD_normalized': svd_normalized})
    print('Descriptive Statistics for Raw SVD:')
    print('Mean: ' + str(svd_df['SVD_raw'].mean()))
    print('Std:  ' + str(svd_df['SVD_raw'].std()))
    print('Min:  ' + str(svd_df['SVD_raw'].min()))
    print('Max:  ' + str(svd_df['SVD_raw'].max()))
    print('\nDescriptive Statistics for Normalized SVD:')
    print('Mean: ' + str(svd_df['SVD_normalized'].mean()))
    print('Std:  ' + str(svd_df['SVD_normalized'].std()))
    print('Min:  ' + str(svd_df['SVD_normalized'].min()))
    print('Max:  ' + str(svd_df['SVD_normalized'].max()))
    svd_output_path = 'data/svd_series.csv'
    vol_output_path = 'data/rolling_realized_volatility.csv'
    svd_df.to_csv(svd_output_path)
    rolling_vol.to_csv(vol_output_path)
    print('\nSaved SVD series to: ' + svd_output_path)
    print('Saved rolling realized volatility to: ' + vol_output_path)

if __name__ == '__main__':
    construct_svd()