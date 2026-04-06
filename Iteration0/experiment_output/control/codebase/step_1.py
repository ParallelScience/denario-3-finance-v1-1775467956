# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import os

def preprocess_data():
    file_path = '/home/node/work/data/finance/close_prices.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    vix_ticker = '^VIX'
    gspc_ticker = '^GSPC'
    selected_columns = sector_etfs + [vix_ticker, gspc_ticker]
    df_filtered = df[selected_columns].copy()
    df_filtered = df_filtered.dropna(subset=sector_etfs)
    log_returns = np.log(df_filtered[sector_etfs + [gspc_ticker]] / df_filtered[sector_etfs + [gspc_ticker]].shift(1))
    log_returns = log_returns.dropna(how='all')
    vix_levels = df_filtered[vix_ticker].loc[log_returns.index]
    combined_df = log_returns.copy()
    combined_df['^VIX_raw'] = vix_levels
    combined_df = combined_df.dropna()
    output_path = 'data/cleaned_data_step1.csv'
    combined_df.to_csv(output_path)
    print('Data Preprocessing and Harmonization Complete.')
    print('Shape of the cleaned dataset: ' + str(combined_df.shape))
    print('Saved cleaned data to: ' + output_path)

if __name__ == '__main__':
    preprocess_data()