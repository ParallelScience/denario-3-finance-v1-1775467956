# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import statsmodels.api as sm
from datetime import datetime

def main():
    mpl.rcParams['text.usetex'] = False
    data_dir = 'data/'
    df = pd.read_csv(os.path.join(data_dir, 'cleaned_data_step1.csv'), index_col=0, parse_dates=True)
    features_df = pd.read_csv(os.path.join(data_dir, 'engineered_features_step3.csv'), index_col=0, parse_dates=True)
    loocv_df = pd.read_csv(os.path.join(data_dir, 'loocv_auroc_metrics.csv'))
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    vix_raw = df['^VIX_raw'].loc[features_df.index]
    regimes = features_df['VIX_regime']
    axes[0].plot(vix_raw.index, vix_raw, color='black', linewidth=1.5, label='VIX Level')
    regime_colors = {'Low': 'green', 'Moderate': 'yellow', 'High': 'red'}
    regime_num = regimes.map({'Low': 0, 'Moderate': 1, 'High': 2})
    changes = regime_num.diff().fillna(0) != 0
    change_indices = changes[changes].index.tolist()
    if regimes.index[0] not in change_indices:
        change_indices.insert(0, regimes.index[0])
    if regimes.index[-1] not in change_indices:
        change_indices.append(regimes.index[-1])
    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i+1]
        regime_val = regimes.loc[start_idx]
        axes[0].axvspan(start_idx, end_idx, color=regime_colors.get(regime_val, 'white'), alpha=0.3, lw=0)
    legend_elements = [Patch(facecolor='green', alpha=0.3, label='Low Regime'), Patch(facecolor='yellow', alpha=0.3, label='Moderate Regime'), Patch(facecolor='red', alpha=0.3, label='High Regime'), plt.Line2D([0], [0], color='black', lw=1.5, label='VIX Level')]
    axes[0].legend(handles=legend_elements, loc='upper left')
    axes[0].set_title('VIX Levels and Volatility Regimes')
    axes[0].set_ylabel('VIX (Points)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(features_df.index, features_df['SVD_normalized'], color='blue', linewidth=1.5)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title('Sectoral Volatility Dispersion (SVD) - Normalized')
    axes[1].set_ylabel('SVD (Z-score)')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(features_df.index, features_df['Avg_Sector_Corr_21d'], color='purple', linewidth=1.5)
    axes[2].set_title('21-Day Rolling Average Sector Pairwise Correlation')
    axes[2].set_ylabel('Correlation')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    multi_panel_path = os.path.join(data_dir, 'vix_svd_correlation_multipanel_1_' + timestamp + '.png')
    plt.savefig(multi_panel_path, dpi=300)
    plt.close()
    print('Multi-panel figure saved to ' + multi_panel_path)
    target = (features_df['VIX_regime'] == 'High').astype(int)
    universe = features_df['VIX_regime'].shift(1) != 'High'
    X_log = features_df[['SVD_t-1']].copy()
    X_log = sm.add_constant(X_log)
    log_data = pd.concat([target.rename('Target'), X_log, universe.rename('Universe')], axis=1).dropna()
    log_data = log_data[log_data['Universe']]
    log_model = sm.Logit(log_data['Target'], log_data[['const', 'SVD_t-1']])
    log_result = log_model.fit(disp=0)
    preds = log_result.predict(log_data[['const', 'SVD_t-1']])
    fpr, tpr, _ = roc_curve(log_data['Target'], preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(log_data['Target'], preds)
    pr_auc = average_precision_score(log_data['Target'], preds)
    print('Base Model AUROC: ' + str(round(roc_auc, 4)))
    print('Base Model PR AUC: ' + str(round(pr_auc, 4)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    label_roc = 'ROC curve (area = ' + str(round(roc_auc, 3)) + ')'
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=label_roc)
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    label_pr = 'PR curve (area = ' + str(round(pr_auc, 3)) + ')'
    axes[1].plot(recall, precision, color='green', lw=2, label=label_pr)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    perf_plot_path = os.path.join(data_dir, 'model_performance_curves_2_' + timestamp + '.png')
    plt.savefig(perf_plot_path, dpi=300)
    plt.close()
    print('Model performance curves saved to ' + perf_plot_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(loocv_df['Excluded_ETF'], loocv_df['AUROC'], color='skyblue', edgecolor='black')
    label_base = 'Base Model AUROC (' + str(round(roc_auc, 3)) + ')'
    ax.axhline(roc_auc, color='red', linestyle='--', label=label_base)
    min_auroc = loocv_df['AUROC'].min()
    max_auroc = loocv_df['AUROC'].max()
    ax.set_ylim([min_auroc - 0.02, max_auroc + 0.02])
    ax.set_ylabel('AUROC Score')
    ax.set_xlabel('Excluded Sector ETF')
    ax.set_title('Leave-One-Out Cross-Validation AUROC Scores')
    ax.legend()
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, str(round(yval, 3)), ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    loocv_plot_path = os.path.join(data_dir, 'loocv_auroc_barchart_3_' + timestamp + '.png')
    plt.savefig(loocv_plot_path, dpi=300)
    plt.close()
    print('LOOCV AUROC bar chart saved to ' + loocv_plot_path)

if __name__ == '__main__':
    main()