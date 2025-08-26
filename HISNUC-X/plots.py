
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error
)


def plot_loss(history, current_path):
    """
    Plot training and validation MSE loss over epochs.
    """
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(current_path / "mse_validation_loss.png", bbox_inches="tight")
    plt.show()


def plot_log_loss(history, current_path):
    """
    Plot training and validation log-MSE loss over epochs.
    """
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(np.log(history.history['loss']))
    plt.plot(np.log(history.history['val_loss']))
    plt.title('Model Log Loss')
    plt.ylabel('Log(MSE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(current_path / "mse_validation_loss-log.png", bbox_inches="tight")
    plt.show()


def aggregate_value(y_pred, test_label, image_ids, current_path):
    """
    Aggregate predictions and labels per image by computing median across patches.
    """
    value_to_indices = {}
    for i, val in enumerate(image_ids):
        value_to_indices.setdefault(val, []).append(i)

    grouped_pred = np.empty((len(value_to_indices), 100))
    grouped_label = np.empty((len(value_to_indices), 100))

    for i, val in enumerate(value_to_indices.keys()):
        idxs = value_to_indices[val]
        grouped_pred[i] = np.median(y_pred[idxs], axis=0)
        grouped_label[i] = np.median(test_label[idxs], axis=0)

    return grouped_pred, grouped_label


def calculate_correlation(y_pred_wsi, test_label_wsi):
    """
    Compute Pearson correlation and multiple metrics for each gene.
    Save metrics to an Excel file.
    """
    n_genes = y_pred_wsi.shape[1]
    corr_df, pval_df = np.zeros(n_genes), np.zeros(n_genes)
    corr_df_raw, pval_df_raw = np.zeros(n_genes), np.zeros(n_genes)

    metrics_log = pd.DataFrame(columns=["MSE_log", "MAE_log", "MEAN_log", "Med_AE_log", "Median_log"])
    metrics_raw = pd.DataFrame(columns=["MSE_raw", "MAE_raw", "MEAN_raw", "Med_AE_raw", "Median_raw"])

    for i in range(n_genes):
        # Log-space metrics
        corr, p = stats.pearsonr(y_pred_wsi[:, i], test_label_wsi[:, i])
        corr_df[i], pval_df[i] = corr, p

        metrics_log.loc[i] = [
            mean_squared_error(test_label_wsi[:, i], y_pred_wsi[:, i]),
            mean_absolute_error(test_label_wsi[:, i], y_pred_wsi[:, i]),
            np.mean(test_label_wsi[:, i]),
            median_absolute_error(test_label_wsi[:, i], y_pred_wsi[:, i]),
            np.median(test_label_wsi[:, i]),
        ]

        # Raw TPM space
        y_true_raw = np.exp(test_label_wsi[:, i]) - 1
        y_pred_raw = np.exp(y_pred_wsi[:, i]) - 1

        corr_raw, p_raw = stats.pearsonr(y_pred_raw, y_true_raw)
        corr_df_raw[i], pval_df_raw[i] = corr_raw, p_raw

        metrics_raw.loc[i] = [
            mean_squared_error(y_true_raw, y_pred_raw),
            mean_absolute_error(y_true_raw, y_pred_raw),
            np.mean(y_true_raw),
            median_absolute_error(y_true_raw, y_pred_raw),
            np.median(y_true_raw),
        ]

    # Multiple testing correction
    fdr_log = multipletests(pval_df, alpha=0.05, method='hs')[1]
    fdr_raw = multipletests(pval_df_raw, alpha=0.05, method='hs')[1]

    final_corr_df = pd.DataFrame({
        "log_corr": corr_df,
        "log_p": pval_df,
        "log_FDR_p": fdr_log,
        "log_corr_FDR_005": np.where(fdr_log < 0.05, corr_df, np.nan),
        "raw_corr": corr_df_raw,
        "raw_p": pval_df_raw,
        "raw_FDR_p": fdr_raw,
        "raw_corr_FDR_005": np.where(fdr_raw < 0.05, corr_df_raw, np.nan)
    })

    # Save to Excel
    with pd.ExcelWriter("expression_prediction_metrics.xlsx") as writer:
        final_corr_df.to_excel(writer, sheet_name="Correlation_Log_and_Raw")
        metrics_log.to_excel(writer, sheet_name="Log_Space_Metrics")
        metrics_raw.to_excel(writer, sheet_name="Raw_TPM_Metrics")

    return final_corr_df["log_corr_FDR_005"], final_corr_df["raw_corr_FDR_005"]


def plot_corr_distribution(data, current_path):
    """
    Plot the distribution of significant correlation values.
    """
    data = data[~np.isnan(data)]
    data = data[data > 0]
    
    plt.figure(figsize=(2, 6))
    plt.style.use("fast")
    plt.boxplot(data)
    plt.ylim(0, 1.0)
    plt.tick_params(labelsize=14)
    plt.savefig(current_path / "correlation-distribution.png", bbox_inches="tight")
    plt.show()


def all_plots(history, y_pred, test_label, image_ids, current_path):
    """
    Generate all plots and metrics in one function.
    """
    plot_loss(history, current_path)
    plot_log_loss(history, current_path)

    y_pred_wsi, test_label_wsi = aggregate_value(y_pred, test_label, image_ids, current_path)
    log_corr_sig, _ = calculate_correlation(y_pred_wsi, test_label_wsi)
    plot_corr_distribution(log_corr_sig, current_path)

    np.save("y_pred_wsi.npy", y_pred_wsi)
    np.save("test_label_wsi.npy", test_label_wsi)

    return y_pred_wsi, test_label_wsi
