
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import median_absolute_error, mean_squared_error
from lifelines.utils import concordance_index

# === Plot Training and Validation Loss ===
def plot_loss(history, current_path):
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Model Loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(current_path / "mse_validation_loss.png", bbox_inches="tight")
    plt.show()


# === Plot Log-Scaled Loss ===
def plot_log_loss(history, current_path):
    plt.figure(figsize=(8.5, 8))
    plt.style.use("classic")
    plt.plot(np.log(history.history['loss']), label='train')
    plt.plot(np.log(history.history['val_loss']), label='val')
    plt.title('Model Loss (Log MSE)')
    plt.ylabel('Log Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(current_path / "mse_validation_loss-log.png", bbox_inches="tight")
    plt.show()


# === Aggregate predictions and labels per WSI ===
def aggregate_value(y_pred, test_label, Image_ID):
    value_to_indices = {}
    for i, value in enumerate(Image_ID):
        value_to_indices.setdefault(value, []).append(i)

    # Compute per-WSI averages
    n_samples = len(value_to_indices)
    grouped_array_label = np.empty((n_samples, 1))
    grouped_array_pred = np.empty((n_samples, 1))

    for i, value in enumerate(value_to_indices.keys()):
        indices = value_to_indices[value]
        grouped_array_label[i] = np.mean(test_label[indices])
        grouped_array_pred[i] = np.mean(y_pred[indices])

    # Save for future use
    np.save("y_pred_wsi.npy", grouped_array_pred)
    np.save("test_label_wsi.npy", grouped_array_label)

    return grouped_array_pred, grouped_array_label


# === Calculate correlation, FDR-corrected p-values, MAE ===
def calculate_correlation(y_pred_wsi, test_label_wsi):
    # Pearson correlation
    r, p = stats.pearsonr(y_pred_wsi[:, 0], test_label_wsi[:, 0])

    # MAE
    mae = median_absolute_error(test_label_wsi[:, 0], y_pred_wsi[:, 0])
    mean_age = np.mean(test_label_wsi[:, 0])

    # Compile results into DataFrames
    corr_df = pd.DataFrame({"raw_corr": [r], "raw_p": [p]})
    mae_df = pd.DataFrame({"mae": [mae], "mean_age": [mean_age]})

    # Multiple testing correction (trivial here since only 1 gene)
    fdr_p = multipletests(corr_df["raw_p"], alpha=0.05, method='hs')[1]
    corr_df["multi"] = fdr_p
    corr_df["corr_multi_005"] = corr_df["raw_corr"].where(corr_df["multi"] < 0.05)

    # Save
    corr_df.to_excel("aging_prediction_corr_005_genes.xlsx")
    mae_df.to_excel("aging_prediction_mae_genes.xlsx")

    return corr_df["corr_multi_005"], fdr_p, mae


# === Plot correlation distribution ===
def plot_corr_distribution(data, current_path):
    data = data[~np.isnan(data)]
    fig = plt.figure(figsize=(2, 6))
    plt.style.use("fast")
    plt.boxplot(data)
    plt.ylim(0, 1.0)
    plt.tick_params(labelsize=14)
    plt.savefig(current_path / "correlation-distribution.png", bbox_inches="tight")
    plt.show()


# === Plot prediction vs. label ===
def plot_scatter(y_pred_wsi, test_label_wsi, mae):
    y_true = test_label_wsi.reshape(-1)
    y_pred = y_pred_wsi.reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.6)
    plt.xlabel('True Age')
    plt.ylabel('Predicted Age')
    plt.title('WSI-Level Age Prediction')
    plt.grid(True)

    # Add statistics
    corr, p = stats.pearsonr(y_pred, y_true)
    plt.text(0.03, 0.85, f'Pearson r: {corr:.2f}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.03, 0.8, f'P-value: {p:.2e}', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.03, 0.7, f'Median AE: {mae:.2f}', fontsize=12, transform=plt.gca().transAxes)

    plt.savefig('scatter_plot_WSI.png')
    plt.show()


# === Run all plots and metrics ===
def all_plots(history, y_pred, test_label, Image_ID_test, current_path):
    plot_loss(history, current_path)
    plot_log_loss(history, current_path)

    # Aggregate per WSI
    y_pred_wsi, test_label_wsi = aggregate_value(y_pred, test_label, Image_ID_test)

    # Correlation + MAE
    data, multi, mae = calculate_correlation(y_pred_wsi, test_label_wsi)
    plot_corr_distribution(data, current_path)

    # Concordance Index
    c_index = concordance_index(test_label_wsi, y_pred_wsi)
    print("C-index:", c_index)

    # Scatter plot
    plot_scatter(y_pred_wsi, test_label_wsi, mae)

    return None
