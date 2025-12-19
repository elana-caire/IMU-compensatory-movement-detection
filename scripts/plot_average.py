import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import config.paths as paths_config

models_summary_path_task_agnostic = paths_config.MODELS_PATH / 'model_results_LOSO_task_agnostic.csv'
models_summary_path_task_specific = paths_config.MODELS_PATH / 'model_results_LOSO_task_specific.csv'

PLOT_PATH = paths_config.PLOTS_PATH

METRICS = [
    "f1_macro_mean",
    "accuracy_mean",
    "precision_macro_mean",
    "recall_macro_mean",
    "roc_auc_mean"
]
 
def plot_average_results(
    CSV_PATH: str,
    SAVE_FOLDER:  str
):
    """Plot average results across tasks from a CSV file.

    Args:
        CSV_PATH (str): Path to the CSV file containing model results.
        SAVE_FOLDER (str): Folder to save the generated plots.
    """
    df = pd.read_csv(CSV_PATH)
    df["window_size"] = pd.to_numeric(df["window_size"], errors="coerce")
    agg_list = {}
    for m in METRICS:
        agg_list[m] = "mean"     # average metric value
        agg_list[m.replace("_mean", "_sd")] = (m, "std")   # standard deviation across tasks
    rows = []
    for (model, window), group in df.groupby(["model_name", "window_size"]):
        row = {
            "model_name": model,
            "window_size": window
        }
        for metric in METRICS:
            values = group[metric].values
            row[metric] = values.mean()
            row[metric.replace("_mean", "_sd")] = values.std()
        rows.append(row)

    df_avg = pd.DataFrame(rows)
    df_avg = df_avg.sort_values("window_size")
    return df_avg


def plot_metric(metric_name, df_avg, save_folder):
    plt.figure(figsize=(10, 6))

    for model in df_avg["model_name"].unique():
        d = df_avg[df_avg["model_name"] == model]

        plt.errorbar(
            d["window_size"],
            d[metric_name],
            yerr=d[metric_name.replace("_mean", "_sd")],
            marker="o",
            capsize=5,
            label=model
        )

    plt.xlabel("Window Size")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"Average {metric_name.replace('_', ' ').title()} Across Tasks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        Path(save_folder) / f"average_{metric_name}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


avg_loso_task_agnostic = plot_average_results(CSV_PATH=models_summary_path_task_agnostic, SAVE_FOLDER=PLOT_PATH)
for metric in METRICS:
    plot_metric(metric, avg_loso_task_agnostic, SAVE_FOLDER=PLOT_PATH)

avg_loso_task_specific = plot_average_results(CSV_PATH=models_summary_path_task_specific, SAVE_FOLDER=PLOT_PATH)
for metric in METRICS:
    plot_metric(metric, avg_loso_task_specific, SAVE_FOLDER=PLOT_PATH)  

print(f"✔ Saved plots in: {PLOT_PATH}")
