from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.feature_selection import VarianceThreshold

# -----------------------------
# Paths and parameters
# -----------------------------
features_folder = "/Users/sorresoayarey/Desktop/relab/Data/Features"
plots_folder = Path("/Users/sorresoayarey/Desktop/relab/Data/Plots/Data_Exploration")
plots_folder.mkdir(parents=True, exist_ok=True)

target_col = "condition"
alpha = 0.05

# -----------------------------
# Utility functions
# -----------------------------
def find_feature_files(features_folder):
    folder_path = Path(features_folder)
    feature_files = []

    for file_path in folder_path.glob("*.csv"):
        filename = file_path.name
        if filename == "features.csv":
            feature_files.append((file_path, "nowin"))
        elif filename.startswith("features_win_") and filename.endswith(".csv"):
            win_name = filename.replace("features_win_", "").replace(".csv", "")
            feature_files.append((file_path, win_name))

    def sort_key(item):
        _, win_name = item
        if win_name == "nowin":
            return (0, "nowin")
        try:
            return (1, int(win_name))
        except ValueError:
            return (2, win_name)

    return sorted(feature_files, key=sort_key)

def load_feature_data(path):
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    print(f"Shape: {df.shape}")
    return df

def check_normality_statistical(X_scaled, feature_cols, alpha=0.05):
    normality_results = []
    for i, col in enumerate(feature_cols):
        # Shapiro-Wilk test (n <= 5000) else Kolmogorov-Smirnov
        if len(X_scaled[:, i]) <= 5000:
            stat, p_value = stats.shapiro(X_scaled[:, i])
        else:
            stat, p_value = stats.kstest(X_scaled[:, i], 'norm')

        k2_stat, k2_p = stats.normaltest(X_scaled[:, i])

        normality_results.append({
            'feature': col,
            'shapiro_p': p_value if len(X_scaled[:, i]) <= 5000 else np.nan,
            'dagostino_p': k2_p,
            'is_normal': (p_value > alpha) or (k2_p > alpha)
        })
    return pd.DataFrame(normality_results)

def plot_correlation_heatmap(X_scaled, feature_cols, task, win_name, save_folder):
    corr_matrix = pd.DataFrame(X_scaled, columns=feature_cols).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlation Heatmap - Task: {task} - Window: {win_name}')
    plt.tight_layout()
    plt.savefig(save_folder / f"correlation_heatmap_{task}_{win_name}.png")
    plt.close()

def plot_variance_distribution(plot_df, win_name, save_folder):
    g = sns.FacetGrid(plot_df, col="task", col_wrap=3, sharex=False, sharey=False, height=4)
    g.map_dataframe(sns.violinplot, x="class", y="variance", palette="Set2", inner="quartile")
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Class", "Feature Variance")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Feature Variance Distribution by Class - Window {win_name}")
    plt.savefig(save_folder / f"variance_distribution_{win_name}.png")
    plt.close()

def plot_normality_pvalues(window_normality_df, win_name, save_folder):
    plt.figure(figsize=(12, 6))
    sns.histplot(window_normality_df, x='shapiro_p', hue='task', bins=20, multiple='dodge', palette='Set2', alpha=0.7)
    plt.axvline(alpha, color='red', linestyle='--', label=f'alpha = {alpha}')
    plt.title(f'Shapiro-Wilk p-value Distribution - Window {win_name}')
    plt.xlabel('p-value')
    plt.ylabel('Number of Features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f"normality_pvalues_shapiro_{win_name}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(window_normality_df, x='dagostino_p', hue='task', bins=20, multiple='dodge', palette='Set2', alpha=0.7)
    plt.axvline(alpha, color='red', linestyle='--', label=f'alpha = {alpha}')
    plt.title(f"D'Agostino K² p-value Distribution - Window {win_name}")
    plt.xlabel('p-value')
    plt.ylabel('Number of Features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder / f"normality_pvalues_dagostino_{win_name}.png")
    plt.close()

def plot_class_distribution(df, target_col, win_name, save_folder):
    class_counts = df.groupby(['task', target_col]).size().reset_index(name='count')
    g2 = sns.catplot(
        data=class_counts,
        x=target_col, y='count',
        col='task', col_wrap=3,
        kind='bar', palette='Set3',
        sharey=False, height=4
    )
    g2.set_titles(col_template="{col_name}")
    g2.set_axis_labels("Class", "Count")
    plt.subplots_adjust(top=0.9)
    g2.fig.suptitle(f"Class Distribution per Task - Window {win_name}")
    plt.savefig(save_folder / f"class_distribution_{win_name}.png")
    plt.close()

# -----------------------------
# Main loop per window
# -----------------------------
features_files = find_feature_files(features_folder)

for feature_file, win_name in features_files:
    print(f"\nProcessing window: {win_name}")
    df = load_feature_data(feature_file)

    plot_data = []  # For variance plots
    window_normality_results = []

    for task in df['task'].unique():
        print(f"  Analyzing task: {task}")
        task_df = df[df['task'] == task]
        feature_cols = [col for col in task_df.columns if col not in ['subject', 'task', 'condition', 'label']]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(task_df[feature_cols])

        # Correlation heatmap per task
        plot_correlation_heatmap(X_scaled, feature_cols, task, win_name, plots_folder)

        # Normality tests per task
        normality_results = check_normality_statistical(X_scaled, feature_cols, alpha)
        normality_results['task'] = task
        window_normality_results.append(normality_results)

        # Variance per class
        for col in feature_cols:
            for cls in task_df[target_col].unique():
                cls_values = task_df[task_df[target_col] == cls][col].values
                var = np.var(cls_values)
                plot_data.append({
                    'task': task,
                    'feature': col,
                    'class': cls,
                    'variance': var
                })

    # -----------------------------
    # Plots per window (after all tasks)
    # -----------------------------
    plot_df = pd.DataFrame(plot_data)
    plot_variance_distribution(plot_df, win_name, plots_folder)

    window_normality_df = pd.concat(window_normality_results, ignore_index=True)
    window_normality_df.to_csv(plots_folder / f"normality_results_window_{win_name}.csv", index=False)
    plot_normality_pvalues(window_normality_df, win_name, plots_folder)

    plot_class_distribution(df, target_col, win_name, plots_folder)


# ccl :
# feature are not normal distributed
# feature show clusters, correlated features
# class are balanced
# variance of features are low