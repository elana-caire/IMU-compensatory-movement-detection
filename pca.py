from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Paths
features_folder = "/Users/sorresoayarey/Desktop/relab/Data/Features"
plots_folder = Path("/Users/sorresoayarey/Desktop/relab/Data/Plots/PCA")
plots_folder.mkdir(parents=True, exist_ok=True)

# --- Utility functions ---
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

def plot_taskwise_pca(df, target_col="condition", facet_col="task", subject_col="subject",
                       save_path=None, title=None):
    """
    Compute PCA separately for each task and plot facets.
    Each facet uses only data from that task.
    Points colored by `condition`, shaped by `subject`.
    """
    tasks = df[facet_col].unique()
    n_tasks = len(tasks)

    # Determine subplot layout
    n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        df_task = df[df[facet_col] == task]
        features = [col for col in df_task.columns if col not in [facet_col, target_col, subject_col, "label"]]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_task[features].values)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plot_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        plot_df[target_col] = df_task[target_col].values
        plot_df[subject_col] = df_task[subject_col].values

        ax = axes[i]
        # Plot without legend for now
        sns.scatterplot(data=plot_df, x="PC1", y="PC2",
                        hue=target_col, style=subject_col, s=60, alpha=0.8, ax=ax, legend=False)
        ax.set_title(f"Task: {task}")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)

    # Create custom legend
    # Get unique values for conditions and subjects
    unique_conditions = df[target_col].unique()
    unique_subjects = df[subject_col].unique()
    
    # Create proxy artists for legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = []
    
    # Add condition colors
    # Get color palette
    palette = sns.color_palette("husl", len(unique_conditions))
    for condition, color in zip(unique_conditions, palette):
        legend_elements.append(Patch(facecolor=color, edgecolor=color, label=f"{target_col}: {condition}"))
    
    # Add subject shapes
    # Get marker styles
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    for subject, marker in zip(unique_subjects, markers):
        legend_elements.append(Line2D([0], [0], marker=marker, color='gray', 
                                      markerfacecolor='gray', markersize=8, 
                                      label=f"{subject_col}: {subject}", linestyle=''))
    
    # Create legend
    fig.legend(handles=legend_elements, 
               loc='upper right', 
               bbox_to_anchor=(1.12, 1), 
               title="Legend")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --- Main loop ---
features_files = find_feature_files(features_folder)

for feature_file, win_name in features_files:
    df = load_feature_data(feature_file)
    save_path = plots_folder / f"PCA_taskwise_win_{win_name}.png"
    plot_taskwise_pca(df, target_col="condition", facet_col="task", subject_col="subject",
                      save_path=save_path, title=f"PCA per task, window: {win_name}")
