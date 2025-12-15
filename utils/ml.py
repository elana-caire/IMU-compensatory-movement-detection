#!/usr/bin/env python3
import os
import argparse

import hashlib  
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils import shuffle as sk_shuffle
from sklearn.inspection import permutation_importance
import shap


# SHAP might be heavy; import here and handle if missing
try:
    import shap
    SHAP_AVAILABLE = True
    from shap.explainers import _permutation


except Exception as e:
    print("shap not available:", e)
    SHAP_AVAILABLE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def encode_condition_labels(df):
    """
    Utility function to encoder labels for conditions
    """

    df.loc[:,'Label'] = -1

    mask = df['condition'] == 'natural'
    df.loc[mask, 'Label'] = 0
    mask = df['condition'] == 'elbow_brace'
    df.loc[mask, 'Label'] = 1
    mask = df['condition'] == 'elbow_wrist_brace'
    df.loc[mask, 'Label'] = 2
    
    return df


def create_temporal_train_test_folds(df):

    unique_subjects   = df["subject"].unique()
    unique_tasks      = df["task"].unique()
    unique_conditions = df["condition"].unique()

    K = 5  # 5 folds = 20% each

    fold_splits = []  # list of (fold_idx, df_train, df_test)

    for fold_idx in range(K):
        df_train_parts = []
        df_test_parts  = []

        for subject in unique_subjects:
            df_subj = df[df["subject"] == subject]

            for task in unique_tasks:
                for condition in unique_conditions:
                    df_subset = df_subj[(df_subj["task"] == task) & (df_subj["condition"] == condition)]
                    if df_subset.empty:
                        continue

                    # IMPORTANT: keep temporal order
                    # If you have a time column, sort by it here:
                    # df_subset = df_subset.sort_values("time_col").reset_index(drop=True)
                    df_subset = df_subset.reset_index(drop=True)

                    n = len(df_subset)
                    fold_size = int(np.floor(n / K))
                    start = fold_idx * fold_size

                    # last fold takes the remainder
                    end = (start + fold_size) if fold_idx < K - 1 else n

                    test_idx = np.arange(start, end)
                    train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])

                    df_test_parts.append(df_subset.iloc[test_idx])
                    df_train_parts.append(df_subset.iloc[train_idx])

        df_train_fold = pd.concat(df_train_parts, axis=0, ignore_index=True)
        df_test_fold  = pd.concat(df_test_parts,  axis=0, ignore_index=True)

        fold_splits.append((fold_idx, df_train_fold, df_test_fold))

        print(f"Fold {fold_idx}: train={len(df_train_fold)}, test={len(df_test_fold)}")
    return fold_splits

def return_feature_columns(df, sensors_to_consider, 
                           time_features:list, 
                           frequency_features:list, 
                           exclude_quat = False, 
                           exclude_acc=False, 
                           exclude_gyro = False, 
                           exclude_mag = False):
    import itertools
    feat_columns = []
    for sensor in sensors_to_consider:
        if time_features is not None:
            for time_feat in time_features:
                quaternion_columns = (
                    df.columns[df.columns.str.contains("Quat") & df.columns.str.contains(sensor) & df.columns.str.contains(time_feat)]
                    if not exclude_quat else []
                )
                acc_columns = (
                    df.columns[df.columns.str.contains("Acc") & df.columns.str.contains(sensor) & df.columns.str.contains(time_feat) ]
                    if not exclude_acc else []
                )
                gyr_columns = (
                    df.columns[df.columns.str.contains("Gyr") & df.columns.str.contains(sensor) & df.columns.str.contains(time_feat) ]
                    if not exclude_gyro else []
                )
                mag_columns = (
                    df.columns[df.columns.str.contains("Mag")& df.columns.str.contains(sensor) & df.columns.str.contains(time_feat)]
                    if not exclude_mag else []
                )
                feat_columns.append(list(quaternion_columns) + list(acc_columns) + list(gyr_columns) + list(mag_columns))
        if frequency_features is not None:
            for freq_feat in frequency_features:
                quaternion_columns = (
                    df.columns[df.columns.str.contains("Quat") & df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat)]
                    if not exclude_quat else []
                )
                acc_columns = (
                    df.columns[df.columns.str.contains("Acc") & df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat) ]
                    if not exclude_acc else []
                )
                gyr_columns = (
                    df.columns[df.columns.str.contains("Gyr") & df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat) ]
                    if not exclude_gyro else []
                )
                mag_columns = (
                    df.columns[df.columns.str.contains("Mag")& df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat)]
                    if not exclude_mag else []
                )
                feat_columns.append(list(quaternion_columns) + list(acc_columns) + list(gyr_columns) + list(mag_columns))
        


    return list(itertools.chain.from_iterable(feat_columns))


def ensure_features_only(df, feature_cols):
    """
    Return X (numpy) and feature column list verifying presence.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    X = df[feature_cols].to_numpy(dtype=float)
    return X


# ------------------------ HELPERS -----------------------------

def find_feature_files(folder):
    folder = Path(folder)
    files = []
    for f in folder.glob("*.csv"):
        win = "nowin"
        if f.name.startswith("features_win_"):
            win = f.name.replace("features_win_", "").replace(".csv", "")
        files.append((f, win))
    return sorted(files, key=lambda x: x[1])


def encode_labels(df, target_col):
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df[target_col])
    return df, le


def create_model(name, params=None):
    base_models = {
        "RF": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
        "KNN": KNeighborsClassifier(),
        "MLP": MLPClassifier(random_state=RANDOM_STATE, max_iter=100),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
        "LASSO_LR": LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=1000, random_state=RANDOM_STATE)
    }
    model = base_models[name]
    if params:
        model.set_params(**params)
    return model


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(pd.get_dummies(y_true), y_proba, multi_class='ovr', average='macro')
        except Exception:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    return metrics


def bootstrap_ci(scores, n_bootstrap=500, ci=95):
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)
    mean = np.mean(scores)
    return mean, lower, upper


def plot_metrics_across_windows(results_df, task, save_path):
    df_task = results_df[results_df['task'] == task]
    metrics = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']

    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in df_task['model_name'].unique():
            model_data = df_task[df_task['model_name'] == model]
            ax.errorbar(model_data['window_size'], model_data[f'{metric}_mean'],
                        yerr=[model_data[f'{metric}_mean'] - model_data[f'{metric}_lower'],
                              model_data[f'{metric}_upper'] - model_data[f'{metric}_mean']],
                        marker='o', capsize=5, label=model)
        ax.set_xlabel("Window size")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Model Performance Across Window Sizes - {task}", fontsize=16)
    plt.tight_layout()
    plt.savefig(Path(save_path)/f"metrics_across_windows_{task}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_heatmap(df_imp, task, save_folder, top_n):
    df_task = df_imp[df_imp['task'] == task]
    # Aggregate by model and window_size: keep top_n per (model, window)
    df_top = df_task.sort_values("importance", ascending=False).groupby(["model", "window_size"]).head(top_n)
    # Pivot with (model, window) columns
    pivot = df_top.pivot_table(index="feature", columns=["model", "window_size"], values="importance", fill_value=0)

    # Re-order rows by overall mean importance so top features float to top
    pivot['mean_imp'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_imp', ascending=False).drop(columns=['mean_imp'])

    plt.figure(figsize=(max(10, pivot.shape[1]*0.6), min(40, pivot.shape[0]*0.25)))
    sns.heatmap(pivot, cmap="viridis")
    plt.title(f"Feature Importance Heatmap – Task: {task}")
    plt.tight_layout()
    plt.savefig(Path(save_folder) / f"feature_importance_heatmap_{task}.png", dpi=300)
    plt.close()


def plot_shap_heatmap(df_shap, task, save_folder, top_n):
    df_task = df_shap[df_shap['task'] == task]
    # For SHAP, importance is mean absolute SHAP
    df_top = df_task.sort_values("shap_mean_abs", ascending=False).groupby(["model", "window_size"]).head(top_n)
    pivot = df_top.pivot_table(index="feature", columns=["model", "window_size"], values="shap_mean_abs", fill_value=0)
    pivot['mean_shap'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_shap', ascending=False).drop(columns=['mean_shap'])

    plt.figure(figsize=(max(10, pivot.shape[1]*0.6), min(40, pivot.shape[0]*0.25)))
    sns.heatmap(pivot, cmap="coolwarm")
    plt.title(f"SHAP Importance Heatmap – Task: {task}")
    plt.tight_layout()
    plt.savefig(Path(save_folder) / f"shap_heatmap_{task}.png", dpi=300)
    plt.close()


def umap_projection(df_task, feature_columns, task, label_encoder):
    import umap.umap_ as umap

    # Features matrix and labels
    X = df_task[feature_columns].values
    y = df_task["Label"].values  # numeric labels from encode_labels

    # UMAP reducer
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    emb = reducer.fit_transform(X)  # shape: (n_samples, 2)

    # Optional: get human-readable class names, if label_encoder is available
    unique_labels = np.unique(y)
    try:
        label_names = label_encoder.inverse_transform(unique_labels)
    except Exception:
        # fallback: just use numeric labels
        label_names = [str(l) for l in unique_labels]

    # Plot UMAP projection
    plt.figure(figsize=(6, 5))
    for lab, name in zip(unique_labels, label_names):
        idx = y == lab
        plt.scatter(
            emb[idx, 0],
            emb[idx, 1],
            s=10,
            alpha=0.7,
            label=name,
        )

    plt.title(f"UMAP projection – task: {task}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()



def balanced_subsample_after_scaling(X_train_scaled, y_train, n_total=100, random_state=0):
    rng = np.random.default_rng(random_state)

    y = np.asarray(y_train)
    idx_all = np.arange(len(y))
    classes = np.unique(y)
    k = len(classes)

    base = n_total // k
    rem  = n_total % k

    chosen = []
    for i, c in enumerate(classes):
        idx_c = idx_all[y == c]
        n_c = base + (1 if i < rem else 0)

        # sample with replacement if not enough examples in class
        replace = len(idx_c) < n_c
        chosen_c = rng.choice(idx_c, size=n_c, replace=replace)
        chosen.append(chosen_c)

    chosen_idx = np.concatenate(chosen)
    rng.shuffle(chosen_idx)

    X_sub = X_train_scaled[chosen_idx]
    y_sub = y[chosen_idx]
    return X_sub, y_sub



def shap_to_3d(shap_values):
    """
    Return SHAP values as (n_samples, n_features, n_classes).
    Handles:
      - array (n, f, c)
      - array (n, f) -> assumes single output -> (n, f, 1)
      - list of arrays (per class) each (n, f) -> (n, f, c)
    """
    if isinstance(shap_values, list):
        sv = np.stack(shap_values, axis=-1)  # (n, f, c)
        return sv
    sv = np.asarray(shap_values)
    if sv.ndim == 2:
        return sv[:, :, None]
    if sv.ndim == 3:
        return sv
    raise ValueError(f"Unexpected SHAP shape/type: {type(shap_values)}, shape={getattr(sv,'shape',None)}")

def agg_stats(series: pd.Series):
    """Return mean/std + median/IQR (more robust to outliers)."""
    s = series.dropna().astype(float)
    if len(s) == 0:
        return pd.Series({"mean": np.nan, "std": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan})
    return pd.Series({
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
    })

def params_to_id(params: dict) -> str:  # <-- NEW
    """Stable, filesystem-safe id for a param dict."""
    s = json.dumps(params, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
    return f"p_{h}"


def upsert_rows(df_old: pd.DataFrame, df_new: pd.DataFrame, key_cols: list) -> pd.DataFrame:  # <-- NEW
    """Append and drop duplicates by key (keep last)."""
    if df_old is None or df_old.empty:
        return df_new.copy()
    out = pd.concat([df_old, df_new], ignore_index=True)
    out = out.drop_duplicates(subset=key_cols, keep="last")
    return out


def parse_params_cell(v):
    """Robust parse for dict-ish strings in CSV."""
    if isinstance(v, dict):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    s = str(v).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.encode("utf-8").decode("unicode_escape").strip()
    if not s.lstrip().startswith("{"):
        s = "{" + s.strip().strip(",") + "}"
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        d = ast.literal_eval(s)
    if not isinstance(d, dict):
        raise ValueError(f"Parsed value is not a dict: {type(d)}")
    return d
