import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import traceback
from collections import Counter

import time
from contextlib import contextmanager


import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle as sk_shuffle

# classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import config.training_common as training_config
import config.loso as loso_config
import config.paths as paths_config

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = training_config.RANDOM_STATE
PLOTS_DIR = paths_config.PLOTS_PATH
MODELS_DIR = paths_config.MODELS_PATH
FEATURES_DIR = paths_config.FEATURES_PATH
RESULTS_DIR = paths_config.RESULTS_PATH

TARGET = training_config.TARGET
MODELS = training_config.MODELS_TO_USE

PARAMS_GRID = training_config.PARAM_GRIDS
N_BOOTSTRAP_SAMPLES = loso_config.N_BOOTSTRAP_SAMPLES
CI = loso_config.CI
USE_RANDOM_SEARCH = loso_config.USE_RANDOM_SEARCH
MAX_INNER_EVALS = loso_config.MAX_INNER_EVALS


# ------------------ Helpers ------------------

@contextmanager
def timed_block(name: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"[TIMER] {name} took {elapsed:.2f}s")


def find_feature_files(folder: str) -> List[Tuple[Path, str]]:
    folder = Path(folder)
    files = []
    for f in folder.glob("*.csv"):
        win = 'nowin'
        if f.name.startswith('features_win_'):
            win = f.name.replace('features_win_', '').replace('.csv', '')
        files.append((f, win))
    try:
        files_sorted = sorted(files, key=lambda x: float(x[1]) if x[1] not in ['nowin'] else -1)
    except Exception:
        files_sorted = sorted(files, key=lambda x: x[1])
    return files_sorted


def encode_labels(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df = df.copy()
    df['Label'] = le.fit_transform(df[target_col].astype(str))
    return df, le


def create_model(name: str, params: Optional[Dict[str, Any]] = None):
    base_models = {
        'RF': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': SVC(random_state=RANDOM_STATE, probability=True),
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
        'LightGBM': LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
        'LASSO_LR': LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=500, random_state=RANDOM_STATE)
    }
    if name not in base_models:
        raise ValueError(f'Model {name} not implemented')
    model = base_models[name]
    if params:
        try:
            model.set_params(**params)
        except Exception as e:
            logger.warning(f'Failed to set params for {name}: {e} -- using defaults')
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(pd.get_dummies(y_true), y_proba, multi_class='ovr', average='macro')
        except Exception:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    return metrics


def bootstrap_ci(values: List[float], n_bootstrap: int = 100, ci: float = 95) -> Tuple[float, float, float]:
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    arr = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    mean = np.mean(arr)
    return mean, lower, upper


# ------------------ Parameter grid helpers ------------------

def param_grid_to_list(param_grid_input):
    """
    Accepts either:
      - dict (e.g. {'C': [0.1,1], 'kernel': ['rbf']})
      - list of dicts (e.g. [{'C':[...], ...}, {...}])
    Returns a list of param dicts (i.e. all combinations).
    """
    try:
        combos = list(ParameterGrid(param_grid_input))
        return combos
    except Exception:
        # fallback: if param_grid_input already is iterable of dicts, try to coerce
        if isinstance(param_grid_input, list):
            return param_grid_input
        return [{}]


def first_param_from_grid(param_grid_input):
    combos = param_grid_to_list(param_grid_input)
    return combos[0] if len(combos) > 0 else {}


# ------------------ Nested CV / Evaluation ------------------

def inner_tune_params(df_task: pd.DataFrame, features: List[str], groups: np.ndarray,
                      model_name: str, param_grid_input) -> Dict[str, Any]:

    unique_groups = np.unique(groups)
    combos = param_grid_to_list(param_grid_input)

    if len(unique_groups) == 1 or len(combos) == 0:
        return combos[0] if combos else {}

    if USE_RANDOM_SEARCH and len(combos) > MAX_INNER_EVALS:
        rng = np.random.default_rng(RANDOM_STATE)
        combos = list(rng.choice(combos, size=MAX_INNER_EVALS, replace=False))

    best_score = -np.inf
    best_params = None

    with timed_block(f"Inner CV ({model_name})"):
        for params in tqdm(
            combos,
            desc=f"Param search [{model_name}]",
            leave=False
        ):
            fold_scores = []

            for val_group in unique_groups:
                train_mask = groups != val_group
                val_mask = groups == val_group

                X_train = df_task.loc[train_mask, features].values
                y_train = df_task.loc[train_mask, 'Label'].values
                X_val = df_task.loc[val_mask, features].values
                y_val = df_task.loc[val_mask, 'Label'].values

                if len(np.unique(y_train)) < 2 or len(X_val) == 0:
                    continue

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                model = create_model(model_name, params)
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro')
                except Exception:
                    score = 0.0

                fold_scores.append(score)

            if fold_scores:
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

    return best_params if best_params is not None else combos[0]


def evaluate_task_window(file_path: Path, window_label: str):

    logger.info(f'Loading features from {file_path} (window={window_label})')
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df, label_encoder = encode_labels(df, TARGET)

    features = [c for c in df.columns if c not in ['subject', 'task', TARGET, 'Label']]
    subjects = df['subject'].unique()
    tasks = df['task'].unique()

    results_records = []

    df_all = df.reset_index(drop=True)  # ALL tasks pooled

    for task in tqdm(tasks, desc=f"Tasks (win={window_label})"):
        df_test_task = df_all[df_all['task'] == task].reset_index(drop=True)
        if df_test_task.empty:
            continue

        unique_subjects = df_test_task['subject'].unique()

        for model_name in tqdm(MODELS, desc=f"Models [{task}]", leave=False):

            with timed_block(f"Model {model_name} | Task {task}"):

                per_subject_metrics = {}
                best_params_across_folds = []

                for test_subject in tqdm(
                    unique_subjects,
                    desc=f"LOSO [{model_name}]",
                    leave=False
                ):
                    train_mask = df_all['subject'] != test_subject
                    test_mask = (df_all['subject'] == test_subject) & (df_all['task'] == task)

                    X_train = df_all.loc[train_mask, features]
                    y_train = df_all.loc[train_mask, 'Label']
                    X_test  = df_all.loc[test_mask, features]
                    y_test  = df_all.loc[test_mask, 'Label']

                    if len(X_train) == 0 or len(X_test) == 0:
                        continue

                    train_groups = df_all.loc[train_mask, 'subject'].values

                    best_params = inner_tune_params(
                        df_all.loc[train_mask].reset_index(drop=True),
                        features,
                        train_groups,
                        model_name,
                        PARAMS_GRID.get(model_name, {})
                    )

                    best_params_across_folds.append(best_params)

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    model = create_model(model_name, best_params)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                    per_subject_metrics[test_subject] = compute_metrics(
                        y_test.values, y_pred, y_proba
                    )

            # After outer folds, aggregate metrics and importances
            metric_names = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']
            metric_summary = {}
            for m in metric_names:
                vals = [per_subject_metrics[s].get(m, np.nan) for s in unique_subjects]
                vals_clean = [v for v in vals if (not pd.isna(v))]
                mean, lower, upper = bootstrap_ci(vals_clean, n_bootstrap=N_BOOTSTRAP_SAMPLES, ci=CI)
                metric_summary[f'{m}_mean'] = mean
                metric_summary[f'{m}_lower'] = lower
                metric_summary[f'{m}_upper'] = upper

            # Determine a representative best_params for logging:
            def dict_key(d):
                try:
                    return json.dumps(d, sort_keys=True)
                except Exception:
                    return str(d)

            best_key_counts = Counter(dict_key(bp) for bp in best_params_across_folds if isinstance(bp, dict))
            if len(best_key_counts) > 0:
                best_key_most = best_key_counts.most_common(1)[0][0]
                try:
                    best_params_repr = json.loads(best_key_most)
                except Exception:
                    best_params_repr = {}
            else:
                best_params_repr = first_param_from_grid(PARAMS_GRID.get(model_name, {}))

            record = {
                'window_size': window_label,
                'task': task,
                'model_name': model_name,
                'best_params': str(best_params_repr)
            }
            record.update(metric_summary)
            results_records.append(record)

    results_df = pd.DataFrame(results_records)
    
    return results_df


# ------------------ Plotting utilities ------------------

def plot_metrics_across_windows(results_df: pd.DataFrame, task: str, save_path: Path):
    df_task = results_df[results_df['task'] == task]
    metrics = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in df_task['model_name'].unique():
            model_data = df_task[df_task['model_name'] == model]
            ax.errorbar(model_data['window_size'].astype(str), model_data[f'{metric}_mean'],
                        yerr=[model_data[f'{metric}_mean'] - model_data[f'{metric}_lower'],
                              model_data[f'{metric}_upper'] - model_data[f'{metric}_mean']],
                        marker='o', capsize=5, label=model)
        ax.set_xlabel('Window size')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'Model Performance Across Window Sizes - {task}', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path / f'metrics_across_windows_{task}.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------ Main Orchestration ------------------

def main():
    feature_files = find_feature_files(FEATURES_DIR)
    all_results = []
    for file_path, window_size in tqdm(feature_files, desc='Processing windows'):
        res_df = evaluate_task_window(file_path, window_size)
        if not res_df.empty:
            all_results.append(res_df)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(MODELS_DIR / 'model_results_LOSO_task_agnostic.csv', index=False)
    else:
        results_df = pd.DataFrame()

    # plotting
    if not results_df.empty:
        for task in results_df['task'].unique():
            plot_metrics_across_windows(results_df, task, PLOTS_DIR)

    logger.info('Done. Results saved in: %s', RESULTS_DIR)


if __name__ == '__main__':
    main()
