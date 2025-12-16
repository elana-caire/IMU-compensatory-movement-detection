import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import traceback
from collections import Counter

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

# optional shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# user config (expects config.py in same folder)
from config import CONFIG

# -------------------- Defaults and reproducibility --------------------
RANDOM_STATE = CONFIG.get('random_state', 42)
np.random.seed(RANDOM_STATE)

# Permutation importance & SHAP defaults in CONFIG (if not provided)
CONFIG.setdefault('compute_permutation_importance', False)
CONFIG.setdefault('pi_n_repeats', 5)
CONFIG.setdefault('models_requiring_permutation_importance', ['SVM', 'KNN', 'MLP'])
CONFIG.setdefault('compute_shap', True)
CONFIG.setdefault('shap_possible_models', ['RF', 'XGBoost', 'LightGBM', 'LASSO_LR'])
CONFIG.setdefault('shap_sample_size', 500)
CONFIG.setdefault('allow_kernel_shap_for_other_models', False)

# output folders (refactored suffixes)
OUT_BASE = Path(CONFIG.get('output_folder', './results_task_specific'))
PLOTS_DIR = OUT_BASE / (Path(CONFIG.get('plots_folder', 'plots')).name + '_task_specific')
IMP_DIR = OUT_BASE / (Path(CONFIG.get('importance_folder', 'importances')).name + '_task_specific')
SHAP_DIR = OUT_BASE / (Path(CONFIG.get('shap_folder', 'shap')).name + '_task_specific')

for d in [OUT_BASE, PLOTS_DIR, IMP_DIR, SHAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------ Helpers ------------------

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
        'LASSO_LR': LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=5000, random_state=RANDOM_STATE)
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


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 95) -> Tuple[float, float, float]:
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
    """
    Inner CV over training subjects to select hyperparameters.
    param_grid_input can be dict or list-of-dicts (like sklearn).
    Returns best_params (dict).
    """
    unique_groups = np.unique(groups)
    combos = param_grid_to_list(param_grid_input)
    if len(unique_groups) == 1 or len(combos) == 0:
        return combos[0] if combos else {}

    best_score = -np.inf
    best_params = None

    max_evals = CONFIG.get('max_inner_evals', 100)
    grid = combos
    if CONFIG.get('use_random_search', False) and len(grid) > max_evals:
        rng = np.random.default_rng(RANDOM_STATE)
        grid = list(rng.choice(grid, size=max_evals, replace=False))

    for params in grid:
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
            X_train, y_train = sk_shuffle(X_train, y_train, random_state=RANDOM_STATE)

            model = create_model(model_name, params)
            try:
                model.fit(X_train, y_train)
            except Exception:
                fold_scores.append(0.0)
                continue

            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='macro')
            fold_scores.append(score)

        if len(fold_scores) == 0:
            continue
        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params if best_params is not None else (combos[0] if combos else {})


def evaluate_task_window(file_path: Path, window_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info(f'Loading features from {file_path} (window={window_label})')
    df = pd.read_csv(file_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df, label_encoder = encode_labels(df, CONFIG['target'])

    features = [c for c in df.columns if c not in ['subject', 'task', CONFIG['target'], 'Label']]
    subjects = df['subject'].unique()
    tasks = df['task'].unique()

    results_records = []
    imp_records = []
    shap_records = []

    for task in tasks:
        logger.info(f'Evaluating Task: {task} (window: {window_label})')
        df_task = df[df['task'] == task].reset_index(drop=True)
        if df_task.empty:
            continue

        groups = df_task['subject'].values
        unique_subjects = np.unique(groups)

        for model_name in CONFIG['models_to_use']:
            logger.info(f'  Model: {model_name}')

            per_subject_metrics = {s: {} for s in unique_subjects}
            fold_importances = []
            best_params_across_folds = []

            # New: collect SHAP per fold (mean-abs per feature)
            fold_shap_meanabs_list = []

            for test_subject in unique_subjects:
                train_mask = df_task['subject'] != test_subject
                test_mask = df_task['subject'] == test_subject

                X_train = df_task.loc[train_mask, features]
                y_train = df_task.loc[train_mask, 'Label']
                X_test = df_task.loc[test_mask, features]
                y_test = df_task.loc[test_mask, 'Label']

                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f'Empty train/test for subject {test_subject}, skipping fold')
                    continue

                train_groups = df_task.loc[train_mask, 'subject'].values
                param_grid_input = CONFIG['param_grids'].get(model_name, {})
                best_params = inner_tune_params(df_task.loc[train_mask].reset_index(drop=True),
                                               features, train_groups, model_name, param_grid_input)
                # store for later logging
                best_params_across_folds.append(best_params if isinstance(best_params, dict) else {})

                scaler = StandardScaler()
                X_train_arr = scaler.fit_transform(X_train.values)
                X_test_arr = scaler.transform(X_test.values)

                X_train_arr, y_train = sk_shuffle(X_train_arr, y_train.values, random_state=RANDOM_STATE)

                model = create_model(model_name, best_params)
                model.fit(X_train_arr, y_train)

                # Predictions for evaluation
                try:
                    y_pred = model.predict(X_test_arr)
                except Exception:
                    # fallback to most frequent label if predict fails
                    y_pred = np.array([np.bincount(y_train).argmax()] * X_test_arr.shape[0])

                try:
                    y_proba = model.predict_proba(X_test_arr) if hasattr(model, 'predict_proba') else None
                except Exception:
                    y_proba = None

                metrics = compute_metrics(y_test.values, y_pred, y_proba)
                per_subject_metrics[test_subject] = metrics

                # -------- Feature importance logic (native -> permutation if needed) --------
                imp = np.zeros(len(features))

                # Prefer native attribute if available
                try:
                    if hasattr(model, 'feature_importances_'):
                        imp_attr = getattr(model, 'feature_importances_', None)
                        if imp_attr is not None:
                            imp = np.array(imp_attr)
                    elif hasattr(model, 'coef_') and model_name in ['LASSO_LR']:  # linear models
                        coef = getattr(model, 'coef_', None)
                        if coef is not None:
                            # multiclass logistic: coef_ shape (n_classes, n_features)
                            imp = np.mean(np.abs(coef), axis=0)
                    else:
                        # Permutation importance only if explicitly enabled and model in list
                        if (CONFIG.get('compute_permutation_importance', False)
                                and model_name in CONFIG.get('models_requiring_permutation_importance', [])):
                            try:
                                r = permutation_importance(
                                    model,
                                    X_test_arr,
                                    y_test.values,
                                    n_repeats=CONFIG.get('pi_n_repeats', 5),
                                    random_state=RANDOM_STATE,
                                    n_jobs=1
                                )
                                imp = r.importances_mean
                            except Exception as e:
                                logger.warning(f'Permutation importance failed on fold (subject={test_subject}, model={model_name}): {e}')
                                imp = np.zeros(len(features))
                except Exception as e:
                    logger.warning(f'Error extracting feature importance for model {model_name} (subject={test_subject}): {e}')
                    imp = np.zeros(len(features))

                fold_importances.append(imp)

                # -------------------- FOLD-WISE SHAP computation (NEW) --------------------
                if SHAP_AVAILABLE and CONFIG.get('compute_shap', True) and model_name in CONFIG.get('shap_possible_models', []):
                    try:
                        # Use a background/sample from training fold only to build explainer (no leakage)
                        sample_size = min(CONFIG.get('shap_sample_size', 500), X_train_arr.shape[0])
                        if sample_size <= 0:
                            raise ValueError('shap_sample_size <= 0')

                        # Subsample training for background/explainer if large
                        if sample_size < X_train_arr.shape[0]:
                            rng = np.random.default_rng(RANDOM_STATE)
                            bg_idx = rng.choice(np.arange(X_train_arr.shape[0]), size=sample_size, replace=False)
                            X_bg = X_train_arr[bg_idx]
                        else:
                            X_bg = X_train_arr

                        # compute SHAP values on the validation (test_subject) fold
                        # optionally subsample validation if it's huge (rare in LOSO)
                        test_sample_size = min(CONFIG.get('shap_sample_size', 500), X_test_arr.shape[0])
                        if test_sample_size < X_test_arr.shape[0]:
                            rng = np.random.default_rng(RANDOM_STATE)
                            test_idx = rng.choice(np.arange(X_test_arr.shape[0]), size=test_sample_size, replace=False)
                            X_test_shap = X_test_arr[test_idx]
                        else:
                            X_test_shap = X_test_arr

                        # Choose explainer type carefully
                        explainer = None
                        shap_values = None
                        # Prefer model-specific explainers for speed/accuracy
                        if model_name in ['XGBoost', 'LightGBM', 'RF']:
                            # TreeExplainer accepts the model directly; background used by TreeExplainer internally if needed
                            explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
                            shap_out = explainer.shap_values(X_test_shap)
                        elif model_name == 'LASSO_LR':
                            # Linear explainer on linear models
                            explainer = shap.LinearExplainer(model, X_bg, feature_perturbation='interventional')
                            shap_out = explainer.shap_values(X_test_shap)
                        else:
                            # fallback to general Explainer which may pick a reasonable explainer
                            try:
                                explainer = shap.Explainer(model, X_bg)
                                shap_out = explainer(X_test_shap).values
                            except Exception:
                                # optionally allow KernelExplainer for other models if configured (very slow)
                                if CONFIG.get('allow_kernel_shap_for_other_models', False):
                                    explainer = shap.KernelExplainer(model.predict_proba, X_bg)
                                    shap_out = explainer.shap_values(X_test_shap)
                                else:
                                    shap_out = None

                        if shap_out is not None:
                            # Normalise shap_out to numpy array with shape (n_samples, n_features)
                            if isinstance(shap_out, list):
                                # multiclass: list of arrays (n_classes, n_samples, n_features) or list of (n_samples, n_features)
                                # shap library often returns list with each element (n_samples, n_features)
                                try:
                                    # stack to (n_classes, n_samples, n_features) then average abs across classes
                                    arrs = [np.array(a) for a in shap_out]
                                    arr_stack = np.stack(arrs, axis=-1)  # (n_samples, n_features, n_classes)
                                    # convert to (n_samples, n_features) by averaging absolute values across classes
                                    shap_arr = np.mean(np.abs(arr_stack), axis=-1)
                                except Exception:
                                    # fallback: try converting each to abs mean then average
                                    shap_arr = np.mean([np.mean(np.abs(a), axis=0) for a in shap_out], axis=0)
                                    # shap_arr shape -> (n_features,) ; convert to (n_samples, n_features) by repeating
                                    if shap_arr.ndim == 1:
                                        shap_arr = np.tile(shap_arr, (X_test_shap.shape[0], 1))
                            else:
                                shap_arr = np.array(shap_out)  # (n_samples, n_features)

                            # ensure we have 2D array
                            if shap_arr.ndim == 1:
                                shap_arr = shap_arr.reshape(-1, shap_arr.shape[0])

                            # mean absolute per feature for this fold
                            shap_mean_abs_fold = np.mean(np.abs(shap_arr), axis=0)  # shape: (n_features,)
                            # If shap_mean_abs_fold length mismatches features, try to handle gracefully
                            if shap_mean_abs_fold.shape[0] != len(features):
                                # attempt transpose or flatten
                                try:
                                    shap_mean_abs_fold = shap_mean_abs_fold.flatten()[:len(features)]
                                except Exception:
                                    shap_mean_abs_fold = np.zeros(len(features))

                            fold_shap_meanabs_list.append(shap_mean_abs_fold)

                    except Exception as e:
                        logger.error(
                            f'Fold-wise SHAP failed for (task={task}, model={model_name}, test_subject={test_subject}, win={window_label}).'
                            f'\nError: {e}\nTraceback:\n{traceback.format_exc()}'
                        )
                        # do not raise — continue with other folds

            # After outer folds, aggregate metrics and importances
            metric_names = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']
            metric_summary = {}
            for m in metric_names:
                vals = [per_subject_metrics[s].get(m, np.nan) for s in unique_subjects]
                vals_clean = [v for v in vals if (not pd.isna(v))]
                mean, lower, upper = bootstrap_ci(vals_clean, n_bootstrap=CONFIG.get('n_bootstrap', 1000), ci=CONFIG.get('ci', 95))
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
                best_params_repr = first_param_from_grid(CONFIG['param_grids'].get(model_name, {}))

            record = {
                'window_size': window_label,
                'task': task,
                'model_name': model_name,
                'best_params': str(best_params_repr)
            }
            record.update(metric_summary)
            results_records.append(record)

            # aggregate feature importances across outer folds (mean)
            if len(fold_importances) > 0:
                try:
                    importances = np.mean(np.vstack(fold_importances), axis=0)
                except Exception:
                    importances = np.zeros(len(features))
            else:
                importances = np.zeros(len(features))

            for feat, imp in zip(features, importances):
                imp_records.append({'feature': feat, 'importance': float(imp), 'task': task, 'model': model_name, 'window_size': window_label})

            # -------------------- AGGREGATE FOLD-WISE SHAP VALUES (NEW) --------------------
            if len(fold_shap_meanabs_list) > 0:
                try:
                    # stack per-fold meanabs -> shape (n_folds, n_features)
                    shap_stack = np.vstack(fold_shap_meanabs_list)
                    shap_mean_across_folds = np.mean(shap_stack, axis=0)  # mean across folds
                except Exception:
                    shap_mean_across_folds = np.zeros(len(features))
            else:
                shap_mean_across_folds = np.zeros(len(features))

            # append aggregated SHAP record entries
            for feat, s in zip(features, shap_mean_across_folds):
                shap_records.append({'feature': feat, 'shap_mean_abs': float(s), 'task': task, 'model': model_name, 'window_size': window_label})

    results_df = pd.DataFrame(results_records)
    importances_df = pd.DataFrame(imp_records)
    shap_df = pd.DataFrame(shap_records)
    return results_df, importances_df, shap_df


# ------------------ Plotting utilities ------------------

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')


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


def plot_feature_importance_heatmap(df_imp: pd.DataFrame, task: str, save_folder: Path, top_n: int = 30):
    df_task = df_imp[df_imp['task'] == task]
    df_top = df_task.sort_values('importance', ascending=False).groupby(['model', 'window_size']).head(top_n)
    pivot = df_top.pivot_table(index='feature', columns=['model', 'window_size'], values='importance', fill_value=0)
    if pivot.shape[0] == 0 or pivot.shape[1] == 0:
        return
    pivot['mean_imp'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_imp', ascending=False).drop(columns=['mean_imp'])
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.6), min(40, pivot.shape[0] * 0.25)))
    sns.heatmap(pivot, cmap='viridis')
    plt.title(f'Feature Importance Heatmap – Task: {task}')
    plt.tight_layout()
    plt.savefig(save_folder / f'feature_importance_heatmap_{task}.png', dpi=300)
    plt.close()


def plot_shap_heatmap(df_shap: pd.DataFrame, task: str, save_folder: Path, top_n: int = 30):
    df_task = df_shap[df_shap['task'] == task]
    df_top = df_task.sort_values('shap_mean_abs', ascending=False).groupby(['model', 'window_size']).head(top_n)
    pivot = df_top.pivot_table(index='feature', columns=['model', 'window_size'], values='shap_mean_abs', fill_value=0)
    if pivot.shape[0] == 0 or pivot.shape[1] == 0:
        return
    pivot['mean_shap'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_shap', ascending=False).drop(columns=['mean_shap'])
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.6), min(40, pivot.shape[0] * 0.25)))
    sns.heatmap(pivot, cmap='coolwarm')
    plt.title(f'SHAP Importance Heatmap – Task: {task}')
    plt.tight_layout()
    plt.savefig(save_folder / f'shap_heatmap_{task}.png', dpi=300)
    plt.close()


# ------------------ Main Orchestration ------------------

def main():
    feature_files = find_feature_files(CONFIG['features_folder'])
    all_results = []
    all_feature_importances = []
    all_shap_importances = []

    for file_path, window_size in tqdm(feature_files, desc='Processing windows'):
        res_df, imp_df, shap_df = evaluate_task_window(file_path, window_size)
        if not res_df.empty:
            all_results.append(res_df)
        if not imp_df.empty:
            all_feature_importances.append(imp_df)
        if not shap_df.empty:
            all_shap_importances.append(shap_df)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(PLOTS_DIR / 'model_results_summary_refactored.csv', index=False)
    else:
        results_df = pd.DataFrame()

    if all_feature_importances:
        df_imp_all = pd.concat(all_feature_importances, ignore_index=True)
        df_imp_all.to_csv(IMP_DIR / 'feature_importances_all_refactored.csv', index=False)
    else:
        df_imp_all = pd.DataFrame()

    if all_shap_importances and SHAP_AVAILABLE:
        df_shap_all = pd.concat(all_shap_importances, ignore_index=True)
        df_shap_all.to_csv(SHAP_DIR / 'shap_importances_all_refactored.csv', index=False)
    else:
        df_shap_all = pd.DataFrame()

    # plotting
    if not results_df.empty:
        for task in results_df['task'].unique():
            plot_metrics_across_windows(results_df, task, PLOTS_DIR)

    if not df_imp_all.empty:
        for task in df_imp_all['task'].unique():
            plot_feature_importance_heatmap(df_imp_all, task, IMP_DIR, top_n=CONFIG.get('top_n_features', 30))

    if not df_shap_all.empty:
        for task in df_shap_all['task'].unique():
            plot_shap_heatmap(df_shap_all, task, SHAP_DIR, top_n=CONFIG.get('top_n_features', 30))

    logger.info('Done. Results saved in: %s', OUT_BASE)


if __name__ == '__main__':
    main()
