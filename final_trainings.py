import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.ml import *
from config import CONFIG
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------------------------- CONFIG ---------------------------
# create folders
Path(CONFIG["plots_folder"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["importance_folder"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["models_folder"]).mkdir(parents=True, exist_ok=True)

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
        "MLP": MLPClassifier(random_state=RANDOM_STATE, max_iter=500),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
        "LASSO_LR": LogisticRegression(
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            max_iter=5000,
            random_state=RANDOM_STATE
        )
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
            metrics['roc_auc'] = roc_auc_score(
                pd.get_dummies(y_true),
                y_proba,
                multi_class='ovr',
                average='macro'
            )
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
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    mean = np.mean(scores)
    return mean, lower, upper


def plot_metrics_across_windows(results_df, task, save_path):
    df_task = results_df[results_df['task'] == task]
    metrics = ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model in df_task['model_name'].unique():
            model_data = df_task[df_task['model_name'] == model]
            ax.errorbar(
                model_data['window_size'],
                model_data[f'{metric}_mean'],
                yerr=[
                    model_data[f'{metric}_mean'] - model_data[f'{metric}_lower'],
                    model_data[f'{metric}_upper'] - model_data[f'{metric}_mean']
                ],
                marker='o',
                capsize=5,
                label=model
            )
        ax.set_xlabel("Window size")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Model Performance Across Window Sizes - {task}", fontsize=16)
    plt.tight_layout()
    plt.savefig(Path(save_path) / f"metrics_across_windows_{task}.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_heatmap(df_imp, task, save_folder, top_n):
    df_task = df_imp[df_imp['task'] == task]
    df_top = (
        df_task.sort_values("importance", ascending=False)
        .groupby(["model", "window_size"])
        .head(top_n)
    )
    pivot = df_top.pivot_table(
        index="feature",
        columns=["model", "window_size"],
        values="importance",
        fill_value=0
    )

    pivot['mean_imp'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_imp', ascending=False).drop(columns=['mean_imp'])

    plt.figure(figsize=(max(10, pivot.shape[1] * 0.6),
                        min(40, pivot.shape[0] * 0.25)))
    sns.heatmap(pivot, cmap="viridis")
    plt.title(f"Feature Importance Heatmap – Task: {task}")
    plt.tight_layout()
    plt.savefig(Path(save_folder) / f"feature_importance_heatmap_{task}.png",
                dpi=300)
    plt.close()


# --------------------------- MAIN -----------------------------

window_to_process = "1000"  # specify the window size you want to process

def main():
    # ---------- Load data ----------
    df = pd.read_csv(CONFIG["features_folder"] + f"/features_win_{window_to_process}.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df, label_encoder = encode_labels(df, CONFIG["target"])

    # ---------- Manual temporal 80/20 split per subject–task–condition ----------
    unique_subjects = df['subject'].unique()
    unique_tasks = df['task'].unique()
    unique_conditions = df['condition'].unique()

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for subject in unique_subjects:
        df_subj = df[df['subject'] == subject]
        for task in unique_tasks:
            for condition in unique_conditions:
                df_subset = df_subj[
                    (df_subj['task'] == task) &
                    (df_subj['condition'] == condition)
                ]
                if df_subset.empty:
                    continue
                n_samples = len(df_subset)
                n_test = int(0.2 * n_samples)
                n_train = n_samples - n_test
                df_train = pd.concat([df_train, df_subset.iloc[:n_train]], axis=0)
                df_test = pd.concat([df_test, df_subset.iloc[n_train:]], axis=0)

    print("Training dataset contains (subjects):")
    print(df_train['subject'].value_counts())
    print("Training labels:")
    print(df_train['Label'].value_counts())

    print("\nTest dataset contains (subjects):")
    print(df_test['subject'].value_counts())
    print("Test labels:")
    print(df_test['Label'].value_counts())

    # ---------- Feature selection from config ----------
    features = return_feature_columns(
        df,
        sensors_to_consider=CONFIG["sensors_to_consider"],
        time_features=CONFIG["time_features"],
        frequency_features=CONFIG["frequency_features"],
        exclude_acc=CONFIG["exclude_acc"],
        exclude_gyro=CONFIG["exclude_gyro"],
        exclude_mag=CONFIG["exclude_mag"],
        exclude_quat=CONFIG["exclude_quat"]
    )

    all_results = []
    all_feature_importances = []

    # ---------- Load LOSO results to get best hyperparams ----------
    results_file = Path(CONFIG['models_folder']) / "model_results_summary.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found – run LOSO search first.")

    results_summary = pd.read_csv(results_file)

    if  not CONFIG["task_specific"]:
        print("Training on all tasks")
        # select only "all_tasks" entries for this window
        df_res = results_summary[
            (results_summary["task"] == "all_tasks") &
            (results_summary["window_size"] == window_to_process)
        ]

        # check if the summary file exists already
        imp_file = Path(CONFIG['importance_folder']) / 'feature_importances_models.csv'
        summary_file = Path(CONFIG['models_folder']) / 'retrain_results_models.csv'

        if summary_file.exists():
            print(f"Summary file:{summary_file} already exists!")
            df_results_all = pd.read_csv(summary_file)
            # the imp file should exist as well
            if imp_file.exists() == False:
                print("Importance file does not exist, check!")
                sys.exit(0)
            else:
                df_imp_all = pd.read_csv(imp_file)
        else:
            df_results_all=pd.DataFrame()
            df_imp_all = pd.DataFrame()


        


        for row in df_res.itertuples():
            model_name = row.model_name
            full_model_path = Path(CONFIG["models_folder"]) / f"{model_name}_win_{window_to_process}_all.pkl"
            if (full_model_path.exists()):
                print(f"model: {full_model_path} already exists!")
            else:
                
                print(f"\n=== Resuming model: {model_name} (win={window_to_process}) ===")
                best_params = eval(row.best_params)
                print(f"Best params: {best_params}")

                # ---------- Train full model on all features ----------
                model = create_model(model_name, best_params)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(df_train[features].values)
                y_train = df_train['Label'].values

                X_test = scaler.transform(df_test[features].values)
                y_test = df_test['Label'].values

                model.fit(X_train, y_train)

                # ---------- Evaluate on test set ----------
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                metrics = compute_metrics(y_test, y_pred, y_proba)
                print("Metrics on test set (full features):", metrics)

                # bootstrap CI over a constant vector (as in your previous pattern)
                for metric in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']:
                    values = [metrics[metric]] * len(df_test)
                    mean, lower, upper = bootstrap_ci(values, n_bootstrap=CONFIG["n_bootstrap"])
                    metrics[f"{metric}_mean"] = mean
                    metrics[f"{metric}_lower"] = lower
                    metrics[f"{metric}_upper"] = upper

                metrics_full = metrics.copy()
                metrics_full.update({
                    'window_size': window_to_process,
                    'task': "all_tasks",
                    'model_name': model_name,
                    'n_features': len(features),
                    'feature_subset': 'all',
                })
                all_results.append(metrics_full)

                # ---------- Save full model ----------
                
                joblib.dump(
                    {
                        "model": model,
                        "scaler": scaler,
                        "features": features,
                        "model_name": model_name,
                        "window_size": window_to_process,
                        "feature_subset": "all",
                    },
                    full_model_path
                )
                print(f"Saved full model to {full_model_path}")

                # ---------- Feature importances (approx) ----------
                importances = None
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    print("Using feature_importances_ for", model_name)
                elif hasattr(model, 'coef_'):
                    coef = model.coef_
                    if coef.ndim == 1:
                        importances = np.abs(coef)
                    else:
                        importances = np.mean(np.abs(coef), axis=0)
                    print("Using coef_ for", model_name)
                else:
                    # fallback: permutation importance on training set
                    try:
                        r = permutation_importance(
                            model,
                            X_train,
                            y_train,
                            n_repeats=20,
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                        importances = r.importances_mean
                        print("Using permutation importance for", model_name)
                    except Exception as e:
                        print("Permutation importance failed for", model_name, e)
                        importances = np.zeros(len(features))

                df_imp = pd.DataFrame({
                    'feature': features,
                    'importance': importances,
                    'task': "all_tasks",
                    'model': model_name,
                    'window_size': window_to_process
                })
                all_feature_importances.append(df_imp)

                # ---------- Retrain on top-k important features ----------
                top_k = 20
                idx_sorted = np.argsort(importances)[::-1]
                top_idx = idx_sorted[:min(top_k, len(idx_sorted))]
                top_features = [features[i] for i in top_idx]

                print(f"Retraining {model_name} with top-{len(top_features)} features (win={window_to_process}):")
                print(top_features)

                # slice & rescale
                X_train_top = X_train[:, top_idx]
                X_test_top = X_test[:, top_idx]

                scaler_top = StandardScaler()
                X_train_top = scaler_top.fit_transform(X_train_top)
                X_test_top = scaler_top.transform(X_test_top)

                model_top = create_model(model_name, best_params)
                model_top.fit(X_train_top, y_train)

                y_pred_top = model_top.predict(X_test_top)
                y_proba_top = model_top.predict_proba(X_test_top) if hasattr(model_top, "predict_proba") else None

                metrics_top = compute_metrics(y_test, y_pred_top, y_proba_top)
                print(f"[TOP-{len(top_features)}] {model_name} (win={window_to_process}) "
                    f"– F1_macro={metrics_top['f1_macro']:.3f}, Acc={metrics_top['accuracy']:.3f}")

                for metric in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']:
                    values = [metrics_top[metric]] * len(df_test)
                    mean, lower, upper = bootstrap_ci(values, n_bootstrap=CONFIG["n_bootstrap"])
                    metrics_top[f"{metric}_mean"] = mean
                    metrics_top[f"{metric}_lower"] = lower
                    metrics_top[f"{metric}_upper"] = upper

                metrics_top.update({
                    'window_size': window_to_process,
                    'task': "all_tasks",
                    'model_name': model_name,
                    'n_features': len(top_features),
                    'feature_subset': 'top_importance',
                })
                all_results.append(metrics_top)

                # Save top-k model
                top_model_path = Path(CONFIG["models_folder"]) / f"{model_name}_win_{window_to_process}_topimportance.pkl"
                joblib.dump(
                    {
                        "model": model_top,
                        "scaler": scaler_top,
                        "features": top_features,
                        "model_name": model_name,
                        "window_size": window_to_process,
                        "feature_subset": "top_importance",
                    },
                    top_model_path
                )
                print(f"Saved top-importance model to {top_model_path}")

    # ------------- SAVE FEATURE IMPORTANCES & RESULTS SUMMARY -------------
    if all_feature_importances:
        df_imp_all_tmp = pd.concat(all_feature_importances, ignore_index=True)
        df_imp = pd.concat((df_imp, df_imp_all_tmp))
        df_imp_all.to_csv(imp_file, index=False)
        print(f"Saved feature importances to {imp_file}")

        # optional: heatmaps per task
        for task in df_imp_all['task'].unique():
            plot_feature_importance_heatmap(df_imp_all, task, CONFIG['importance_folder'],
                                            top_n=CONFIG['top_n_features'])

    if all_results:
        df_results_all_tmp = pd.DataFrame(all_results)
        df_results_all = pd.concat((df_results_all, df_results_all_tmp))
        df_results_all.to_csv(summary_file, index=False)
        print(f"Saved results summary to {summary_file}")


if __name__ == '__main__':
    main()
