import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils.ml import *
from config import CONFIG

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------- CONFIG ---------------------------
# create folders
Path(CONFIG["plots_folder"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["importance_folder"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["shap_folder"]).mkdir(parents=True, exist_ok=True)

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
        "LASSO_LR": LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial', max_iter=5000, random_state=RANDOM_STATE)
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

# --------------------------- MAIN -----------------------------

def main():
    feature_files = find_feature_files(CONFIG["features_folder"])
    all_results = []
    all_feature_importances = []
    all_shap_importances = []

    
    # Check if results file exists to resume


    results_file = Path(CONFIG['models_folder'])/"model_results_summary.csv"
    if results_file.exists():
        print("Loading existing results from", results_file)
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame()
        

    for file_path, window_size in tqdm(feature_files, desc="Processing windows"):
        print("Processing window :", window_size)
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df, label_encoder = encode_labels(df, CONFIG["target"])


        #features = [c for c in df.columns if c not in ['subject', 'task', CONFIG["target"], 'Label']]

        ## Consider Experimental Parameters from config.py

        features = return_feature_columns(df, 
                                        sensors_to_consider=CONFIG["sensors_to_consider"], 
                                        time_features=CONFIG["time_features"], 
                                        frequency_features=CONFIG["frequency_features"], 
                                        exclude_acc=CONFIG["exclude_acc"], 
                                        exclude_gyro=CONFIG["exclude_gyro"], 
                                        exclude_mag=CONFIG["exclude_mag"], 
                                        exclude_quat=CONFIG["exclude_quat"])
        print("Considering features:", len(features))
        print("Features:", features)
        
        
        subjects = df['subject'].unique()
        tasks = df['task'].unique()



        # check if we want to train on specific tasks only
        if not CONFIG["task_specific"]:
            print("Training on all tasks combined")
            for model_name in CONFIG["models_to_use"]:
                print("Model :", model_name)
                best_score = -np.inf
                best_params = None
                best_predictions = None

                # grid search via ParameterGrid
                for params in ParameterGrid(CONFIG["param_grids"][model_name]):
                    subject_scores = []
                    preds_all, probs_all, y_true_all = [], [], []

                    for test_subject in subjects:
                        train_mask = df['subject'] != test_subject
                        test_mask = df['subject'] == test_subject

                        X_train = df.loc[train_mask, features].values
                        y_train = df.loc[train_mask, 'Label'].values
                        X_test = df.loc[test_mask, features].values
                        y_test = df.loc[test_mask, 'Label'].values

                        if len(X_train) == 0 or len(X_test) == 0:
                            continue

                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                        X_train, y_train = sk_shuffle(X_train, y_train, random_state=RANDOM_STATE)

                        model = create_model(model_name, params)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                        preds_all.append(y_pred)
                        y_true_all.append(y_test)
                        if y_proba is not None:
                            probs_all.append(y_proba)

                        subject_scores.append(f1_score(y_test, y_pred, average='macro'))

                    if not preds_all:
                        continue

                    mean_score = np.mean(subject_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                        best_predictions = (
                            np.concatenate(y_true_all),
                            np.concatenate(preds_all),
                            np.vstack(probs_all) if probs_all else None
                        )

                if best_predictions is None:
                    continue

                y_true_all, y_pred_all, y_proba_all = best_predictions
                metrics = compute_metrics(y_true_all, y_pred_all, y_proba_all)
                print("Metrics computed:", metrics)
                for metric in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']:
                    values = [metrics[metric]] * len(subjects)
                    mean, lower, upper = bootstrap_ci(values, n_bootstrap=CONFIG["n_bootstrap"])
                    
                    metrics[f"{metric}"] = metrics[metric]
                    metrics[f"{metric}_mean"] = mean
                    metrics[f"{metric}_lower"] = lower
                    metrics[f"{metric}_upper"] = upper
                    print("Without CI for", metric, ":", metrics[metric])
                    print("Bootstrap CI for", metric, ":", mean, lower, upper)
                metrics.update({
                    'window_size': window_size,
                    'task': "all_tasks",
                    'model_name': model_name,
                    'best_params': str(best_params)
                })
                all_results.append(metrics)
                # concat to results_df
                results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
                # save after each model
                print(f"Completed {model_name} on all tasks (win={window_size})")
                print(f"Best params: {best_params}")
                print(f"Accuracy={metrics['accuracy_mean']:.3f} (+{metrics['accuracy_upper'] - metrics['accuracy_mean']:.3f}/-{metrics['accuracy_mean'] - metrics['accuracy_lower']:.3f}),")
                
                results_df.to_csv(results_file, index=False)

        if CONFIG["task_specific"]:
            # Train task specific models    
            for task in tasks:
                print("Task :",task)
                df_task = df[df['task'] == task]
                if df_task.empty:
                    continue

                for model_name in CONFIG["models_to_use"]:
                    print("Model :", model_name)
                    best_score = -np.inf
                    best_params = None
                    best_predictions = None

                    # grid search via ParameterGrid
                    for params in ParameterGrid(CONFIG["param_grids"][model_name]):
                        subject_scores = []
                        preds_all, probs_all, y_true_all = [], [], []

                        for test_subject in subjects:
                            train_mask = df_task['subject'] != test_subject
                            test_mask = df_task['subject'] == test_subject

                            X_train = df_task.loc[train_mask, features].values
                            y_train = df_task.loc[train_mask, 'Label'].values
                            X_test = df_task.loc[test_mask, features].values
                            y_test = df_task.loc[test_mask, 'Label'].values

                            if len(X_train) == 0 or len(X_test) == 0:
                                continue

                            scaler = StandardScaler()
                            X_train = scaler.fit_transform(X_train)
                            X_test = scaler.transform(X_test)
                            X_train, y_train = sk_shuffle(X_train, y_train, random_state=RANDOM_STATE)

                            model = create_model(model_name, params)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

                            preds_all.append(y_pred)
                            y_true_all.append(y_test)
                            if y_proba is not None:
                                probs_all.append(y_proba)

                            subject_scores.append(f1_score(y_test, y_pred, average='macro'))

                        if not preds_all:
                            continue

                        mean_score = np.mean(subject_scores)
                        if mean_score > best_score:
                            best_score = mean_score
                            best_params = params
                            best_predictions = (
                                np.concatenate(y_true_all),
                                np.concatenate(preds_all),
                                np.vstack(probs_all) if probs_all else None
                            )

                    if best_predictions is None:
                        continue

                    y_true_all, y_pred_all, y_proba_all = best_predictions
                    metrics = compute_metrics(y_true_all, y_pred_all, y_proba_all)

                    for metric in ['f1_macro', 'accuracy', 'precision_macro', 'recall_macro', 'roc_auc']:
                        values = [metrics[metric]] * len(subjects)
                        mean, lower, upper = bootstrap_ci(values, n_bootstrap=CONFIG["n_bootstrap"])
                        metrics[f"{metric}_mean"] = mean
                        metrics[f"{metric}_lower"] = lower
                        metrics[f"{metric}_upper"] = upper

                    metrics.update({
                        'window_size': window_size,
                        'task': task,
                        'model_name': model_name,
                        'best_params': str(best_params)
                    })
                    all_results.append(metrics)
                    # concat to results_df
                    results_df = pd.concat([results_df, pd.DataFrame([metrics])], ignore_index=True)
                    # save after each model
                    print(f"Completed {model_name} on task {task} (win={window_size})")
                    print(f"Accuracy={metrics['accuracy_mean']:.3f} (+{metrics['accuracy_upper'] - metrics['accuracy_mean']:.3f}/-{metrics['accuracy_mean'] - metrics['accuracy_lower']:.3f}),")
                    
                    results_df.to_csv(results_file, index=False)
                
                    # ------------------ Retrain Best Performing on full task data ------------------
                    full_model = create_model(model_name, best_params)
                    scaler = StandardScaler()

                    X_all = scaler.fit_transform(df_task[features].values)
                    y_all = df_task['Label'].values

                    # If no samples, skip
                    if len(X_all) == 0:
                        continue

                    full_model.fit(X_all, y_all)

                    # ------------------ Feature importances ------------------
                    importances = None
                    if hasattr(full_model, 'feature_importances_'):
                        importances = full_model.feature_importances_
                    elif hasattr(full_model, 'coef_'):
                        # For multiclass, coef_ is (n_classes, n_features); take mean abs
                        coef = full_model.coef_
                        if coef.ndim == 1:
                            importances = np.abs(coef)
                        else:
                            importances = np.mean(np.abs(coef), axis=0)
                    else:
                        # fallback to permutation importance
                        try:
                            r = permutation_importance(full_model, X_all, y_all, n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
                            importances = r.importances_mean
                        except Exception as e:
                            print("Permutation importance failed for", model_name, e)
                            importances = np.zeros(len(features))

                    df_imp = pd.DataFrame({
                        'feature': features,
                        'importance': importances,
                        'task': task,
                        'model': model_name,
                        'window_size': window_size
                    })
                    all_feature_importances.append(df_imp)

                    # ------------------ SHAP for tree models + Logistic (if available) ------------------
                    shap_mean_abs = None
                    shap_possible_models = ['RF', 'XGBoost', 'LightGBM', 'LASSO_LR']
                    if SHAP_AVAILABLE and model_name in shap_possible_models:
                        try:
                            # sample background to limit compute
                            sample_size = min(CONFIG['shap_sample_size'], X_all.shape[0])
                            if sample_size <= 0:
                                continue
                            # pick background/sample rows
                            if sample_size < X_all.shape[0]:
                                idx = np.random.choice(np.arange(X_all.shape[0]), size=sample_size, replace=False)
                                X_shap = X_all[idx]
                            else:
                                X_shap = X_all

                            if model_name in ['XGBoost', 'LightGBM', 'RF']:
                                explainer = shap.TreeExplainer(full_model)
                                shap_values = explainer.shap_values(X_shap)
                            elif model_name == 'LASSO_LR':
                                explainer = shap.LinearExplainer(full_model, X_shap, feature_perturbation='interventional')
                                shap_values = explainer.shap_values(X_shap)
                            else:
                                shap_values = None

                            if shap_values is not None:
                                # unify to array for mean absolute computation
                                if isinstance(shap_values, list):
                                    # multiclass tree / linear models: list of arrays (n_samples, n_features)
                                    shap_arr = np.stack(shap_values, axis=-1)  # shape: (n_samples, n_features, n_classes)
                                else:
                                    shap_arr = shap_values  # shape: (n_samples, n_features)

                                shap_arr = np.abs(shap_arr)

                                if shap_arr.ndim == 2:
                                    shap_mean_abs = shap_arr.mean(axis=0)
                                elif shap_arr.ndim == 3:
                                    shap_mean_abs = shap_arr.mean(axis=0).mean(axis=1)  # mean over samples, then classes
                                else:
                                    raise ValueError(f"Unexpected SHAP array dimensions: {shap_arr.shape}")

                        except Exception as e:
                            print(f"SHAP failed for {model_name} (task={task}, win={window_size}):", e)
                            shap_mean_abs = None
                
                    if shap_mean_abs is not None:
                        df_shap = pd.DataFrame({
                            'feature': features,
                            'shap_mean_abs': shap_mean_abs,
                            'task': task,
                            'model': model_name,
                            'window_size': window_size
                        })
                        all_shap_importances.append(df_shap)
    # -----------------------------------------------------------------------
        


    # ------------------ Save results & plots ------------------
    #results_df = pd.DataFrame(all_results)
    #results_df.to_csv(Path(CONFIG['models_folder']), index=False)

    if all_feature_importances:
        df_imp_all = pd.concat(all_feature_importances, ignore_index=True)
        df_imp_all.to_csv(Path(CONFIG['importance_folder']) / 'feature_importances_all.csv', index=False)
        for task in df_imp_all['task'].unique():
            plot_feature_importance_heatmap(df_imp_all, task, CONFIG['importance_folder'], top_n=CONFIG['top_n_features'])

    if all_shap_importances and SHAP_AVAILABLE:
        df_shap_all = pd.concat(all_shap_importances, ignore_index=True)
        df_shap_all.to_csv(Path(CONFIG['shap_folder']) / 'shap_importances_all.csv', index=False)
        for task in df_shap_all['task'].unique():
            plot_shap_heatmap(df_shap_all, task, CONFIG['shap_folder'], top_n=CONFIG['top_n_features'])

    # also produce metric plots per task
    if not results_df.empty:
        for task in results_df['task'].unique():
            plot_metrics_across_windows(results_df, task, CONFIG['plots_folder'])

    print('Done. Results saved in:', CONFIG['plots_folder'], CONFIG['importance_folder'], CONFIG['shap_folder'])


if __name__ == '__main__':
    main()
