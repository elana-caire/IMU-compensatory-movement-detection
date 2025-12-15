import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import shap

import matplotlib.pyplot as plt  # not used here but fine
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, make_scorer

from utils.ml import (
    encode_labels,
    return_feature_columns,
    create_temporal_train_test_folds,
    balanced_subsample_after_scaling,
    parse_params_cell,
    shap_to_3d,
    agg_stats,
)
from config import CONFIG

# --------------------------- CONFIG ---------------------------
BG_N = 100
EXPL_N = 50
KERNEL_NSAMPLES = 200

PERM_N_REPEATS = 10
PERM_SCORER = make_scorer(f1_score, average="macro")

MODELS_TO_EXPLAIN = ["MLP"]
WINDOW_TO_PROCESS = 750

models_root = Path(CONFIG["models_folder"])
results_agg_path = models_root / "Global_CV_Results_agg.csv"

out_root = models_root / "Global_CV_Explainability_bestcfg"
out_shap = out_root / "shap"
out_perm = out_root / "perm"
out_metrics = out_root / "metrics"
out_shap.mkdir(parents=True, exist_ok=True)
out_perm.mkdir(parents=True, exist_ok=True)
out_metrics.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = CONFIG.get("random_state", 42)


def safe_balanced_subsample(X, y, n_total, random_state):
    """Avoid requesting more samples than available."""
    n_total = int(min(n_total, len(y)))
    if n_total <= 0:
        return X[:0], y[:0]
    return balanced_subsample_after_scaling(X, y, n_total=n_total, random_state=random_state)


def main():
    if not results_agg_path.exists():
        raise FileNotFoundError(f"Missing: {results_agg_path}")

    # ---------- load data ----------
    df = pd.read_csv(Path(CONFIG["features_folder"]) / f"features_win_{WINDOW_TO_PROCESS}.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df, label_encoder = encode_labels(df, CONFIG["target"])

    features = return_feature_columns(
        df,
        sensors_to_consider=CONFIG["sensors_to_consider"],
        time_features=CONFIG["time_features"],
        frequency_features=CONFIG["frequency_features"],
        exclude_acc=CONFIG["exclude_acc"],
        exclude_gyro=CONFIG["exclude_gyro"],
        exclude_mag=CONFIG["exclude_mag"],
        exclude_quat=CONFIG["exclude_quat"],
    )

    # ---------- tasks ----------
    if CONFIG["task_specific"]:
        tasks_to_run = list(df["task"].unique())
    else:
        tasks_to_run = ["all_tasks"]  # <-- all-tasks model

    # In all_tasks mode, we still want to evaluate per-task
    eval_tasks = sorted(df["task"].unique())  # e.g., 4 tasks

    # ---------- load agg + pick best params per (task, model) ----------
    df_agg = pd.read_csv(results_agg_path)

    if "f1_macro_mean" not in df_agg.columns:
        raise ValueError("Global_CV_Results_agg.csv must contain f1_macro_mean")

    df_agg = df_agg[df_agg["window_size"].astype(str) == str(WINDOW_TO_PROCESS)].copy()
    df_agg = df_agg[df_agg["model_name"].isin(MODELS_TO_EXPLAIN)].copy()

    df_best = (
        df_agg.sort_values("f1_macro_mean", ascending=False)
              .groupby(["task", "model_name"], as_index=False)
              .head(1)
    )

    best_lookup = {}
    for _, r in df_best.iterrows():
        best_lookup[(r["task"], r["model_name"])] = {
            "params_id": r["params_id"],
            "best_params": parse_params_cell(r["best_params"]),
        }

    # ---------- print best aggregated metrics ----------
    metric_bases = ["f1_macro", "accuracy", "precision_macro", "recall_macro", "roc_auc"]
    print("\n================= BEST CONFIG METRICS (mean ± std across folds) =================")
    for _, r in df_best.iterrows():
        parts = []
        for mb in metric_bases:
            mcol, scol = f"{mb}_mean", f"{mb}_std"
            if mcol in r.index:
                parts.append(f"{mb}={float(r[mcol]):.3f}±{float(r[scol]) if scol in r.index and not pd.isna(r[scol]) else float('nan'):.3f}")
        print(f"- task={r['task']} | model={r['model_name']} | params_id={r['params_id']} | " + ", ".join(parts))
    print("===============================================================================\n")

    # ---------- loop tasks/models, reload fold models ----------
    for task_name in tasks_to_run:
        # all_tasks training fold split should be created from full df
        df_task = df.copy() if (not CONFIG["task_specific"]) else df[df["task"] == task_name].copy()

        folds = create_temporal_train_test_folds(df_task)

        for model_name in MODELS_TO_EXPLAIN:
            key = (task_name, model_name)
            if key not in best_lookup:
                print(f"[WARN] No best config for task={task_name}, model={model_name}")
                continue

            params_id = best_lookup[key]["params_id"]
            best_params = best_lookup[key]["best_params"]
            print(f"[INFO] Using best params task={task_name}, model={model_name}: {best_params}")

            model_folder = models_root / task_name / model_name / params_id

            # outputs: include eval_task in all of them
            per_fold_metrics_rows = []
            per_fold_perm_rows = []
            per_fold_shap_rows = []

            for fold_idx, df_train, df_test in folds:
                model_file = model_folder / f"fold_{fold_idx}.joblib"
                if not model_file.exists():
                    print(f"[SKIP] missing model: {model_file}")
                    continue

                bundle = joblib.load(model_file)
                model = bundle["model"]
                scaler = bundle["scaler"]
                feat_order = bundle.get("features", features)

                # Scale full fold once
                X_train = df_train[feat_order].values
                y_train = df_train["Label"].values
                X_test = df_test[feat_order].values
                y_test = df_test["Label"].values

                X_train_s = scaler.transform(X_train)
                X_test_s = scaler.transform(X_test)

                # ------------------- METRICS -------------------
                # Evaluate on whole fold test (eval_task="all_tasks")
                y_pred_all = model.predict(X_test_s)
                acc_all = accuracy_score(y_test, y_pred_all)
                f1_all = f1_score(y_test, y_pred_all, average="macro")

                per_fold_metrics_rows.append({
                    "window_size": WINDOW_TO_PROCESS,
                    "trained_task": task_name,        # model training scope (here: all_tasks)
                    "eval_task": "all_tasks",         # evaluation subset
                    "model_name": model_name,
                    "params_id": params_id,
                    "fold_idx": fold_idx,
                    "accuracy": float(acc_all),
                    "f1_macro": float(f1_all),
                })

                # In all_tasks mode: compute per-task accuracy/F1 within the same fold
                if (not CONFIG["task_specific"]) and task_name == "all_tasks":
                    for t in eval_tasks:
                        m = (df_test["task"].values == t)
                        if m.sum() == 0:
                            continue
                        acc_t = accuracy_score(y_test[m], y_pred_all[m])
                        f1_t = f1_score(y_test[m], y_pred_all[m], average="macro")
                        per_fold_metrics_rows.append({
                            "window_size": WINDOW_TO_PROCESS,
                            "trained_task": task_name,
                            "eval_task": t,
                            "model_name": model_name,
                            "params_id": params_id,
                            "fold_idx": fold_idx,
                            "accuracy": float(acc_t),
                            "f1_macro": float(f1_t),
                        })

                # ------------------- EXPLAINABILITY -------------------
                # Helper: run explainability on (train_subset, test_subset) but same model
                def run_explainability(eval_task_label, df_train_sub, df_test_sub):
                    if df_train_sub.empty or df_test_sub.empty:
                        return

                    Xtr = scaler.transform(df_train_sub[feat_order].values)
                    ytr = df_train_sub["Label"].values
                    Xte = scaler.transform(df_test_sub[feat_order].values)
                    yte = df_test_sub["Label"].values

                    # If a task subset collapses to 1 class, skip (PI/SHAP not meaningful)
                    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
                        return

                    X_bg, _ = safe_balanced_subsample(Xtr, ytr, n_total=BG_N, random_state=RANDOM_STATE)
                    X_expl, y_expl = safe_balanced_subsample(Xte, yte, n_total=EXPL_N, random_state=RANDOM_STATE)
                    if X_bg.shape[0] == 0 or X_expl.shape[0] == 0:
                        return

                    # --- PI on test subset ---
                    try:
                        pi = permutation_importance(
                            model,
                            X_expl,
                            np.asarray(y_expl),
                            scoring=PERM_SCORER,
                            n_repeats=PERM_N_REPEATS,
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                        for f, m_imp, s_imp in zip(feat_order, pi.importances_mean, pi.importances_std):
                            per_fold_perm_rows.append({
                                "window_size": WINDOW_TO_PROCESS,
                                "trained_task": task_name,
                                "eval_task": eval_task_label,
                                "model_name": model_name,
                                "params_id": params_id,
                                "fold_idx": fold_idx,
                                "feature": f,
                                "perm_importance_mean": float(m_imp),
                                "perm_importance_std": float(s_imp),
                                "n_repeats": PERM_N_REPEATS,
                            })
                    except Exception as e:
                        print(f"[WARN] PI failed eval_task={eval_task_label} fold={fold_idx}: {e}")

                    # --- SHAP on test subset ---
                    try:
                        if model_name == "LASSO_LR":
                            explainer = shap.LinearExplainer(model, X_bg)
                            shap_vals = explainer.shap_values(X_expl)
                        else:
                            explainer = shap.KernelExplainer(model.predict_proba, X_bg)
                            shap_vals = explainer.shap_values(X_expl, nsamples=KERNEL_NSAMPLES)

                        sv3 = shap_to_3d(shap_vals)           # (n, f, c)
                        imp_fc = np.mean(np.abs(sv3), axis=0) # (f, c)
                        n_classes = imp_fc.shape[1]

                        for fi, feat in enumerate(feat_order):
                            for ci in range(n_classes):
                                per_fold_shap_rows.append({
                                    "window_size": WINDOW_TO_PROCESS,
                                    "trained_task": task_name,
                                    "eval_task": eval_task_label,
                                    "model_name": model_name,
                                    "params_id": params_id,
                                    "fold_idx": fold_idx,
                                    "feature": feat,
                                    "class": ci,
                                    "mean_abs_shap": float(imp_fc[fi, ci]),
                                })
                    except Exception as e:
                        print(f"[WARN] SHAP failed eval_task={eval_task_label} fold={fold_idx}: {e}")

                # (A) explainability on full fold test (eval_task="all_tasks")
                run_explainability("all_tasks", df_train, df_test)

                # (B) in all_tasks mode: per-task explainability using SAME model
                if (not CONFIG["task_specific"]) and task_name == "all_tasks":
                    for t in eval_tasks:
                        df_train_t = df_train[df_train["task"] == t]
                        df_test_t  = df_test[df_test["task"] == t]
                        run_explainability(t, df_train_t, df_test_t)

            # ---------------- SAVE METRICS ----------------
            df_m = pd.DataFrame(per_fold_metrics_rows)
            if not df_m.empty:
                metrics_per_fold_path = out_metrics / f"metrics_{task_name}_{model_name}_best_per_fold_per_task.csv"
                df_m.to_csv(metrics_per_fold_path, index=False)

                # per-task aggregate across folds
                df_m_agg = (
                    df_m.groupby(["trained_task", "eval_task", "model_name", "params_id"], as_index=False)[["accuracy", "f1_macro"]]
                        .agg(["mean", "std"])
                        .reset_index()
                )
                df_m_agg.columns = [f"{a}_{b}" if b else a for a, b in df_m_agg.columns.to_flat_index()]
                metrics_agg_path = out_metrics / f"metrics_{task_name}_{model_name}_best_agg_per_task.csv"
                df_m_agg.to_csv(metrics_agg_path, index=False)

                print(f"[Saved] {metrics_per_fold_path}")
                print(f"[Saved] {metrics_agg_path}")

                # print mean across tasks (unweighted), using per-fold values
                if (not CONFIG["task_specific"]) and task_name == "all_tasks":
                    df_only_tasks = df_m[df_m["eval_task"].isin(eval_tasks)]
                    if not df_only_tasks.empty:
                        per_task_mean = df_only_tasks.groupby("eval_task")[["accuracy", "f1_macro"]].mean()
                        print("\n[all_tasks model] Mean across folds per eval_task:\n", per_task_mean)

                        mean_across_tasks = per_task_mean.mean(axis=0)  # unweighted mean over tasks
                        print("\n[all_tasks model] Unweighted mean across tasks:")
                        print(f"  accuracy={mean_across_tasks['accuracy']:.3f}, f1_macro={mean_across_tasks['f1_macro']:.3f}\n")

            # ---------------- SAVE PERM ----------------
            df_perm = pd.DataFrame(per_fold_perm_rows)
            if not df_perm.empty:
                perm_per_fold_path = out_perm / f"perm_{task_name}_{model_name}_best_per_fold_evalTask.csv"
                df_perm.to_csv(perm_per_fold_path, index=False)

                df_perm_agg = (
                    df_perm.groupby(["trained_task", "eval_task", "model_name", "feature"])["perm_importance_mean"]
                           .apply(agg_stats)
                           .reset_index()
                           .rename(columns={"level_4": "stat"})  # <-- note: level index depends on pandas; adjust if needed
                )
                df_perm_agg = df_perm_agg.pivot_table(
                    index=["trained_task", "eval_task", "model_name", "feature"],
                    columns="stat",
                    values="perm_importance_mean",
                    aggfunc="first"
                ).reset_index()

                perm_agg_path = out_perm / f"perm_{task_name}_{model_name}_best_agg_evalTask.csv"
                df_perm_agg.to_csv(perm_agg_path, index=False)

                print(f"[Saved] {perm_per_fold_path}")
                print(f"[Saved] {perm_agg_path}")

            # ---------------- SAVE SHAP ----------------
            df_shap = pd.DataFrame(per_fold_shap_rows)
            if not df_shap.empty:
                shap_per_fold_path = out_shap / f"shap_{task_name}_{model_name}_best_per_fold_evalTask.csv"
                df_shap.to_csv(shap_per_fold_path, index=False)

                df_shap_agg = (
                    df_shap.groupby(["trained_task", "eval_task", "model_name", "feature", "class"])["mean_abs_shap"]
                           .apply(agg_stats)
                           .reset_index()
                           .rename(columns={"level_5": "stat"})  # <-- same note as above
                )
                df_shap_agg = df_shap_agg.pivot_table(
                    index=["trained_task", "eval_task", "model_name", "feature", "class"],
                    columns="stat",
                    values="mean_abs_shap",
                    aggfunc="first"
                ).reset_index()

                shap_agg_path = out_shap / f"shap_{task_name}_{model_name}_best_agg_evalTask.csv"
                df_shap_agg.to_csv(shap_agg_path, index=False)

                print(f"[Saved] {shap_per_fold_path}")
                print(f"[Saved] {shap_agg_path}")


if __name__ == "__main__":
    main()
