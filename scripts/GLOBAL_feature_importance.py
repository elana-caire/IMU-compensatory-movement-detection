import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import shap

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, make_scorer
)

from utils.ml import (
    encode_labels,
    return_feature_columns,
    create_temporal_train_test_folds,
    balanced_subsample_after_scaling,
    parse_params_cell,
    shap_to_3d,
)

import config.training_common as training_config
import config.time_split as time_split_config
import config.paths as paths_config

MODELS_PATH = paths_config.MODELS_PATH
FEATURES_PATH = paths_config.FEATURES_PATH
BEST_MODELS_PATH = paths_config.BEST_MODELS_PATH
SHAP_PATH = paths_config.SHAP_PATH
IMPORTANCE_PATH = paths_config.IMPORTANCE_PATH

BG_N = time_split_config.BG_N
EXPL_N = time_split_config.EXPL_N
KERNEL_NSAMPLES = time_split_config.KERNEL_NSAMPLES

# Permutation importance control
PERM_N_REPEATS = time_split_config.PERM_N_REPEATS
PERM_SCORER = make_scorer(f1_score, average="macro")

# Calibration (reliability diagram on confidence)
CAL_N_BINS = time_split_config.CAL_N_BINS

MODELS_TO_EXPLAIN = time_split_config.BEST_MODELS_FOR_EXPLAINABILITY
WINDOW_TO_PROCESS = time_split_config.WINDOW_TO_PROCESS
TARGET = training_config.TARGET

results_folds_path = MODELS_PATH / "model_results_Global_folds.csv"
results_agg_path   = MODELS_PATH / "model_results_Global_agg.csv"

out_shap = SHAP_PATH
out_perm = IMPORTANCE_PATH
out_metrics = BEST_MODELS_PATH
out_cal = BEST_MODELS_PATH

out_shap.mkdir(parents=True, exist_ok=True)
out_perm.mkdir(parents=True, exist_ok=True)
out_metrics.mkdir(parents=True, exist_ok=True)
out_cal.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = training_config.RANDOM_STATE
TARGET = training_config.TARGET
SENSORS = training_config.SENSORS
TIME_FEATURES = training_config.TIME_FEATURES
FREQ_FEATURES = training_config.FREQ_FEATURES
EXCLUDE_QUAT = training_config.EXCLUDE_QUAT
EXCLUDE_ACC = training_config.EXCLUDE_ACC
EXCLUDE_GYRO = training_config.EXCLUDE_GYRO
EXCLUDE_MAG = training_config.EXCLUDE_MAG
PARAM_GRIDS = training_config.PARAM_GRIDS

TASK_SPECIFIC = time_split_config.TASK_SPECIFIC


# --------------------------- HELPERS ---------------------------
def safe_balanced_subsample(X, y, n_total, random_state):
    """Avoid requesting more samples than available."""
    n_total = int(min(n_total, len(y)))
    if n_total <= 0:
        return X[:0], y[:0]
    return balanced_subsample_after_scaling(X, y, n_total=n_total, random_state=random_state)


def compute_fold_metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float("nan"),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(
                roc_auc_score(
                    pd.get_dummies(y_true),
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except Exception:
            out["roc_auc"] = float("nan")
    return out


def reliability_bins_from_proba(y_true, y_proba, n_bins=10):
    """
    Multiclass reliability via confidence:
      conf = max_k p(y=k)
      correct = 1[pred==y_true]
    Returns fixed-length arrays with NaNs for empty bins:
      acc_per_bin, conf_per_bin, count_per_bin, bin_edges
    """
    preds = np.argmax(y_proba, axis=1)
    conf = np.max(y_proba, axis=1)
    correct = (preds == y_true).astype(int)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc = np.full(n_bins, np.nan, dtype=float)
    cavg = np.full(n_bins, np.nan, dtype=float)
    cnt = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        if b < n_bins - 1:
            m = (conf >= edges[b]) & (conf < edges[b + 1])
        else:
            m = (conf >= edges[b]) & (conf <= edges[b + 1])
        cnt[b] = int(m.sum())
        if cnt[b] > 0:
            acc[b] = float(correct[m].mean())
            cavg[b] = float(conf[m].mean())

    return acc, cavg, cnt, edges


def save_agg_mean_std(df, group_cols, value_cols, out_path):
    """
    Save mean/std across folds (or subjects) for the given value_cols.
    Produces columns like: {col}_mean and {col}_std.
    """
    if df.empty:
        return None
    agg = (
        df.groupby(group_cols, as_index=False)[value_cols]
          .agg(["mean", "std"])
          .reset_index()
    )
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns.to_flat_index()]
    agg.to_csv(out_path, index=False)
    return agg


# --------------------------- MAIN ---------------------------
def main():
    if not results_agg_path.exists():
        raise FileNotFoundError(f"Missing: {results_agg_path}")

    # ---------- load data ----------
    df = pd.read_csv(Path(FEATURES_PATH) / f"features_win_{WINDOW_TO_PROCESS}.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df, label_encoder = encode_labels(df, TARGET)

    features = return_feature_columns(
        df,
        sensors_to_consider=SENSORS,
        time_features=TIME_FEATURES,
        frequency_features=FREQ_FEATURES,
        exclude_acc=EXCLUDE_ACC,
        exclude_gyro=EXCLUDE_GYRO,
        exclude_mag=EXCLUDE_MAG,
        exclude_quat=EXCLUDE_QUAT,
    )

    # ---------- decide tasks ----------
    if TASK_SPECIFIC:
        tasks_to_run = list(df["task"].unique())
    else:
        tasks_to_run = ["all_tasks"]  # global model

    # In all_tasks mode, we still want to evaluate per-task
    eval_tasks = sorted(df["task"].unique())

    # ---------- load agg results and select best params_id per (task, model) ----------
    df_agg = pd.read_csv(results_agg_path)

    if "f1_macro_mean" not in df_agg.columns:
        raise ValueError("Global_CV_Results_agg.csv must contain f1_macro_mean (from your aggregation).")

    df_agg = df_agg[df_agg["window_size"].astype(str) == str(WINDOW_TO_PROCESS)].copy()
    df_agg = df_agg[df_agg["model_name"].isin(MODELS_TO_EXPLAIN)].copy()

    # pick best by mean macro-F1 for each (task, model)
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
                mean_val = float(r[mcol])
                std_val = float(r[scol]) if (scol in r.index and not pd.isna(r[scol])) else float("nan")
                parts.append(f"{mb}={mean_val:.3f}±{std_val:.3f}")
        print(f"- task={r['task']} | model={r['model_name']} | params_id={r['params_id']} | " + ", ".join(parts))
    print("===============================================================================\n")

    # ---------- loop tasks/models, reload fold models ----------
    for task_name in tasks_to_run:
        df_task = df.copy() if (not TASK_SPECIFIC) else df[df["task"] == task_name].copy()
        folds = create_temporal_train_test_folds(df_task)

        for model_name in MODELS_TO_EXPLAIN:
            key = (task_name, model_name)
            if key not in best_lookup:
                print(f"[WARN] No best config for task={task_name}, model={model_name}. Skipping.")
                continue

            params_id = best_lookup[key]["params_id"]
            best_params = best_lookup[key]["best_params"]
            print(f"[INFO] Using best params task={task_name}, model={model_name}: {best_params}")

            model_folder = BEST_MODELS_PATH / task_name / model_name / params_id

            # output paths (evalTask format)
            metrics_per_fold_path = out_metrics / f"metrics_{task_name}_{model_name}_best_per_fold_per_task.csv"
            metrics_agg_path      = out_metrics / f"metrics_{task_name}_{model_name}_best_agg_per_task.csv"

            perm_per_fold_path    = out_perm / f"perm_{task_name}_{model_name}_best_per_fold_evalTask.csv"
            perm_agg_path         = out_perm / f"perm_{task_name}_{model_name}_best_agg_evalTask.csv"

            shap_per_fold_path    = out_shap / f"shap_{task_name}_{model_name}_best_per_fold_evalTask.csv"
            shap_agg_path         = out_shap / f"shap_{task_name}_{model_name}_best_agg_evalTask.csv"

            cal_csv_path          = out_cal / f"calibration_{task_name}_{model_name}_confidence_bins.csv"
            cal_fig_path          = out_cal / f"calibration_{task_name}_{model_name}_confidence.png"

            # skip expensive parts if already computed
            skip_perm = perm_per_fold_path.exists() and perm_agg_path.exists()
            skip_shap = shap_per_fold_path.exists() and shap_agg_path.exists()

            if skip_perm:
                print(f"[SKIP] Permutation importance already computed: {perm_per_fold_path.name}")
            if skip_shap:
                print(f"[SKIP] SHAP already computed: {shap_per_fold_path.name}")

            per_fold_metrics_rows = []
            per_fold_perm_rows = [] if not skip_perm else None
            per_fold_shap_rows = [] if not skip_shap else None

            # calibration storage (only global eval_task="all_tasks")
            cal_rows = []

            for fold_idx, df_train, df_test in folds:
                model_file = model_folder / f"fold_{fold_idx}.joblib"
                if not model_file.exists():
                    print(f"[SKIP] missing model: {model_file}")
                    continue

                bundle = joblib.load(model_file)
                model = bundle["model"]
                scaler = bundle["scaler"]
                feat_order = bundle.get("features", features)

                # scale full fold once
                X_train = df_train[feat_order].values
                y_train = df_train["Label"].values
                X_test  = df_test[feat_order].values
                y_test  = df_test["Label"].values

                X_train_s = scaler.transform(X_train)
                X_test_s  = scaler.transform(X_test)

                # ------------------- METRICS (ALL) -------------------
                y_pred_all = model.predict(X_test_s)
                y_proba_all = model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None

                m_all = compute_fold_metrics(y_test, y_pred_all, y_proba_all)

                per_fold_metrics_rows.append({
                    "window_size": WINDOW_TO_PROCESS,
                    "trained_task": task_name,
                    "eval_task": "all_tasks",
                    "model_name": model_name,
                    "params_id": params_id,
                    "fold_idx": fold_idx,
                    **m_all,
                })

                # per-task metrics inside all_tasks model
                if (not TASK_SPECIFIC) and task_name == "all_tasks":
                    for t in eval_tasks:
                        mask = (df_test["task"].values == t)
                        if mask.sum() == 0:
                            continue
                        y_true_t = y_test[mask]
                        y_pred_t = y_pred_all[mask]
                        y_proba_t = y_proba_all[mask] if y_proba_all is not None else None

                        m_t = compute_fold_metrics(y_true_t, y_pred_t, y_proba_t)

                        per_fold_metrics_rows.append({
                            "window_size": WINDOW_TO_PROCESS,
                            "trained_task": task_name,
                            "eval_task": t,
                            "model_name": model_name,
                            "params_id": params_id,
                            "fold_idx": fold_idx,
                            **m_t,
                        })

                # ------------------- CALIBRATION (GLOBAL ONLY) -------------------
                # only for global model eval (all_tasks), requires predict_proba
                if (
                    (not TASK_SPECIFIC)
                    and task_name == "all_tasks"
                    and y_proba_all is not None
                ):
                    acc_bins, conf_bins, cnt_bins, edges = reliability_bins_from_proba(
                        y_test, y_proba_all, n_bins=CAL_N_BINS
                    )
                    for b in range(CAL_N_BINS):
                        cal_rows.append({
                            "trained_task": task_name,
                            "model_name": model_name,
                            "params_id": params_id,
                            "fold_idx": fold_idx,
                            "bin_idx": b,
                            "bin_left": float(edges[b]),
                            "bin_right": float(edges[b + 1]),
                            "count": int(cnt_bins[b]),
                            "mean_confidence": float(conf_bins[b]) if not np.isnan(conf_bins[b]) else np.nan,
                            "accuracy_in_bin": float(acc_bins[b]) if not np.isnan(acc_bins[b]) else np.nan,
                        })

                # ------------------- EXPLAINABILITY -------------------
                def run_explainability(eval_task_label, df_train_sub, df_test_sub):
                    if df_train_sub.empty or df_test_sub.empty:
                        return

                    # scale subset
                    Xtr = scaler.transform(df_train_sub[feat_order].values)
                    ytr = df_train_sub["Label"].values
                    Xte = scaler.transform(df_test_sub[feat_order].values)
                    yte = df_test_sub["Label"].values

                    # skip if degenerate (1 class)
                    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
                        return

                    X_bg, _ = safe_balanced_subsample(Xtr, ytr, n_total=BG_N, random_state=RANDOM_STATE)
                    X_expl, y_expl = safe_balanced_subsample(Xte, yte, n_total=EXPL_N, random_state=RANDOM_STATE)
                    if X_bg.shape[0] == 0 or X_expl.shape[0] == 0:
                        return

                    # --- PI ---
                    if not skip_perm:
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

                    # --- SHAP ---
                    if not skip_shap:
                        try:
                            if model_name == "LASSO_LR":
                                explainer = shap.LinearExplainer(model, X_bg)
                                shap_vals = explainer.shap_values(X_expl)
                            else:
                                explainer = shap.KernelExplainer(model.predict_proba, X_bg)
                                shap_vals = explainer.shap_values(X_expl, nsamples=KERNEL_NSAMPLES)

                            sv3 = shap_to_3d(shap_vals)            # (n, f, c)
                            imp_fc = np.mean(np.abs(sv3), axis=0)  # (f, c)
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

                # (B) per-task explainability using SAME all_tasks model
                if (not TASK_SPECIFIC) and task_name == "all_tasks":
                    for t in eval_tasks:
                        df_train_t = df_train[df_train["task"] == t]
                        df_test_t  = df_test[df_test["task"] == t]
                        run_explainability(t, df_train_t, df_test_t)

            # ---------------- SAVE METRICS (ALL) ----------------
            df_m = pd.DataFrame(per_fold_metrics_rows)
            if not df_m.empty:
                df_m.to_csv(metrics_per_fold_path, index=False)
                print(f"[Saved] {metrics_per_fold_path}")

                metric_cols = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
                save_agg_mean_std(
                    df_m,
                    group_cols=["trained_task", "eval_task", "model_name", "params_id"],
                    value_cols=metric_cols,
                    out_path=metrics_agg_path
                )
                print(f"[Saved] {metrics_agg_path}")

            # ---------------- SAVE CALIBRATION (GLOBAL ONLY) ----------------
            if cal_rows:
                df_cal = pd.DataFrame(cal_rows)
                df_cal.to_csv(cal_csv_path, index=False)
                print(f"[Saved] {cal_csv_path}")

                # Plot: per-fold curves + mean curve
                plt.figure(figsize=(6.5, 5.5))
                plt.plot([0, 1], [0, 1], "--", label="Ideal")

                # per-fold curves
                for fold_idx in sorted(df_cal["fold_idx"].unique()):
                    dff = df_cal[df_cal["fold_idx"] == fold_idx].sort_values("bin_idx")
                    x = dff["mean_confidence"].to_numpy()
                    y = dff["accuracy_in_bin"].to_numpy()
                    m = (~np.isnan(x)) & (~np.isnan(y))
                    if m.sum() >= 2:
                        plt.plot(x[m], y[m], marker="o", linewidth=1.0, alpha=0.35)

                # mean curve across folds per bin (ignore NaNs)
                mean_by_bin = (
                    df_cal.groupby("bin_idx")[["mean_confidence", "accuracy_in_bin"]]
                          .mean(numeric_only=True)
                          .reset_index()
                          .sort_values("bin_idx")
                )
                x = mean_by_bin["mean_confidence"].to_numpy()
                y = mean_by_bin["accuracy_in_bin"].to_numpy()
                m = (~np.isnan(x)) & (~np.isnan(y))
                if m.sum() >= 2:
                    plt.plot(x[m], y[m], marker="o", linewidth=2.5, alpha=0.95, label="Mean (across folds)")

                plt.xlabel("Confidence")
                plt.ylabel("Accuracy")
                plt.title(f"Reliability (confidence) - {task_name} / {model_name}")
                plt.grid(True, alpha=0.25)
                plt.legend()
                plt.tight_layout()
                plt.savefig(cal_fig_path, dpi=300, bbox_inches="tight")
                print(f"[Saved] {cal_fig_path}")
                plt.show()

            # ---------------- SAVE PERM ----------------
            if not skip_perm and per_fold_perm_rows is not None:
                df_perm = pd.DataFrame(per_fold_perm_rows)
                if not df_perm.empty:
                    df_perm.to_csv(perm_per_fold_path, index=False)
                    print(f"[Saved] {perm_per_fold_path}")

                    save_agg_mean_std(
                        df_perm,
                        group_cols=["trained_task", "eval_task", "model_name", "feature"],
                        value_cols=["perm_importance_mean"],
                        out_path=perm_agg_path
                    )
                    print(f"[Saved] {perm_agg_path}")

            # ---------------- SAVE SHAP ----------------
            if not skip_shap and per_fold_shap_rows is not None:
                df_shap = pd.DataFrame(per_fold_shap_rows)
                if not df_shap.empty:
                    df_shap.to_csv(shap_per_fold_path, index=False)
                    print(f"[Saved] {shap_per_fold_path}")

                    save_agg_mean_std(
                        df_shap,
                        group_cols=["trained_task", "eval_task", "model_name", "feature", "class"],
                        value_cols=["mean_abs_shap"],
                        out_path=shap_agg_path
                    )
                    print(f"[Saved] {shap_agg_path}")


if __name__ == "__main__":
    main()
