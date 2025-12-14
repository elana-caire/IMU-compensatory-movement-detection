import os, glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score

# -------------------- CONFIG --------------------
SAVE_DIR = r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\ModelsFinal"
MODEL_NAMES = ["LASSO_LR", "MLP"]

# If your task-agnostic models are stored under one of these task folder names:
TASK_AGNOSTIC_TASK_NAMES = ["all_task", "global"]   # adjust if needed

# calibration
N_BINS = 15

# plotting
FIG_W = 22
FIG_H = 7
TITLE_FONTSIZE  = 14
LABEL_FONTSIZE  = 12
TICK_FONTSIZE   = 10
LEGEND_FONTSIZE = 11

# Use a fixed task order if you want:
TASK_ORDER = ["cup-placing", "peg", "wiping", "pouring"]  # adjust to your exact task names
# -----------------------------------------------


def find_feature_file_for_window(features_folder, window_size="750"):
    cand = glob.glob(os.path.join(features_folder, f"*{window_size}*.csv"))
    if len(cand) == 0:
        raise FileNotFoundError(f"Could not find feature CSV for window={window_size} in {features_folder}")
    return cand[0]


def load_dataset_750():
    from config import CONFIG
    from utils.ml import encode_labels, return_feature_columns

    feature_csv = find_feature_file_for_window(CONFIG["features_folder"], window_size="750")
    df = pd.read_csv(feature_csv)
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
    return df, features, label_encoder


# --------- YOUR RELIABILITY DEFINITION, but with FIXED bins ----------
def reliability_fixed_bins(y_true, proba, n_bins=15):
    """
    Multiclass reliability:
      preds = argmax(proba)
      conf  = max(proba)
      correct = (preds == y_true)
    Then bin by conf in [0,1] with fixed bins, return per-bin:
      acc_bin[b]  = mean(correct in bin)
      conf_bin[b] = mean(conf in bin)

    We also return bin_centers to always show the full x-range.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    preds = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    correct = (preds == y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    acc_bin = np.full(n_bins, np.nan, dtype=float)
    conf_bin = np.full(n_bins, np.nan, dtype=float)

    for b in range(n_bins):
        if b < n_bins - 1:
            m = (conf >= bins[b]) & (conf < bins[b + 1])
        else:
            m = (conf >= bins[b]) & (conf <= bins[b + 1])

        if m.sum() > 0:
            acc_bin[b] = correct[m].mean()
            conf_bin[b] = conf[m].mean()

    return bin_centers, acc_bin, conf_bin


def list_bundles_for_model(save_dir, model_name):
    """
    Load ALL bundles for that model from saved_models/*/{model}/...joblib.
    Returns list of bundles.
    """
    pattern = os.path.join(save_dir, "saved_models", "*", model_name, f"{model_name}_task_*_test_subject_*.joblib")
    paths = sorted(glob.glob(pattern))
    bundles = []
    for p in paths:
        try:
            bundles.append(joblib.load(p))
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    return bundles


def build_bundle_maps(bundles):
    """
    Create:
      ts_map[(task, subject)] = bundle   for task-specific
      ta_map[subject] = bundle           for task-agnostic (task in TASK_AGNOSTIC_TASK_NAMES)
    """
    ts_map = {}
    ta_map = {}

    for b in bundles:
        task = str(b.get("task", "")).strip()
        subj = b.get("test_subject", None)
        if subj is None:
            continue

        if task.lower() in [t.lower() for t in TASK_AGNOSTIC_TASK_NAMES]:
            # task-agnostic
            ta_map[subj] = b
        else:
            # task-specific
            ts_map[(task, subj)] = b

    return ts_map, ta_map


def eval_bundle_on_task(bundle, df, task_name, subject):
    """
    Evaluate the saved bundle on a specific (task, subject) subset.
    Returns metrics + proba (for calibration).
    """
    model  = bundle["model"]
    scaler = bundle["scaler"]
    feats  = bundle["features"]

    dft = df[(df["task"] == task_name) & (df["subject"] == subject)]
    if dft.empty:
        return None

    X = dft[feats].values
    y = dft["Label"].values
    Xs = scaler.transform(X)

    y_pred = model.predict(Xs)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro")

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xs)
        except Exception:
            proba = None

    return {"accuracy": float(acc), "f1_macro": float(f1), "y_true": y, "proba": proba, "n": int(len(y))}


def eval_bundle_on_subject_all_tasks(bundle, df, subject):
    """
    Evaluate the task-agnostic bundle on *all tasks combined* for that held-out subject.
    """
    model  = bundle["model"]
    scaler = bundle["scaler"]
    feats  = bundle["features"]

    dft = df[df["subject"] == subject]
    if dft.empty:
        return None

    X = dft[feats].values
    y = dft["Label"].values
    Xs = scaler.transform(X)

    y_pred = model.predict(Xs)
    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, average="macro")

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(Xs)
        except Exception:
            proba = None

    return {"accuracy": float(acc), "f1_macro": float(f1), "y_true": y, "proba": proba, "n": int(len(y))}


def boxplot_with_jitter(ax, data, title):
    """
    Matplotlib-only boxplot + jittered points.
    data: list/array of values (one group).
    """
    data = np.asarray(data, dtype=float)
    ax.boxplot([data], positions=[1], widths=0.55, showfliers=True, patch_artist=True)
    rng = np.random.default_rng(42)
    xj = 1 + rng.normal(0, 0.05, size=len(data))
    ax.scatter(xj, data, s=28, alpha=0.85)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xticks([1])
    ax.set_xticklabels([""], fontsize=TICK_FONTSIZE)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def plot_calibration_collapsed(ax, per_subject_curves, title):
    """
    per_subject_curves: list of (bin_centers, acc_bin, conf_bin)
    Collapse across subjects by mean±std of acc_bin (ignoring NaNs), plot vs bin_centers.
    """
    ax.plot([0, 1], [0, 1], "--", linewidth=1, label="Ideal")

    if len(per_subject_curves) == 0:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "No proba", ha="center", va="center", transform=ax.transAxes)
        return

    # stack acc over subjects on fixed bins
    bin_centers = per_subject_curves[0][0]
    acc_stack = np.vstack([c[1] for c in per_subject_curves])  # (n_subj, n_bins)

    mean_acc = np.nanmean(acc_stack, axis=0)
    std_acc  = np.nanstd(acc_stack, axis=0)

    ok = ~np.isnan(mean_acc)
    ax.plot(bin_centers[ok], mean_acc[ok], marker="o", linewidth=2, label="Mean")
    ax.fill_between(bin_centers[ok], (mean_acc-std_acc)[ok], (mean_acc+std_acc)[ok], alpha=0.2, label="±1 std")

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


# -------------------- RUN --------------------
df, _, _ = load_dataset_750()

# enforce 4 tasks (task-specific columns)
tasks = TASK_ORDER.copy()
# if your df tasks differ, fallback to df unique:
df_tasks = sorted(df["task"].unique())
if not set(tasks).issubset(set(df_tasks)):
    tasks = df_tasks[:4]

columns = tasks + ["task_agnostic"]  # 5 columns total

for model_name in MODEL_NAMES:
    bundles = list_bundles_for_model(SAVE_DIR, model_name)
    ts_map, ta_map = build_bundle_maps(bundles)

    subjects = sorted(df["subject"].unique())

    # collect accuracies + calibration curves
    acc_by_col = {c: [] for c in columns}
    calib_by_col = {c: [] for c in columns}  # list of per-subject curves

    for subj in subjects:
        # ---- task-specific: one model per task ----
        for t in tasks:
            b = ts_map.get((t, subj), None)
            if b is None:
                continue
            out = eval_bundle_on_task(b, df, task_name=t, subject=subj)
            if out is None:
                continue

            acc_by_col[t].append(out["accuracy"])

            if out["proba"] is not None:
                bc, acc_bin, conf_bin = reliability_fixed_bins(out["y_true"], out["proba"], n_bins=N_BINS)
                calib_by_col[t].append((bc, acc_bin, conf_bin))

        # ---- task-agnostic: single model, evaluated on all tasks pooled ----
        bta = ta_map.get(subj, None)
        if bta is not None:
            out = eval_bundle_on_subject_all_tasks(bta, df, subject=subj)
            if out is not None:
                acc_by_col["task_agnostic"].append(out["accuracy"])
                if out["proba"] is not None:
                    bc, acc_bin, conf_bin = reliability_fixed_bins(out["y_true"], out["proba"], n_bins=N_BINS)
                    calib_by_col["task_agnostic"].append((bc, acc_bin, conf_bin))

    # -------------------- PLOT: 2 rows x 5 cols --------------------
    fig, axes = plt.subplots(2, 5, figsize=(FIG_W, FIG_H), sharey="row")

    # Row 1: accuracy boxplots
    for j, col in enumerate(columns):
        ax = axes[0, j]
        vals = acc_by_col[col]
        if len(vals) == 0:
            ax.axis("off")
            continue
        title = col if col != "task_agnostic" else "task-agnostic"
        boxplot_with_jitter(ax, vals, title=title)
        if j == 0:
            ax.set_ylabel("Accuracy\n(across subjects)", fontsize=LABEL_FONTSIZE)

    # Row 2: calibration curves (collapsed)
    for j, col in enumerate(columns):
        ax = axes[1, j]
        title = col if col != "task_agnostic" else "task-agnostic"
        plot_calibration_collapsed(ax, calib_by_col[col], title=title)
        ax.set_xlabel("Confidence bin center", fontsize=LABEL_FONTSIZE)
        if j == 0:
            ax.set_ylabel("Empirical accuracy", fontsize=LABEL_FONTSIZE)

    # shared legend (bottom)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=LEGEND_FONTSIZE, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"{model_name} — Accuracy distributions + Reliability (5 columns)", fontsize=TITLE_FONTSIZE + 2)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()
