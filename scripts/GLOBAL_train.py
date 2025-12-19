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

import config.training_common as training_config
import config.time_split as time_split_config
import config.paths as paths_config

RANDOM_STATE = training_config.RANDOM_STATE
PLOTS_DIR = paths_config.PLOTS_PATH
MODELS_DIR = paths_config.MODELS_PATH
FEATURES_DIR = paths_config.FEATURES_PATH
RESULTS_DIR = paths_config.RESULTS_PATH

window_to_process = time_split_config.WINDOW_TO_PROCESS
TARGET = training_config.TARGET
MODELS = training_config.MODELS_TO_USE
SENSORS = training_config.SENSORS
TIME_FEATURES = training_config.TIME_FEATURES
FREQ_FEATURES = training_config.FREQ_FEATURES
EXCLUDE_QUAT = training_config.EXCLUDE_QUAT
EXCLUDE_ACC = training_config.EXCLUDE_ACC
EXCLUDE_GYRO = training_config.EXCLUDE_GYRO
EXCLUDE_MAG = training_config.EXCLUDE_MAG
MODELS_TO_USE = training_config.MODELS_TO_USE
PARAM_GRIDS = training_config.PARAM_GRIDS

TASK_SPECIFIC = time_split_config.TASK_SPECIFIC

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
        "LASSO_LR": LogisticRegression(
            penalty='l1',
            solver='saga',
            multi_class='multinomial',
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }
    model = base_models[name]
    if params:
        model.set_params(**params)
    return model


# --------------------------- MAIN -----------------------------

def main():
    models_root = Path(MODELS_DIR) 

    # ---------- Load data ----------
    df = pd.read_csv(Path(FEATURES_DIR) / f"features_win_{window_to_process}.csv")
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
        exclude_quat=EXCLUDE_QUAT
    )

    # ---------- Result files ----------
    results_folds_path = models_root / "model_results_Global_folds.csv" 
    results_agg_path   = models_root / "model_results_Global_agg.csv"   

    if results_folds_path.exists():
        df_folds = pd.read_csv(results_folds_path)
        print("[INFO] Found existing per-fold results -> will append/skip as needed.")
    else:
        df_folds = pd.DataFrame()

    # -------------------- DATA SELECTION --------------------
    if TASK_SPECIFIC:
        tasks_to_run = list(df["task"].unique())
    else:
        tasks_to_run = ["all_tasks"]  

    for task_name in tasks_to_run:

        if TASK_SPECIFIC:
            df_task = df[df["task"] == task_name].copy()
            folds = create_temporal_train_test_folds(df_task)
        else:
            df_task = df.copy()
            folds = create_temporal_train_test_folds(df_task)

        for model_name in MODELS_TO_USE:

            model_folder = models_root / task_name / model_name  
            model_folder.mkdir(parents=True, exist_ok=True)

            all_rows_this_model = []  # collect new rows for df_folds

            for params in ParameterGrid(PARAM_GRIDS[model_name]):

                pid = params_to_id(params) 
                params_folder = model_folder / pid
                params_folder.mkdir(parents=True, exist_ok=True)

                for fold_idx, df_train, df_test in folds:

                    model_file = params_folder / f"fold_{fold_idx}.joblib" 

                    # ---------- SKIP if already trained ----------
                    if model_file.exists(): 
                        print(f"[SKIP] exists: {model_file}")
                        continue

                    print(f"Training task={task_name} model={model_name} params={params} fold={fold_idx}")

                    # train
                    model = create_model(model_name, params)
                    scaler = StandardScaler()

                    X_train = scaler.fit_transform(df_train[features].values)
                    y_train = df_train["Label"].values
                    X_test  = scaler.transform(df_test[features].values)
                    y_test  = df_test["Label"].values

                    model.fit(X_train, y_train)

                    # eval
                    y_pred  = model.predict(X_test)
                    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    m = compute_metrics(y_test, y_pred, y_proba)

                    # ---------- store per-fold row ----------
                    row = {
                        "window_size": window_to_process,
                        "task": task_name,
                        "model_name": model_name,
                        "params_id": pid,                
                        "best_params": json.dumps(params, sort_keys=True),  
                        "fold_idx": fold_idx,           
                        **m
                    }
                    all_rows_this_model.append(row)

                    # ---------- save model bundle ----------
                    joblib.dump(
                        {
                            "model": model,
                            "scaler": scaler,
                            "features": features,
                            "model_name": model_name,
                            "task": task_name,
                            "window_size": window_to_process,
                            "params": params,
                            "label_encoder": label_encoder
                        },
                        model_file
                    )
                    print(f"[Saved] {model_file}")

            # ---------- UPDATE per-fold CSV (append + de-dup) ----------
            if all_rows_this_model:  
                df_new = pd.DataFrame(all_rows_this_model)
                key_cols = ["window_size", "task", "model_name", "params_id", "fold_idx"] 
                df_folds = upsert_rows(df_folds, df_new, key_cols=key_cols) 
                df_folds.to_csv(results_folds_path, index=False)
                print(f"[Saved folds] {results_folds_path}")

            # ---------- AGGREGATE mean/std across folds ----------
            if not df_folds.empty: 
                metric_cols = ["f1_macro", "accuracy", "precision_macro", "recall_macro", "roc_auc"]
                df_agg = (
                    df_folds
                    .groupby(["window_size", "task", "model_name", "params_id", "best_params"], as_index=False)[metric_cols]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                # flatten columns
                df_agg.columns = [
                    f"{a}_{b}" if b else a
                    for a, b in df_agg.columns.to_flat_index()
                ]

                df_agg.to_csv(results_agg_path, index=False)
                print(f"[Saved agg] {results_agg_path}")


if __name__ == '__main__':
    main()



    