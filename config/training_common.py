import numpy as np
from config.paths import (
    FEATURES_PATH,
    PLOTS_PATH,
    IMPORTANCE_PATH,
    SHAP_PATH,
    ensure_dirs,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# here define feature column used as class label
TARGET = "condition"

MODELS_TO_USE = [
    "MLP",
    "RF", 
    "SVM", 
    "KNN", 
    "XGBoost", 
    "LightGBM", 
    "LASSO_LR"
]

PARAM_GRIDS = {
        "RF": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "SVM": {"C": [0.1, 1], "kernel": ["linear", "rbf"]},
        "KNN": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
        "MLP": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"]},
        "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
        "LightGBM": {"n_estimators": [100, 200], "max_depth": [5, 10], "learning_rate": [0.01, 0.1]},
        "LASSO_LR": {"C": [ 0.1, 1]},
    }

SENSORS = ["arm_l", "arm_r", "wrist_l", "wrist_r", "trunk"]
TIME_FEATURES = ["MAX", "MIN", "AMP", "MEAN", "JERK", "RMS", "COR", "STD"]
FREQ_FEATURES = ["DOMFREQ", "DOMPOW", "TOTPOW", "SPEC_CENT", "SPEC_SPREAD"]

EXCLUDE_QUAT = False
EXCLUDE_ACC = False
EXCLUDE_GYRO = False
EXCLUDE_MAG = False

ensure_dirs(PLOTS_PATH, IMPORTANCE_PATH, SHAP_PATH)
