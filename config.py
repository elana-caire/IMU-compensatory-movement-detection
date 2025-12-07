import os
from pathlib import Path

# --------------------------------------------
# Global configuration for both scripts
# --------------------------------------------

# --------- Data Preparation Parameters ---------
WINDOW_SIZE_MS = [None, 250, 500, 750, 1000]

# Raw IMU data path
RAW_DATA_PATH = r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\IMU Data"

# Where to save extracted feature CSV files
FEATURE_SAVE_PATH = r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\Features"

# --------- Model Training Parameters ---------
CONFIG = {
    "features_folder": r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\Features",
    "plots_folder": r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\Plots",
    "importance_folder": r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\FeatureImportance",
    "shap_folder": r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\SHAP",
    "models_folder": r"C:\Users\giusy\OneDrive\Desktop\AI_Healtcare\IMU-compensatory-movement-detection\Data\Models",

    "target": "condition",

    "models_to_use": [
        #"RF", 
        #"SVM", 
        #"KNN", 
        "MLP",
        "XGBoost", 
        #"LightGBM", 
        "LASSO_LR"
    ],

    "param_grids": {
        "RF": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "SVM": {"C": [0.1, 1], "kernel": ["linear", "rbf"]},
        "KNN": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
        "MLP": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"]},
        "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
        "LightGBM": {"n_estimators": [100, 200], "max_depth": [5, 10], "learning_rate": [0.01, 0.1]},
        "LASSO_LR": {"C": [0.01, 0.1, 1, 10]},
    },

    "n_bootstrap": 500,
    "top_n_features": 30,

    # SHAP settings
    "shap_sample_size": 500,

    # We can modify these lists to select specific sensors and features
    "sensors_to_consider":  ["arm_l", "arm_r", "wrist_l", "wrist_r", "trunk"], 
    "time_features": ["MAX", "MIN", "AMP", "MEAN", "JERK", "RMS", "COR", "STD"],
    "frequency_features": [],
    #"frequency_features": ["DOMFREQ", "DOMPOW", "TOTPOW", "SPEC_CENT", "SPEC_SPREAD"],
    # We can decide to omit specific modalities from a given sensor (e.g: only IMU)
    "exclude_quat": True,
    "exclude_acc": False,
    "exclude_gyro": False,
    "exclude_mag": True,
}



# ------------------------------------------------
# AUTOMATIC FOLDER CREATION (executed on import)
# ------------------------------------------------

def ensure_directories():
    """Create all necessary output folders if they do not exist."""
    folders = [
        FEATURE_SAVE_PATH,
        CONFIG["features_folder"],
        CONFIG["plots_folder"],
        CONFIG["importance_folder"],
        CONFIG["shap_folder"],
    ]
    for folder in folders:
        
        Path(folder).mkdir(parents=True, exist_ok=True)

# Run automatically
ensure_directories()
