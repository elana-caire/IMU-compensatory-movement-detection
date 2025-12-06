import os
from pathlib import Path

# --------------------------------------------
# Global configuration for both scripts
# --------------------------------------------

# --------- Data Preparation Parameters ---------
WINDOW_SIZE_MS = [None, 250, 500, 750, 1000]

# Raw IMU data path
RAW_DATA_PATH = "/Users/sorresoayarey/Desktop/relab/Data/IMU Data"

# Where to save extracted feature CSV files
FEATURE_SAVE_PATH = "/Users/sorresoayarey/Desktop/relab/Data/Features"


# --------- Model Training Parameters ---------
CONFIG = {
    "features_folder": "/storage/elana/relab/Data/Features",
    "plots_folder": "/storage/elana/relab/Data/Plots",
    "importance_folder": "/storage/elana/relab/Data/Plots/FeatureImportance",
    "shap_folder": "/storage/elana/relab/Data/Plots/SHAP",

    "target": "condition",

    "models_to_use": [
        "RF", "SVM", "KNN", "MLP",
        "XGBoost", "LightGBM", "LASSO_LR"
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
        CONFIG["shap_folder"]
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

# Run automatically
ensure_directories()
