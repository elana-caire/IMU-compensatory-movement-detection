from pathlib import Path

BASE_DIR = Path(r"/Users/sorresoayarey/Desktop/relab/")

RESULTS_PATH = BASE_DIR / "Results"

RAW_DATA_PATH = BASE_DIR / "Data/IMU Data"
FEATURES_PATH = BASE_DIR / "Data/Features"

PLOTS_PATH = RESULTS_PATH / "Plots"
IMPORTANCE_PATH = RESULTS_PATH / "FeatureImportance"
SHAP_PATH = RESULTS_PATH / "SHAP"
MODELS_PATH = RESULTS_PATH / "Models"
BEST_MODELS_PATH = MODELS_PATH / "BestModels"

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
