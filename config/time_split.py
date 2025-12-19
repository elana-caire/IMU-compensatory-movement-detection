from pathlib import Path
from config.training_common import RANDOM_STATE

WINDOW_TO_PROCESS = 750

TASK_SPECIFIC = True

# Models to explain
BEST_MODELS_FOR_EXPLAINABILITY = ["LASSO_LR", "MLP"]

# SHAP
BG_N = 100
EXPL_N = 50
KERNEL_NSAMPLES = 200

# Permutation importance
PERM_N_REPEATS = 10
PERM_SCORER = "f1_macro"

# Calibration
CAL_N_BINS = 10

MODELS_TO_EXPLAIN = ["MLP", "LASSO_LR"]
