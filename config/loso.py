from config.training_common import RANDOM_STATE

# Task handling
BEST_MODEL = "task_specific" # or "task_agnostic"

# Permutation importance
COMPUTE_PERMUTATION_IMPORTANCE = False
PERM_N_REPEATS = 10
PERM_SCORING = "f1_macro"

# SHAP
COMPUTE_SHAP = True
SHAP_SAMPLE_SIZE = 500
ALLOW_KERNEL_SHAP = False

# Models to explain
BEST_MODELS_FOR_EXPLAINABILITY = ["LASSO_LR", "MLP"]

# Bootstrap
N_BOOTSTRAP_SAMPLES = 50
CI = 95
# to limit the hyperparameter search space
USE_RANDOM_SEARCH = True
MAX_INNER_EVALS = 50