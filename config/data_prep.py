from config.paths import RAW_DATA_PATH, FEATURES_PATH, ensure_dirs

# Define window sizes in milliseconds
WINDOW_SIZE_MS = [None, 250, 500, 750, 1000]  
TASK_NAMES = ["cup-placing", "peg", "wiping", "pouring"]
CONDITION_NAMES = ["natural", "elbow_brace", "elbow_wrist_brace"]

ensure_dirs(FEATURES_PATH)
