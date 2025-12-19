from collections import Counter
from pathlib import Path
from tqdm import tqdm

import pandas as pd

import utils.filtering as flt
from utils.feature_extraction import *
from utils.load_data import *
from utils.filtering import *
from utils.movement_onset import *

from config.data_prep import WINDOW_SIZE_MS
from config.paths import RAW_DATA_PATH, FEATURES_PATH
import config.data_prep as data_prep_config

WINDOW_SIZE_MS = data_prep_config.WINDOW_SIZE_MS
TASK_NAMES = data_prep_config.TASK_NAMES
CONDITION_NAMES = data_prep_config.CONDITION_NAMES

if __name__ == '__main__':

    for window_size in WINDOW_SIZE_MS:

        if window_size is not None:
            saving_path = f"{FEATURES_PATH}/features_win_{window_size}.csv"
        else:
            saving_path = f"{FEATURES_PATH}/features.csv"

        if Path(saving_path).exists():
            print(f"Feature file {saving_path} already exists, skipping.")
            continue

        # Load all subjects
        data = load_all_subjects(RAW_DATA_PATH)

        # Build raw dataset
        all_subjects, dataset = build_raw_dataset(
            data, tasks=TASK_NAMES, conditions=CONDITION_NAMES
        )

        all_subjects_preproc = []

        for item in all_subjects:
            df_raw = item['data']

            df_filt = filter_butterworth(df_raw, fs=60, cutoff=2)
            df_cut = aling_to_movement_onset(df_filt, plot=False,
                                             metadata=[item['subject'], item['task'], item['condition']])

            print(f"Extracting features (window = {window_size})")
            df_feat = extract_all_features(df_cut, window_ms=window_size)

            df_feat["task"] = item["task"]
            df_feat["condition"] = item["condition"]

            all_subjects_preproc.append({
                "subject": item["subject"],
                "task": item["task"],
                "condition": item["condition"],
                "data": df_cut,
                "features": df_feat
            })

        # Aggregate all features
        all_rows = []
        for item in all_subjects_preproc:
            df_feat = item["features"].copy()
            df_feat["subject"] = item["subject"]
            df_feat["task"] = item["task"]
            df_feat["condition"] = item["condition"]
            all_rows.append(df_feat)

        all_feats = pd.concat(all_rows, axis=0)

        all_feats.to_csv(saving_path, index=True)
        print(f"Saved: {saving_path}")
