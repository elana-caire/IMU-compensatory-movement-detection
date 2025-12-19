# IMU_Compensatory_Movements

Pipeline-based framework for detecting compensatory movements from IMU data using feature extraction, LOSO training, time-split training, and post-hoc explainability (SHAP & permutation importance).

## 🧰 Setup Instructions

1. Clone the repository

git clone [<repository_url>](https://github.com/elana-caire/IMU-compensatory-movement-detection.git)

cd IMU_Compensatory_Movements

2. Create the Conda environment

This project uses a predefined Conda environment.

```bash
conda env create -f environment.yml
conda activate AI_health
```

⚠️ Make sure the environment name is AI_health

3. Download the Dataset

Download the dataset from the Polybox folder.

Unzip it.

Rename the folder Course Data → Data

Place the Data/ folder in the root directory of this project:

IMU_Compensatory_Movements/
├── Data/
│   ├── IMU Data/
│   ├── Features/
│   └── ...
📁 Project Structure (Simplified)


IMU_Compensatory_Movements/
│
├── main.py                  # Pipeline entry point
│
├── scripts/                 # Executable pipeline steps
│   ├── data_preparation.py
│   ├── LOSO_task_agnostic_train.py
│   ├── LOSO_task_specific_train.py
│   ├── plot_average.py
│   ├── LOSO_feature_importance.py
│   ├── GLOBAL_train.py
│   └── GLOBAL_feature_importance.py
│
├── config/                  # All configuration files
│   ├── paths.py
│   ├── data_prep.py
│   ├── training_common.py
│   ├── loso.py
│   └── global_time_split.py
│
├── utils/                   # Feature extraction & helpers
├── environment.yml
└── README.md


## 🚀 Running the Full Pipeline (Recommended)

All steps are executed in the correct order using the main pipeline controller.

```bash
python main.py
```

This will run:

Data preparation & feature extraction

--> LOSO 

training

Task-agnostic models

Task-specific models

Average performance plots

Feature importance & SHAP

--> Global 

pick in config/time_split if you run task agnostic or task specific

(time-split) training 

feature importance & SHAP


## 🧰 Pipeline Steps (What Happens Internally)

### 🔹 STEP 1 – Data Preparation

Script: scripts/data_preparation.py

Config: config/data_prep.py, config/paths.py

This step:

Loads raw IMU data

Filters signals and aligns them to movement onset

Extracts window-based features

Saves feature CSV files to Data/Features/

Key parameters (edit in config/data_prep.py)

WINDOW_SIZE_MS = [750]  # e.g. [None, 250, 500, 750, 1000]

Raw and output paths are defined centrally in:

config/paths.py


### 🔹 STEP 2 – LOSO Model Training

2a – Task-Agnostic LOSO Training

Script: scripts/LOSO_task_agnostic_train.py

Config:

config/training_common.py

config/loso.py

Trains models across all tasks combined using Leave-One-Subject-Out CV.

2b – Task-Specific LOSO Training

Script: scripts/LOSO_task_specific_train.py

Trains separate LOSO models per task.

2c – Average Performance Plots

Script: scripts/plot_average.py

Generates summary plots averaged across tasks and subjects.

2d – Feature Importance & SHAP (LOSO)

Script: scripts/LOSO_feature_importance.py

Retrains top-performing LOSO models and computes:

Permutation Importance

SHAP values

Key parameters live in:

config/loso.py

### 🔹 STEP 3 – Global (Time-Split) Training

3a – Global Training

Script: scripts/GLOBAL_train.py

Config: config/global_time_split.py

Uses temporal splits instead of LOSO.

3b – Global Feature Importance & SHAP

Script: scripts/GLOBAL_feature_importance.py

Computes explainability metrics for global models.

### ⚙️ Configuration Philosophy (Important)

❌ No parameters are edited inside scripts

✅ All parameters live in config/

✅ Paths are centralized in config/paths.py

✅ Scripts only import what they need

This ensures:

Reproducibility

Clean experiments

Easy review and modification

## 📊 Feature File Description

Example output file:

Data/Features/features_win_750.csv

Each row corresponds to one window from one:

subject

task

condition

Feature types

Time-domain

*_MAX, *_MIN, *_AMP, *_MEAN, *_RMS, *_STD, *_JERK, *_COR

Frequency-domain

*_DOMFREQ, *_DOMPOW, *_TOTPOW, *_SPEC_CENT, *_SPEC_SPREAD

Metadata

subject

task

condition

🧪 Example Usage (Loading Features)

import pandas as pd

feats = pd.read_csv("Data/Features/features_win_750.csv")

### Select subject

subj_feat = feats[feats["subject"] == "P02"]

### Select task
subj_task = subj_feat[subj_feat["task"] == "cup-placing"]


### ⚠️ Important Notes

Do not run scripts directly

❌ python scripts/data_preparation.py

Always use:

✅ python main.py

This ensures correct imports and reproducible execution.