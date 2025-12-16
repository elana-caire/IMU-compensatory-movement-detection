# IMU_Compensatory_Movements

Git repo for IMU Compensatory Movements Project.


## 🧰 Setup Instructions

### 1. Clone the repository

```bash
git clone <repository_url>
cd <imu_compensatory_movements>
```

### 2. Create a new conda environment
```bash
conda create -n ai_health python=3.12.0
conda activate ai_health
```

### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```


### 4. Download the Dataset

1. Download the dataset from the Polybox folder and unzip it.  
2. Rename the folder **"Course Data"** to **`Data`**.  
3. Place the `Data` folder in the **root directory** of this project.  


## 🧰 Prepare The Data
```bash
python data_preparation.py
```


This script:
1. Loads raw IMU data.
2. Aligns data for each trial -condition (arm_r, arm_l, wrist_r, wrist_l, trunk) sensors
2. Filters and aligns each trial to movement onset.
3. Extracts window-based features.
4. Saves all features to a single CSV.

---

### What to change

```python
WINDOW_SIZE_MS = [None, 250, 500, 750, 1000]  # feature window size in ms. Here, select the desired window size

path = r"...\\Data\\IMU Data"        # folder with raw IMU files
save_path = r"...\\Data\\Features"   # folder where the CSV will be saved
```

- **WINDOW_SIZE_MS**: window length for feature extraction (ms).
- **path**: root directory of raw IMU data.
- **save_path**: output directory (must exist).

---

### What is saved

After running the script:

Example: `features_win_500.csv`

Each **row** = one time window from one (subject, task, condition).

Columns include:
- Per-signal time-domain features:  
  `*_MAX`, `*_MIN`, `*_AMP`, `*_MEAN`, `*_RMS`, `*_STD`, `*_JERK`, `*_COR`
- Per-signal spectral features:  
  `*_DOMFREQ`, `*_DOMPOW`, `*_TOTPOW`, `*_SPEC_CENT`, `*_SPEC_SPREAD`
- Metadata:  
  `subject`, `task`, `condition`

Tasks: 'cup-placing', 'peg', 'pouring', 'wiping'
Conditions: 'normal', 'elbow_brace', 'elbow_wrist_brace'

Index = window start (sample index aligned to movement onset).

### Example of usage

```python
# load features
feats_all = pd.read_csv(r"...\Features\features_win_500.csv")
# To select one subject 
subj_feat = feats_all[feats_all['subject'] == 'P02']
# To select a specific task from the subject
subj_task = subj_feat[subj_feat['task'] == 'cup-placing']
```

## Train models with LOSO configuration

## 🧰 Run Task-Specific Models
```bash
python3 train_model_task_specific.py
```


## Train models with temporal splitting configuration

Set the parameter "task_specific" to true or false to pick between task specific and task agnostic models in the config.py file

```bash
python3 GLOBAL_trainings.py
```

## Post-hoc Explainability & Re-Training Script on top performing models

## LOSO configuration 

This script re-trains top-performing models using Leave-One-Subject-Out (LOSO) cross-validation and computes post-hoc explainability metrics using:
SHAP values (feature attribution per class)
Permutation Importance (model-agnostic feature importance)
It is intended to be run after model selection and hyperparameter tuning have been completed, using the results stored in model_results_summary_refactored.csv.

```bash
python3 LOSO_Importance_Train.py
```

```bash
python3 LOSO_Importance_Eval.py
```


## Time split configuration


```bash
python3 GLOBAL_Importance.py
```

### To-Do Experiments

- [ ] **Feature Subset Experiments**
  - [ ] Train models using **only time-domain features**
    - Set `time_features` as usual
    - Set `frequency_features = []`
  - [ ] Train models using **only frequency-domain features**
    - Set `time_features = []`
    - Set `frequency_features` as usual
  - [ ] Compare performance of:
    - [ ] Time-only vs. Frequency-only vs. Time+Frequency

- [ ] **PCA Experiments**
  - [ ] Implement PCA in the `apply_pca` block:
    - Fit PCA on `X_train_scaled`
    - Transform both `X_train_scaled` and `X_test_scaled`
  - [ ] Train models with `apply_pca = True`
  - [ ] Compare:
    - [ ] No PCA vs. PCA for each model and feature set
