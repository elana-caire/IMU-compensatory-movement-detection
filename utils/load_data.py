from collections import Counter
import pandas as pd

"""
This script provides:
- A standard IMUDataSample object (dataclass-like dict) containing:
    { 'subject', 'file', 'sensor', 'task', 'condition', 'dataframe' }
- A load_subject() function to load all IMU files for one subject.
- A load_all_subjects() function to load the entire dataset into a list.
- A filter + sorting standardization (PacketCounter / SampleTimeFine).
- A filename → (task, condition) resolver.

"""

import os
import re
from glob import glob
import pandas as pd

# Mapping from filename code → (task, condition)
FILECODE_MAP = {
    '02': ('cup-placing', 'natural'),
    '03': ('pouring', 'natural'),
    '04': ('peg', 'natural'),
    '05': ('wiping', 'natural'),
    '06': ('cup-placing', 'elbow_brace'),
    '07': ('pouring', 'elbow_brace'),
    '08': ('peg', 'elbow_brace'),
    '09': ('wiping', 'elbow_brace'),
    '10': ('cup-placing', 'elbow_wrist_brace'),
    '11': ('pouring', 'elbow_wrist_brace'),
    '12': ('peg', 'elbow_wrist_brace'),
    '13': ('wiping', 'elbow_wrist_brace'),
}

TASK_NAMES = ["cup-placing", "peg", "wiping", "pouring"]
CONDITION_NAMES = ["natural", "elbow_brace", "elbow_wrist_brace"]

def load_imu_csv(path):
    """
    Load a single IMU CSV with cleaning:
    - drop unnamed columns
    - sort by PacketCounter or SampleTimeFine if available
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    for col in ["PacketCounter", "SampleTimeFine"]:
        if col in df.columns:
            df = df.sort_values(col).reset_index(drop=True)

    return df


def infer_task_condition(filename):
    """
    Determine (task, condition) from filename.
    Updated pattern to handle: wrist_r_P_06_02_anonymized.csv
    """
    # Look for the 2-digit code that comes after the subject number
    m = re.search(r"_P_\d+_(\d{2})_anonymized", filename)
    if m:
        code = m.group(1)
        return FILECODE_MAP.get(code, (None, None))
    
    # Alternative pattern if the above doesn't match
    m = re.search(r"_(\d{2})_anonymized", filename)
    if m:
        code = m.group(1)
        return FILECODE_MAP.get(code, (None, None))
    
    # Fallback: extract all 2-digit numbers and use the last one
    nums = re.findall(r"(\d{2})", filename)
    for num in reversed(nums):
        if num in FILECODE_MAP:
            return FILECODE_MAP[num]
    
    return (None, None)


def extract_sensor_name(filename):
    """
    Extract sensor name from filename using the simple approach:
    - Remove extension
    - Split by underscores
    - Take first element, or first two elements if second is 'l' or 'r'
    
    Examples:
    - wrist_r_P_06_02_anonymized.csv → wrist_r
    - trunk_P_05_01_anonymized.csv → trunk
    - arm_l_P_06_01_anonymized.csv → arm_l
    """
    # Remove the .csv extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    # Take first element, and second element if it's 'l' or 'r'
    if len(parts) >= 2 and parts[1] in ['l', 'r']:
        return f"{parts[0]}_{parts[1]}"
    else:
        return parts[0]



def load_subject(subject_dir):
    """
    Load all IMU CSV files for one subject directory.

    Returns: list[dict], where each dict is:
        {
            'subject': 'Pxx',
            'file': 'filename.csv',
            'sensor': 'filename-without-ext',
            'task': str or None,
            'condition': str or None,
            'dataframe': pandas.DataFrame
        }
    """
    samples = []
    csv_files = sorted(glob(os.path.join(subject_dir, "*.csv")))

    subject = os.path.basename(subject_dir)

    for path in csv_files:
        fname = os.path.basename(path)
        sensor_label = extract_sensor_name(fname)
        task, cond = infer_task_condition(fname)
        df = load_imu_csv(path)
        samples.append({
            "subject": subject,
            "file": fname,
            "sensor": sensor_label,
            "task": task,
            "condition": cond,
            "dataframe": df,
        })

    return samples


def load_all_subjects(data_dir, subjects=None):
    """
    Load IMU data for all subjects in data_dir.

    If subjects=None → auto-detect directories beginning with 'P'.
    Returns: list of sample dicts.
    """
    if subjects is None:
        subjects = [d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("P")]

    all_samples = []
    for subj in sorted(subjects):
        subj_dir = os.path.join(data_dir, subj)
        all_samples.extend(load_subject(subj_dir))

    return all_samples




def build_raw_dataset(data, tasks, conditions, subject_ids=range(2, 7)):
    all_subjects = []

    for sub_id in subject_ids:
        subject_id = f"P0{sub_id}"
        subject_data = [d for d in data if d.get("subject") == subject_id]

        for curr_task in tasks:
            task_data = [t for t in subject_data if t.get("task") == curr_task]

            # Must be 15 entries = 5 sensors × 3 conditions
            if len(task_data) != 15:
                print("Len different than 15, task is:", curr_task)
                print("skipping...", [f['file'] for f in task_data])
                print()
                continue

            # Unique task name
            task_names = [t['task'] for t in task_data]
            task_names = [item for item, count in Counter(task_names).items() if count > 1]
            if len(task_names) != 1:
                continue
            task_name = task_names[0]

            for curr_condition in conditions:
                df_raw = pd.DataFrame()
                condition_data = [c for c in task_data if c.get("condition") == curr_condition]

                # Unique condition
                condition_names = [c['condition'] for c in condition_data]
                condition_names = [item for item, count in Counter(condition_names).items() if count > 1]
                if len(condition_names) != 1:
                    continue
                condition_name = condition_names[0]

                # --- extract and align sensors ---
                dfs_sensors = [df['dataframe'] for df in condition_data]
                sensor_names = [df['sensor'] for df in condition_data]

                lengths = [len(df) for df in dfs_sensors]
                min_len = min(lengths)

                for curr_df, sensor_name in zip(dfs_sensors, sensor_names):
                    ts = pd.to_datetime(curr_df["SampleTimeFine"])
                    dt_s = ts.diff().dt.total_seconds()

                    if dt_s.dropna().nunique() != 1:
                        print(f"[{subject_id} | {task_name} | {condition_name} | {sensor_name}] Non-constant dt!")

                    if curr_df["PacketCounter"].diff().dropna().nunique() != 1:
                        print(f"[{subject_id} | {task_name} | {condition_name} | {sensor_name}] Packet counter not consecutive!")

                    sensor_data = curr_df.iloc[:min_len].copy()
                    sensor_data = sensor_data[curr_df.columns[2:]]   # drop time + counter
                    sensor_data = sensor_data.add_suffix(f"_{sensor_name}")

                    df_raw = pd.concat([df_raw, sensor_data], axis=1)

                # Add metadata columns
                df_raw["subject"] = subject_id
                df_raw["task"] = task_name
                df_raw["condition"] = condition_name

                print(f"Done with subject {subject_id}, task {task_name}, condition {condition_name}")

                all_subjects.append({
                    "subject": subject_id,
                    "task": task_name,
                    "condition": condition_name,
                    "data": df_raw
                })

        print(f"Done with subject {subject_id}...\n")
    
    dataset = {}
    for entry in all_subjects:
        s = entry["subject"]
        t = entry["task"]
        c = entry["condition"]
        df = entry["data"]

        dataset.setdefault(s, {}).setdefault(t, {})[c] = df



    return all_subjects, dataset