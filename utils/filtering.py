import os
import re
from glob import glob
import pandas as pd
import numpy as np
from scipy.signal import butter,filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



def add_timestamp_column(df, fs=60):
    """
    Add a synthetic timestamp column assuming fixed sampling rate fs (default 60 Hz)
    using PacketCounter.
    - If PacketCounter exists: t = PacketCounter / fs
    - Else: t = index / fs
    """
    if "PacketCounter" in df.columns:
        df = df.copy()
        df["timestamp"] = df["PacketCounter"] / fs
    else:
        df = df.copy()
        df["timestamp"] = df.index / fs
    return df

def quat_to_euler(df, qw='Quat_W', qx='Quat_X', qy='Quat_Y', qz='Quat_Z'):
    """
    Convert quaternions to Euler angles (yaw, pitch, roll).
    Returns a DataFrame with new columns: Yaw, Pitch, Roll.
    """
    q = df[[qw, qx, qy, qz]].to_numpy(dtype=float)

    qw_, qx_, qy_, qz_ = q[:,0], q[:,1], q[:,2], q[:,3]

    # Yaw (Z axis rotation)
    ys = np.arctan2(2*(qw_*qz_ + qx_*qy_), 1 - 2*(qy_**2 + qz_**2))

    # Pitch (Y axis)
    sinp = 2*(qw_*qy_ - qz_*qx_)
    sinp = np.clip(sinp, -1, 1)
    ps = np.arcsin(sinp)

    # Roll (X axis)
    rs = np.arctan2(2*(qw_*qx_ + qy_*qz_), 1 - 2*(qx_**2 + qy_**2))

    df_out = df.copy()
    df_out["Yaw"] = ys
    df_out["Pitch"] = ps
    df_out["Roll"] = rs
    return df_out

def normalize_quaternions(df, qw='Quat_W', qx='Quat_X', qy='Quat_Y', qz='Quat_Z'):
    q = df[[qw, qx, qy, qz]].to_numpy(dtype=float)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    qn = q / norms
    df = df.copy()
    df[[qw, qx, qy, qz]] = qn
    return df

def hampel_filter(series, window_size=7, n_sigmas=3):
    # window_size odd
    new_series = series.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    L = n_sigmas
    n = (window_size - 1) // 2
    for i in range(len(series)):
        start = max(0, i - n)
        end = min(len(series), i + n + 1)
        window = series[start:end]
        med = np.nanmedian(window)
        mad = k * np.nanmedian(np.abs(window - med))
        if mad == 0:
            continue
        if abs(series[i] - med) > L * mad:
            new_series[i] = med
    return new_series

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butter_filtfilt_df(df, cols, fs=60.0, cutoff=6.0, order=4):
    """
    Zero-phase Butterworth filter on specified columns.
    cutoff: Hz (6.0 is a reasonable default for human motion sampled at 60Hz)
    """
    df = df.copy()
    b, a = butter_lowpass(cutoff, fs, order=order)
    for c in cols:
        arr = df[c].to_numpy(dtype=float)
        # if short signals, filtfilt may fail; ensure len > 3*max(len(a), len(b))
        if len(arr) < (3 * max(len(a), len(b))):
            df[c] = arr  # skip filtering if too short
        else:
            df[c] = filtfilt(b, a, arr, method='pad')
    return df


def filter_butterworth(
    df,
    fs=60,
    cutoff=6.0,
    sensor_names=('arm_l', 'arm_r', 'trunk', 'wrist_l', 'wrist_r'),
    exclude_mag=False,
    ):
    """
    Hampel despike -> Butterworth lowpass
    applied to *all* sensors in a wide df with suffixed columns, e.g.:

        Acc_X_arm_l, Acc_Y_arm_l, ..., Gyr_Z_arm_l,
        Acc_X_arm_r, ..., Gyr_Z_arm_r, ...

    Parameters
    ----------
    df : pd.DataFrame
        Wide dataframe containing all sensors.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Low-pass cutoff frequency in Hz.
    sensor_names : iterable of str
        Names used as suffixes in the columns (e.g. 'arm_l', 'trunk', ...).
    exclude_mag : bool
        If False, also filter magnetometer data (columns containing 'Mag').
    """
    df_proc = df.copy()

    for sensor_name in sensor_names:
        # all columns for this sensor
        sensor_columns = df_proc.columns[df_proc.columns.str.endswith(f"_{sensor_name}")]

        # group by type
        quaternion_columns = sensor_columns[sensor_columns.str.contains("Quat")]
        acc_columns        = sensor_columns[sensor_columns.str.contains("Acc")]
        gyr_columns        = sensor_columns[sensor_columns.str.contains("Gyr")]
        mag_columns        = sensor_columns[sensor_columns.str.contains("Mag")] if not exclude_mag else []

        # we usually filter only raw sensor signals (Acc, Gyr, Mag)
        if exclude_mag == False:
            cols_to_consider = list(acc_columns) + list(gyr_columns) + list(quaternion_columns) + list(mag_columns) 
        else:
            cols_to_consider = list(acc_columns) + list(gyr_columns) + list(quaternion_columns)

        # nothing to do for this sensor
        if not cols_to_consider:
            continue

        # 1) Hampel despike
        for c in cols_to_consider:
            df_proc[c] = hampel_filter(
                df_proc[c].to_numpy(),
                window_size=7,
                n_sigmas=3,
            )

        # 2) Butterworth lowpass
        df_proc = apply_butter_filtfilt_df(
            df_proc,
            cols_to_consider,
            fs=fs,
            cutoff=cutoff,
            order=4,
        )

        # --- optional: quaternion renorm + Euler could go here,
        # using `quaternion_columns` if you want to handle them per sensor ---

    return df_proc





############################# OLD: commented for now #################################



# def moving_average_centered(arr, n=5):
#     # n: radius -> window length = 2*n+1
#     window_len = 2 * n + 1
#     if arr.ndim == 1:
#         pad = np.pad(arr, n, mode='reflect')
#         kernel = np.ones(window_len) / window_len
#         return np.convolve(pad, kernel, mode='valid')
#     else:
#         out = np.zeros_like(arr)
#         for i in range(arr.shape[1]):
#             pad = np.pad(arr[:, i], n, mode='reflect')
#             out[:, i] = np.convolve(pad, np.ones(window_len) / window_len, mode='valid')
#         return out

# def apply_moving_avg_df(df, cols, n=5):
#     """
#     Return a copy of df with cols smoothed by a centered moving average (n radius).
#     """
#     df = df.copy()
#     window_len = 2 * n + 1
#     for c in cols:
#         arr = df[c].to_numpy(dtype=float)
#         df[c] = moving_average_centered(arr, n=n)
#     return df


# def filter_butterworth(df, fs=60, cutoff=6.0, sensor_names = []):
#     """
#     Hampel despike -> Butterworth lowpass -> quaternion renorm -> Euler.
#     """
#     df_proc = df.copy()

#     imu_cols = ['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z']

#     # 1) Hampel despike
#     for c in imu_cols:
#         df_proc[c] = hampel_filter(df_proc[c].to_numpy(), window_size=7, n_sigmas=3)

#     # 2) Butterworth lowpass
#     df_proc = apply_butter_filtfilt_df(df_proc, imu_cols, fs=fs, cutoff=cutoff, order=4)

#     # 3) Normalize quaternions (if present)
#     if all(c in df_proc.columns for c in ["Quat_W","Quat_X","Quat_Y","Quat_Z"]):
#         df_proc = normalize_quaternions(df_proc)

#     # 4) Convert to Euler
#     if all(c in df_proc.columns for c in ["Quat_W","Quat_X","Quat_Y","Quat_Z"]):
#         df_proc = quat_to_euler(df_proc)

#     return df_proc

# def filter_moving_average(df, n=5):
#     """
#     Moving average -> quaternion renorm -> Euler
#     """
#     df_ma = df.copy()

#     imu_cols = ['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z']

#     # Apply MA filter
#     df_ma = apply_moving_avg_df(df_ma, imu_cols, n=n)

#     # Normalize quaternions if available
#     if all(c in df_ma.columns for c in ["Quat_W","Quat_X","Quat_Y","Quat_Z"]):
#         df_ma = normalize_quaternions(df_ma)
#         df_ma = quat_to_euler(df_ma)

    return df_ma
