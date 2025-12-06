import numpy as np
import pandas as pd
from numpy.fft import fft, fftfreq  

def feat_max(x):
    return np.nanmax(x)

def feat_min(x):
    return np.nanmin(x)

def feat_amp(x):
    return np.nanmax(x) - np.nanmin(x)

def feat_mean(x):
    return np.nanmean(x)

def feat_rms(x):
    return np.sqrt(np.mean(x**2))

def feat_corr(x):
    """
    Fluctuation measure: STD of the first derivative.
    """
    if len(x) < 2:
        return 0
    dx = np.diff(x)
    return np.std(dx)

def feat_std(x):
    return np.std(x)


def feat_jerk(x, fs = int(60)):
    """
    Smoothness: jerk = derivative of signal (approx).
    Using finite difference: jerk = dx/dt.
    Report RMS jerk.
    """
    if len(x) < 2:
        return 0      
    dx = np.diff(x)
    jerk = dx * fs
    return np.sqrt(np.mean(jerk**2))

def feat_spectral(x, fs):
    """
    Compute spectral-domain features from a 1D signal x.

    Returns a dict with:
      - DOMFREQ: dominant frequency (Hz)
      - DOMPOW:  dominant power
      - TOTPOW:  total power
      - CENT:    spectral centroid (Hz)
      - SPREAD:  spectral spread (Hz)
    """
    x = np.asarray(x)
    n = len(x)
    if n < 2:
        return {
            "DOMFREQ": np.nan,
            "DOMPOW":  np.nan,
            "TOTPOW":  np.nan,
            "CENT":    np.nan,
            "SPREAD":  np.nan,
        }

    # optional: remove DC bias before FFT
    x = x - x.mean()

    # --- Spectral-domain features ---
    signal_fft = fft(x)
    freqs = fftfreq(n, d=1.0/fs)
    power_spectrum = np.abs(signal_fft) ** 2

    # keep only non-negative frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    power_spectrum = power_spectrum[pos_mask]

    if len(power_spectrum) <= 1:
        return {
            "DOMFREQ": np.nan,
            "DOMPOW":  np.nan,
            "TOTPOW":  np.nan,
            "CENT":    np.nan,
            "SPREAD":  np.nan,
        }

    # dominant frequency (skip DC if possible)
    if len(power_spectrum) > 1:
        dom_idx = np.argmax(power_spectrum[1:]) + 1
    else:
        dom_idx = 0
    dom_freq = freqs[dom_idx]
    dom_pow  = power_spectrum[dom_idx]

    # total power
    tot_pow = power_spectrum.sum()

    # spectral centroid & spread
    if tot_pow > 0:
        centroid = np.sum(freqs * power_spectrum) / tot_pow
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power_spectrum) / tot_pow)
    else:
        centroid = np.nan
        spread = np.nan

    return {
        "DOMFREQ": dom_freq,
        "DOMPOW":  dom_pow,
        "TOTPOW":  tot_pow,
        "CENT":    centroid,
        "SPREAD":  spread,
    }


def extract_features_from_signal(df, col, fs):
    x = df[col].to_numpy()
    #t = df["timestamp"].to_numpy()
    spec = feat_spectral(x, fs=fs)
    return {
        f"{col}_MAX": feat_max(x),
        f"{col}_MIN": feat_min(x),
        f"{col}_AMP": feat_amp(x),
        f"{col}_MEAN": feat_mean(x),
        f"{col}_JERK": feat_jerk(x, fs=60),
        f"{col}_RMS": feat_rms(x),
        f"{col}_COR": feat_corr(x),
        f"{col}_STD": feat_std(x),
        # spectral-domain features (using your fft / power_spectrum)
        f"{col}_DOMFREQ":    spec["DOMFREQ"],
        f"{col}_DOMPOW":     spec["DOMPOW"],
        f"{col}_TOTPOW":     spec["TOTPOW"],
        f"{col}_SPEC_CENT":  spec["CENT"],
        f"{col}_SPEC_SPREAD": spec["SPREAD"],

    }



def extract_all_features(
    df,
    exclude_acc=False,
    exclude_quat=False,
    exclude_gyro=False,
    exclude_mag=False,
    window_ms=None,
    fs=60,
):
    """
    Returns a DataFrame with features from all XSens signals
    (Acc, Quat, Gyr, Mag).

    - If window_ms is None: single-row DataFrame with features over entire df.
    - If window_ms is not None: one row per window, indexed by window start
      time (value from df.index).

    Assumes df.index is a regular time axis with step 1/fs.
    """

    def _feature_columns(df_win):
        quaternion_columns = (
            df_win.columns[df_win.columns.str.contains("Quat")]
            if not exclude_quat else []
        )
        acc_columns = (
            df_win.columns[df_win.columns.str.contains("Acc")]
            if not exclude_acc else []
        )
        gyr_columns = (
            df_win.columns[df_win.columns.str.contains("Gyr")]
            if not exclude_gyro else []
        )
        mag_columns = (
            df_win.columns[df_win.columns.str.contains("Mag")]
            if not exclude_mag else []
        )

        return (
            list(quaternion_columns)
            + list(acc_columns)
            + list(gyr_columns)
            + list(mag_columns)
        )

    # ---------- no windowing: one feature vector ----------
    if window_ms is None:
        feature_columns = _feature_columns(df)
        feats = {}
        for col in feature_columns:
            if col not in df.columns:
                continue
            feats.update(extract_features_from_signal(df, col, fs=fs))

        # index = first time sample of this segment
        idx0 = df.index[0] if len(df.index) > 0 else 0
        return pd.DataFrame([feats], index=[idx0])

    # ---------- windowed version (sample-based, index is numeric time) ----------
    # window length in samples
    win_samples = int(round(window_ms * fs / 1000.0))
    if win_samples < 1:
        raise ValueError("window_ms too small for given fs.")

    n = len(df)
    rows = []
    idxs = []

    # non-overlapping windows: 0..win, win..2win, ...
    for start in range(0, n - win_samples + 1, win_samples):
        stop = start + win_samples
        df_win = df.iloc[start:stop]

        if df_win.empty:
            continue

        feature_columns = _feature_columns(df_win)
        feats = {}
        for col in feature_columns:
            if col not in df_win.columns:
                continue
            feats.update(extract_features_from_signal(df_win, col, fs=fs))

        rows.append(feats)
        # use df.index as "time" for this window (start of window)
        idxs.append(df_win.index[0])

    return pd.DataFrame(rows, index=idxs)



# def extract_features_from_signal(df, col):
#     x = df[col].to_numpy()
#     #t = df["timestamp"].to_numpy()

#     return {
#         f"{col}_MAX": feat_max(x),
#         f"{col}_MIN": feat_min(x),
#         f"{col}_AMP": feat_amp(x),
#         f"{col}_MEAN": feat_mean(x),
#         f"{col}_JERK": feat_jerk(x, t),
#         f"{col}_RMS": feat_rms(x),
#         f"{col}_COR": feat_corr(x),
#         f"{col}_STD": feat_std(x),
#     }

# FEATURE_COLUMNS = [
#     "Acc_X","Acc_Y","Acc_Z",
#     "Gyr_X","Gyr_Y","Gyr_Z",
#     "Yaw","Pitch","Roll"
# ]

# def extract_all_features(df):
#     """
#     Returns a single-row DataFrame with 72 features.
#     """
#     feats = {}
#     for col in FEATURE_COLUMNS:
#         if col not in df.columns:
#             continue
#         feats.update(extract_features_from_signal(df, col))
#     return pd.DataFrame([feats])

