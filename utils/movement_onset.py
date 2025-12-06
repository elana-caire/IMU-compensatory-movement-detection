import numpy as np
import matplotlib.pyplot as plt
def jerk_axis(df, sensor_name, axis="X", fs=60):
    """
    Compute jerk for one axis of one IMU.
    axis: 'X', 'Y', or 'Z'
    Returns an array of same length as input (pads first sample with 0).
    """
    col = f"Acc_{axis}_{sensor_name}"
    acc = df[col].to_numpy()

    # discrete derivative (causal, uses past)
    j = np.diff(acc) #* fs -> optionally to match physical units
    j = np.concatenate([[0.0], j])  # pad to same length as acc

    return j

def detect_movement_start_from_jerk(
    jerk,
    fs=60,
    win_ms=500,
    hop_ms=50,
    baseline_samples=300,
    k_on=4.0,
    n_consecutive_on=5,
    ):
    """
    Windowed, causal detector on |jerk|.

    jerk : 1D array
        Jerk signal for one axis.
    """
    jerk = np.asarray(jerk)
    n = len(jerk)
    if n == 0:
        return [], (None, None, None, None)

    # Take positive values
    activity = np.abs(jerk)

    # --- baseline from first baseline_samples (rest) ---
    baseline_samples = min(baseline_samples, n)
    baseline = activity[:baseline_samples]
    mu = baseline.mean()
    sigma = baseline.std() if baseline.std() > 0 else 1e-6

    thr_on  = mu + k_on  * sigma

    win_len = max(1, int(round(win_ms * fs / 1000.0)))
    hop     = max(1, int(round(hop_ms * fs / 1000.0)))

    moving = False
    start_idx = None
    epochs = []

    on_count  = 0
    off_count = 0
    first_on_start = None

    # start detection after baseline and after we have a full window
    start_search_sample = max(baseline_samples, win_len)
    end = start_search_sample - 1
    found = False
    while (end < n) and (found == False):
        start = end - win_len + 1
        win_mean = activity[start:end + 1].mean()

        if not moving:
            if win_mean > thr_on:
                if on_count == 0:
                    first_on_start = start
                on_count += 1
                if on_count >= n_consecutive_on:
                    moving = True
                    found = True
                    start_idx = first_on_start
                    off_count = 0
            else:
                on_count = 0
                first_on_start = None

        end += hop

            
    if first_on_start == None:
        print("reached end without data found")
        first_on_start = np.nan
    return first_on_start


def plot_onset_detected(df_filt, sens_name, jx, jy, jz, jx_st, jy_st, jz_st, title=None):
    fig, axs = plt.subplots(3, 3, figsize=(20,10))
    mean_start_loc = int(np.nanmean([jx_st, jy_st, jz_st]))
    axs[0,0].plot(df_filt[f'Acc_X_{sens_name}'])
    axs[0,1].plot(jx)
    axs[0,2].plot(np.abs(jx))
    axs[0,0].plot(mean_start_loc, np.array(df_filt[f'Acc_X_{sens_name}'])[mean_start_loc], marker='s')
    if np.isnan(jx_st) == False:
        axs[0,0].plot(jx_st, np.array(df_filt[f'Acc_X_{sens_name}'])[jx_st], marker='o', linestyle='')
        axs[0,2].plot(jx_st, np.abs(jx)[jx_st], marker='o', linestyle='')


    axs[1, 0].plot(df_filt[f'Acc_Y_{sens_name}'])

    axs[1, 1].plot(jy)
    axs[1, 2].plot(np.abs(jy))
    axs[1, 0].plot(mean_start_loc, np.array(df_filt[f'Acc_Y_{sens_name}'])[mean_start_loc], marker='s')
    if np.isnan(jy_st) == False:
        axs[1, 0].plot(jy_st, np.array(df_filt[f'Acc_Y_{sens_name}'])[jy_st], marker='o', linestyle='')
        axs[1, 2].plot(jy_st, np.abs(jy)[jy_st], marker='o', linestyle='')

    axs[2, 0].plot(df_filt[f'Acc_Z_{sens_name}'])

    axs[2, 1].plot(jz)
    axs[2, 2].plot(np.abs(jz))
    axs[2, 0].plot(mean_start_loc, np.array(df_filt[f'Acc_Z_{sens_name}'])[mean_start_loc], marker='s')
    if np.isnan(jz_st) == False:
        axs[2, 0].plot(jz_st, np.array(df_filt[f'Acc_Z_{sens_name}'])[jz_st], marker='o', linestyle='') 
        axs[2, 2].plot(jz_st, np.abs(jz)[jz_st], marker='o', linestyle='')
    if title:
        plt.suptitle(title)

def aling_to_movement_onset(df_filt, plot = False, metadata=['P02', 'cup-placing', 'natural']):
        
    sens_name = 'wrist_r'
    jx = jerk_axis(df_filt, sens_name, "X", fs=60)
    jy = jerk_axis(df_filt, sens_name, "Y", fs=60)
    jz = jerk_axis(df_filt, sens_name, "Z", fs=60)
    jx_st = detect_movement_start_from_jerk(jx, baseline_samples=240)
    jy_st = detect_movement_start_from_jerk(jy, baseline_samples=240)
    jz_st = detect_movement_start_from_jerk(jz,  baseline_samples=240)

    # Note: there might be some cases when we don't detect start and stop from wrist
    if np.isnan(jx_st) and np.isnan(jy_st) and np.isnan(jz_st):
        # Enter here only for few condtions
        print('No onset detected here, cutting ...')
        sens_name = 'wrist_l'
        jx = jerk_axis(df_filt, sens_name, "X", fs=60)
        jy = jerk_axis(df_filt, sens_name, "Y", fs=60)
        jz = jerk_axis(df_filt, sens_name, "Z", fs=60)
        jx_st = detect_movement_start_from_jerk(jx, baseline_samples=240)
        jy_st = detect_movement_start_from_jerk(jy, baseline_samples=240)
        jz_st = detect_movement_start_from_jerk(jz,  baseline_samples=240)

        plot = True
    
    # for simplicty, take the mean
    mean_start_loc = int(np.nanmean([jx_st, jy_st, jz_st]))
    if plot:
        plot_onset_detected(df_filt, sens_name, jx, jy, jz, jx_st, jy_st, jz_st, title=f"{metadata[0]} - {metadata[1]} - {metadata[2]} - {sens_name}")

    df_filt_cut = df_filt.iloc[mean_start_loc:].copy()
    return df_filt_cut
