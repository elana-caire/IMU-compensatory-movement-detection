import numpy as np
import matplotlib.pyplot as plt
def plot_imus_5x3(df_raw, sensor_names, base_signals=None, fs=60, title=None):
    """
    df_raw:      dataframe with columns like 'AccX_IMU1', 'AccY_IMU1', ...
    sensor_names: list of 5 sensor names, e.g. ['IMU1', 'IMU2', ...]
    base_signals: list of base names to plot per IMU, e.g. ['AccX', 'AccY', 'AccZ']
    fs:          sampling frequency (for time axis), default 60 Hz
    """
    if base_signals is None:
        base_signals = ['Acc_X', 'Acc_Y', 'Acc_Z']   # 3 signals per IMU

    n_sensors = len(sensor_names)
    n_signals = len(base_signals)

    # time axis (if you don't have time column)
    n_samples = len(df_raw)
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(
        nrows=n_sensors,
        ncols=n_signals,
        figsize=(4 * n_signals, 2.5 * n_sensors),
        sharex=True
    )

    # If only one sensor or one signal, axes might not be 2D
    if n_sensors == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_signals == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, sensor_name in enumerate(sensor_names):
        for j, base in enumerate(base_signals):
            ax = axes[i, j]

            col_name = f"{base}_{sensor_name}"
            if col_name not in df_raw.columns:
                ax.text(0.5, 0.5, f"{col_name}\nnot found", ha='center', va='center')
                ax.set_axis_off()
                continue

            y = df_raw[col_name].values
            ax.plot(t, y)

            # Titles / labels
            if i == 0:
                ax.set_title(base)          # AccX / AccY / AccZ on top row
            if j == 0:
                ax.set_ylabel(sensor_name)  # IMU name on left column

            if i == n_sensors - 1:
                ax.set_xlabel("Time [s]")
    if title:
        plt.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
