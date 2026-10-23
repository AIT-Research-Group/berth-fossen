"""Causal load statistics in BODY axes; simulation labels, not sensor inputs."""
import numpy as np


def load_features(time, wind, wave, window_samples, step_samples):
    time, wind, wave = map(np.asarray, (time, wind, wave))
    if wind.shape != (len(time), 3) or wave.shape != wind.shape:
        raise ValueError("Loads must have shape [time, 3]")
    if window_samples < 1 or step_samples < 1:
        raise ValueError("Positive window and step required")
    if not np.isfinite(wind).all() or not np.isfinite(wave).all():
        raise ValueError("Non-finite load samples")
    rows = []
    for end in range(window_samples-1, len(time), step_samples):
        start = end-window_samples+1
        row = {"time_s": float(time[end]), "window_start_s": float(time[start])}
        for source, values in (("wind", wind), ("wave", wave), ("environment", wind+wave)):
            segment = values[start:end+1]
            for index, (axis, unit) in enumerate((("X", "N"), ("Y", "N"), ("N", "Nm"))):
                signal = segment[:, index]
                for statistic, value in (("mean", signal.mean()), ("rms", np.sqrt(np.mean(signal**2))),
                                         ("absmax", np.max(np.abs(signal)))):
                    row[f"{source}_{axis}_{statistic}_{unit}"] = float(value)
        rows.append(row)
    return rows
