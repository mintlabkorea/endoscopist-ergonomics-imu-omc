"""Detect flat IMU signals and record dependent-angle exclusions."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .exclusions import SENSOR_ANGLES

SAFETY_MARGIN_SEC = 10.0


def write_log(path):
    folder = Path(path).parent
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"[Data Log] {Path(path).name} ({datetime.now():%Y-%m-%d %H:%M})\n")


def find_flat_runs_by_slope(values, min_len=500, window=5, dt=0.01,
                            slope_tol_per_s=1e-6, range_tol=1e-8):
    """Find sustained flat runs after the initial calibration plateau."""
    samples = pd.to_numeric(values, errors="coerce").to_numpy()
    count = samples.size
    if count == 0 or window < 3:
        return []
    if window % 2 == 0:
        window += 1
    half = window // 2
    if count < window:
        return []
    slope = np.full(count, np.nan)
    differences = samples[half:] - samples[:-half]
    slope[half:count - half] = differences[half:] / (2 * half * dt)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(samples, window_shape=window)
    ranges = np.full(count, np.nan)
    ranges[half:count - half] = np.nanmax(windows, axis=1) - np.nanmin(windows, axis=1)
    flat = (np.abs(slope) <= slope_tol_per_s) & (ranges <= range_tol)
    runs = []
    index = 0
    while index < count:
        if not flat[index]:
            index += 1
            continue
        end = index + 1
        while end < count and flat[end]:
            end += 1
        if end - index >= min_len and index > half:
            runs.append((index, end - 1))
        index = end
    return runs


def detect_sensor_failures(folder):
    """Describe flat-signal intervals separately from the exclusion safety margin."""
    folder = Path(folder)
    missing = [sensor + ".csv" for sensor in SENSOR_ANGLES if not (folder / (sensor + ".csv")).is_file()]
    if missing:
        raise FileNotFoundError(f"Filtered IMU files are missing in {folder}: {', '.join(missing)}")
    failures = []
    for sensor in SENSOR_ANGLES:
        name = sensor + ".csv"
        frame = pd.read_csv(folder / name, usecols=[0, 1])
        if "Acc_X" not in frame or frame.empty:
            raise ValueError(f"Filtered IMU data contain no Acc_X samples: {folder / name}")
        timestamps = pd.to_numeric(frame.iloc[:, 0], errors="raise")
        if not np.isfinite(timestamps).all() or not timestamps.is_monotonic_increasing:
            raise ValueError(f"Filtered IMU timestamps must be finite and increasing: {folder / name}")
        for start, end in find_flat_runs_by_slope(frame.Acc_X):
            onset = float(timestamps.iloc[start])
            failures.append({"sensor": sensor, "flat_start_sec": onset,
                             "flat_end_sec": float(timestamps.iloc[end]),
                             "excluded_from_sec": onset - SAFETY_MARGIN_SEC})
    return failures


def check_imu_data(path):
    failures = detect_sensor_failures(Path(path) / "1. imu" / "each")
    with (Path(path).parent / "log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"  Filtered sensors: {len(SENSOR_ANGLES)}\n")
        for failure in failures:
            handle.write(f"  Flat signal: {failure['sensor']}; interval "
                         f"{failure['flat_start_sec']:.2f}-{failure['flat_end_sec']:.2f} s; "
                         f"safety margin starts at {failure['excluded_from_sec']:.2f} s\n")
    return failures


def log_on_the_error(path, name_of_file, timing_it_went_wrong):
    folder = Path(path)
    sensor = Path(name_of_file).stem
    if sensor not in SENSOR_ANGLES:
        raise ValueError(f"Unknown sensor: {sensor}")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "error_log.txt").open("a", encoding="utf-8") as handle:
        handle.write("\n==================== IMU ERROR ====================\n")
        handle.write(f"Procedure      : {folder.name}\n")
        handle.write(f"IMU file       : {name_of_file}\n")
        handle.write(f"Failure time   : {timing_it_went_wrong}\n")
        handle.write("Affected angles:\n")
        for angle in SENSOR_ANGLES[sensor]:
            handle.write(f"  - {angle}.csv\n")
        handle.write("===================================================\n")
