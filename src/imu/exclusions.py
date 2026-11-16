"""Preserve automatic sensor exclusions while applying optional procedure limits."""

import numpy as np
import pandas as pd


SENSOR_ANGLES = {
    "torso_imu": ["neck_Flex", "neck_Twist", "neck_Bend", "r_shoulder_Flex", "r_shoulder_Abd",
                  "r_shoulder_Twist", "l_shoulder_Flex", "l_shoulder_Abd", "l_shoulder_Twist"],
    "head_imu": ["neck_Flex", "neck_Twist", "neck_Bend"],
    "right_humerus_imu": ["r_shoulder_Flex", "r_shoulder_Abd", "r_shoulder_Twist", "r_elbow_Flex"],
    "left_humerus_imu": ["l_shoulder_Flex", "l_shoulder_Abd", "l_shoulder_Twist", "l_elbow_Flex"],
    "right_proximal_radius_imu": ["r_elbow_Flex", "r_wrist_Twist"],
    "left_proximal_radius_imu": ["l_elbow_Flex", "l_wrist_Twist"],
    "right_distal_radius_imu": ["r_wrist_Twist", "r_wrist_Flex", "r_wrist_Abd"],
    "left_distal_radius_imu": ["l_wrist_Twist", "l_wrist_Flex", "l_wrist_Abd"],
    "right_hand_imu": ["r_wrist_Flex", "r_wrist_Abd"],
    "left_hand_imu": ["l_wrist_Flex", "l_wrist_Abd"],
}


COMPONENT_ANGLES = {
    "neck": {"neck_Flex", "neck_Bend"},
    "left_upper_arm": {"l_shoulder_Flex", "l_shoulder_Abd"},
    "right_upper_arm": {"r_shoulder_Flex", "r_shoulder_Abd"},
    "left_lower_arm": {"l_elbow_Flex"},
    "right_lower_arm": {"r_elbow_Flex"},
    "left_wrist": {"l_wrist_Flex", "l_wrist_Abd"},
    "right_wrist": {"r_wrist_Flex", "r_wrist_Abd"},
}


def failures_in_window(failures, start, end, manually_validated=False):
    """Select failures affecting the analyzed window, preserving confirmed manual windows."""
    if not np.isfinite([start, end]).all() or start > end:
        raise ValueError("The analyzed window must have finite ordered bounds.")
    selected = []
    for failure in failures:
        onset = failure["flat_start_sec"] if manually_validated else failure["excluded_from_sec"]
        if onset <= end + 1e-9 and failure["flat_end_sec"] >= start - 1e-9:
            selected.append(failure)
    return selected


def affected_components(failures):
    """Map interrupted sensors to unavailable whole-procedure score estimates."""
    affected = {key: set() for key in COMPONENT_ANGLES}
    for failure in failures:
        sensor = failure["sensor"]
        if sensor not in SENSOR_ANGLES:
            raise ValueError(f"Unknown sensor in automatic exclusion: {sensor}")
        angles = set(SENSOR_ANGLES[sensor])
        for component, dependencies in COMPONENT_ANGLES.items():
            if angles & dependencies:
                affected[component].add(sensor)
    for side in ["left", "right"]:
        affected[side + "_wrist_and_arm"] = set().union(
            *(affected[side + "_" + part] for part in ["upper_arm", "lower_arm", "wrist"]))
    for region in ["upper_arm", "lower_arm", "wrist", "wrist_and_arm"]:
        left, right = affected["left_" + region], affected["right_" + region]
        affected[region] = left | right if left and right else set()
    affected["overall_rula"] = set().union(
        *(affected[key] for key in ["neck", "upper_arm", "lower_arm", "wrist"]))
    return {key: sorted(sensors) for key, sensors in affected.items() if sensors}


def apply_exclusions(frame, failures=(), valid_time=None):
    result = frame.copy()
    if "time" not in result:
        raise ValueError("Angle data must contain a time column.")
    result["time"] = pd.to_numeric(result["time"], errors="raise")
    for sensor, failure_time in failures:
        sensor = str(sensor).removesuffix(".csv")
        if sensor not in SENSOR_ANGLES:
            raise ValueError(f"Unknown sensor in automatic exclusion: {sensor}")
        for column in SENSOR_ANGLES[sensor]:
            if column in result:
                result.loc[result.time >= float(failure_time), column] = np.nan
    if valid_time is not None:
        start, end = valid_time
        result = result.loc[result.time.between(start, end, inclusive="both")].copy()
    if result.empty:
        raise ValueError("No angle samples remain inside the configured valid time range.")
    return result.reset_index(drop=True)


def prepare_analysis_angles(folder, failures, valid_time):
    frame = pd.read_csv(folder / "combined_angle.csv")
    window = frame if valid_time is None else frame.loc[frame.time.between(*valid_time, inclusive="both")]
    if window.empty:
        raise ValueError("No angle samples remain inside the configured valid time range.")
    active = failures_in_window(failures, float(window.time.min()), float(window.time.max()),
                               manually_validated=valid_time is not None)
    intervals = [(failure["sensor"], failure["excluded_from_sec"]) for failure in active]
    valid = apply_exclusions(frame, intervals, valid_time)
    each = folder / "each"
    each.mkdir(parents=True, exist_ok=True)
    for column in valid.columns:
        if column != "time":
            valid[["time", column]].to_csv(each / f"{column}_fixed.csv", index=False)
    valid.to_csv(folder / "valid_angles.csv", index=False)
    return valid
