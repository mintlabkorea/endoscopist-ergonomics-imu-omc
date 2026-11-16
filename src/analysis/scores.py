"""Shared RULA scales and canonical result readers."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Component:
    key: str
    column: str
    maximum: int
    label: str


COMPONENTS = [
    Component("neck", "neck_score", 6, "Neck"),
    Component("upper_arm", "upper_arm_score", 6, "Upper arm"),
    Component("left_upper_arm", "upper_arm_score_left", 6, "Left upper arm"),
    Component("right_upper_arm", "upper_arm_score_right", 6, "Right upper arm"),
    Component("lower_arm", "lower_arm_score", 3, "Lower arm"),
    Component("left_lower_arm", "lower_arm_score_left", 3, "Left lower arm"),
    Component("right_lower_arm", "lower_arm_score_right", 3, "Right lower arm"),
    Component("wrist", "wrist_score", 4, "Wrist"),
    Component("left_wrist", "wrist_score_left", 4, "Left wrist"),
    Component("right_wrist", "wrist_score_right", 4, "Right wrist"),
    Component("wrist_and_arm", "wrist_and_arm_score", 10, "Wrist and arm"),
    Component("left_wrist_and_arm", "wrist_and_arm_score_left", 10, "Left wrist and arm"),
    Component("right_wrist_and_arm", "wrist_and_arm_score_right", 10, "Right wrist and arm"),
    Component("overall_rula", "rula_score", 7, "Overall RULA"),
]


def normalized_percent(values, maximum):
    return (values - 1.0) / (maximum - 1.0) * 100.0


def read_scores(project, kind, name):
    path = project.output(kind, name, "rula", "rula_scores.csv")
    if not path.is_file():
        raise FileNotFoundError(f"RULA results are missing: {path}")
    frame = pd.read_csv(path)
    needed = ["time"] + [c.column for c in COMPONENTS]
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing RULA columns in {path}: {missing}")
    for column in needed:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.empty or frame.time.isna().any() or frame.time.duplicated().any():
        raise ValueError(f"Invalid or duplicate RULA timestamps: {path}")
    frame = frame.sort_values("time").reset_index(drop=True)
    for component in COMPONENTS:
        valid = frame[component.column].dropna()
        if ((valid < 1) | (valid > component.maximum)).any():
            raise ValueError(f"Out-of-range {component.column} values in {path}")
    return frame


def sample_durations(time, max_gap_factor=5.0):
    time = pd.to_numeric(time, errors="raise")
    intervals = time.shift(-1) - time
    positive = intervals[intervals > 0]
    nominal = positive.median() if not positive.empty else 0.01
    if not np.isfinite(nominal) or nominal <= 0:
        raise ValueError("Cannot determine a positive sampling interval.")
    return intervals.where((intervals > 0) & (intervals <= nominal * max_gap_factor), nominal).fillna(nominal)


def analyzed_duration(time):
    """Return the timestamp span used to weight a procedure or trial mean."""
    time = pd.to_numeric(time, errors="raise")
    if len(time) < 2 or not np.isfinite(time.to_numpy()).all() or not time.is_monotonic_increasing:
        raise ValueError("Analyzed timestamps must be finite and increasing.")
    duration = float(time.iloc[-1] - time.iloc[0])
    if duration <= 0 or time.duplicated().any():
        raise ValueError("Analyzed timestamps must span a positive duration without duplicates.")
    return duration


def read_angles(project, kind, name, mode="valid"):
    folder = project.output(kind, name, "angle", mode)
    combined = folder / "angles.csv"
    if combined.is_file():
        frame = pd.read_csv(combined)
    else:
        files = sorted((folder / "by_angle").glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"Angle results are missing: {folder}")
        frame = None
        for path in files:
            part = pd.read_csv(path)
            part.columns = ["time", path.stem]
            frame = part if frame is None else pd.merge(frame, part, on="time", how="outer", validate="one_to_one")
    if "time" not in frame or frame.empty:
        raise ValueError(f"Angle results contain no time samples: {folder}")
    return frame.sort_values("time").reset_index(drop=True)


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
