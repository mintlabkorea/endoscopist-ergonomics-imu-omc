"""Descriptive IMU and OMC hand-motion summaries from existing kinematics."""

import numpy as np
import pandas as pd

from ..paths import parse_omc_trial_id
from .scores import read_angles, write_csv


def run(project, plots=True):
    from ..omc.motion_metrics import choose_best_delim, read_with_delim, fix_rows, extract_metrics
    records, missing = [], []
    for name in project.input_ids("imu"):
        angles = read_angles(project, "imu", name)
        start, end = angles.time.min(), angles.time.max()
        for side in ["left", "right"]:
            path = project.output("imu", name, "sensor/filtered/by_sensor", f"{side}_hand_imu.csv")
            if not path.is_file():
                missing.append(str(path))
                continue
            frame = pd.read_csv(path)
            frame = frame.loc[frame.timestamp.between(start, end)]
            acc = np.linalg.norm(frame[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(float), axis=1) * 9.80665
            gyro = np.linalg.norm(frame[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(float), axis=1)
            records.append({"modality": "imu", "recording": name, "side": side,
                            "acceleration_magnitude_mean": np.nanmean(acc), "acceleration_unit": "m/s^2",
                            "angular_velocity_magnitude_mean": np.nanmean(gyro), "angular_velocity_unit": "degrees/s",
                            "sample_count": len(frame)})
    for name in project.input_ids("omc"):
        path = project.output("omc", name, "hand_motion/kinematics.csv")
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.DataFrame(fix_rows(read_with_delim(path, choose_best_delim(path)))).fillna("")
        participant = parse_omc_trial_id(name)["participant"]
        for side, hand in [("left", "LHN"), ("right", "RHN")]:
            acceleration, velocity = extract_metrics(frame, f"{participant}:{hand}")
            records.append({"modality": "omc", "recording": name, "side": side,
                            "acceleration_magnitude_mean": acceleration, "acceleration_unit": "mm/s^2",
                            "angular_velocity_magnitude_mean": velocity, "angular_velocity_unit": "degrees/s",
                            "sample_count": np.nan})
    if not records:
        raise FileNotFoundError("No hand-motion results are available.")
    output = project.output("analysis", "hand_motion")
    table = pd.DataFrame(records)
    write_csv(table, output / "recording_summary.csv")
    overall_records = []
    for (modality, side), group in table.groupby(["modality", "side"], sort=False):
        weights = group.sample_count if modality == "imu" else pd.Series(1.0, index=group.index)
        row = {"modality": modality, "side": side, "recording_count": len(group),
               "acceleration_unit": group.acceleration_unit.iloc[0], "angular_velocity_unit": "degrees/s"}
        for column in ["acceleration_magnitude_mean", "angular_velocity_magnitude_mean"]:
            valid = group[column].notna() & weights.gt(0)
            row[column] = np.average(group.loc[valid, column], weights=weights.loc[valid]) if valid.any() else np.nan
        overall_records.append(row)
    overall = pd.DataFrame(overall_records)
    write_csv(overall, output / "overall_summary.csv")
    (output / "summary.txt").write_text(
        "Hand Motion: Supplementary Descriptive Analysis\n\n"
        "IMU acceleration magnitude includes gravity; it is not gravity-compensated linear acceleration.\n"
        "IMU samples are restricted to the analyzed procedure interval.\n"
        "OMC acceleration and angular velocity are taken from the exported hand kinematics.\n"
        + ("\nUnavailable files:\n" + "\n".join(missing) if missing else ""), encoding="utf-8")
    if plots:
        import matplotlib.pyplot as plt
        for modality, group in overall.groupby("modality", sort=False):
            for column, unit_column in [("acceleration_magnitude_mean", "acceleration_unit"),
                                        ("angular_velocity_magnitude_mean", "angular_velocity_unit")]:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(group.side.str.title(), group[column])
                ax.set(ylabel=group[unit_column].iloc[0], title=f"{modality.upper()} {column.replace('_', ' ')}")
                fig.tight_layout()
                fig.savefig(output / f"{modality}_{column}.png", dpi=200)
                plt.close(fig)
    return output
