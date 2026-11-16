"""Axis-specific and pooled target-DOF error summaries."""

import numpy as np
import pandas as pd

from ..analysis.scores import write_csv


TARGET_AXES = {"neck": ["x", "y"], "shoulder": ["x", "y"], "elbow": ["x"], "wrist": ["x", "y"]}


def summarize(project):
    records, pooled = [], []
    for joint in project.input_ids("validation"):
        path = project.output("validation", joint, "angle_error/angles_and_errors.csv")
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for axis in ["x", "y", "z"]:
            values = pd.to_numeric(frame["err_" + axis + "_deg"], errors="coerce").dropna().to_numpy()
            if not len(values):
                continue
            selected = axis in TARGET_AXES[joint]
            records.append({"joint": joint, "axis": axis, "target_dof": selected,
                            "sample_count": len(values), "mean_signed_error_deg": float(values.mean()),
                            "mae_deg": float(np.abs(values).mean()), "signed_error_sd_deg": float(values.std(ddof=0))})
            if selected:
                pooled.append(values)
    if not records:
        raise FileNotFoundError("No joint-angle error results are available.")
    values = np.concatenate(pooled)
    records.append({"joint": "pooled_target_dofs", "axis": "selected", "target_dof": True,
                    "sample_count": len(values), "mean_signed_error_deg": float(values.mean()),
                    "mae_deg": float(np.abs(values).mean()), "signed_error_sd_deg": float(values.std(ddof=0))})
    summary = pd.DataFrame(records)
    output = project.output("validation")
    write_csv(summary, output / "metrics_summary.csv")
    (output / "summary.txt").write_text(
        "IMU-OMC Joint-Angle Validation\n\n"
        "Target DOFs: neck x/y, shoulder x/y, elbow x, and wrist x/y.\n"
        "SD denotes the population standard deviation of signed angular errors.\n\n"
        + summary.to_string(index=False) + "\n", encoding="utf-8")
    return summary
