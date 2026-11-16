"""Descriptive temporal burden using explicit score-validity criteria."""

import numpy as np
import pandas as pd

from .scores import COMPONENTS, normalized_percent, read_scores, sample_durations, write_csv


KEYS = [
    "neck",
    "left_upper_arm",
    "right_upper_arm",
    "left_lower_arm",
    "right_lower_arm",
    "left_wrist",
    "right_wrist",
    "overall_rula"
]
CATEGORIES = [("minimal", 0.0, 100.0 / 6.0), ("low", 100.0 / 6.0, 50.0),
              ("moderate", 50.0, 500.0 / 6.0), ("high", 500.0 / 6.0, 100.0)]


def summarize_frame(frame, overall_valid_mode="all-parts"):
    if overall_valid_mode not in {"all-parts", "score-only"}:
        raise ValueError("Unsupported overall validity mode.")
    durations = sample_durations(frame.time)
    records = []
    for component in [c for c in COMPONENTS if c.key in KEYS]:
        values = frame[component.column]
        valid = values.notna()
        if component.key == "overall_rula" and overall_valid_mode == "all-parts":
            required = [c.column for c in COMPONENTS if c.key in KEYS]
            valid &= frame[required].notna().all(axis=1)
        percent = normalized_percent(values, component.maximum)
        valid_duration = float(durations[valid].sum())
        record = {"component": component.key, "max_score_used": component.maximum,
                  "procedure_duration_sec": float(durations.sum()), "valid_duration_sec": valid_duration,
                  "excluded_duration_sec": float(durations[~valid].sum())}
        for key, low, high in CATEGORIES:
            mask = percent.ge(low - 1e-9) if low == 0 else percent.gt(low + 1e-9)
            mask &= percent.le(high + 1e-9) & valid
            seconds = float(durations[mask].sum())
            record[key + "_sec"] = seconds
            record[key + "_pct"] = seconds / valid_duration * 100 if valid_duration else np.nan
        records.append(record)
    return records


def run(project, plots=True, overall_valid_mode="all-parts"):
    if overall_valid_mode not in ["score-only", "all-parts"]:
        raise ValueError("Unsupported overall validity mode.")
    records = []
    for name in project.input_ids("imu"):
        frame = read_scores(project, "imu", name)
        for record in summarize_frame(frame, overall_valid_mode):
            records.append({"procedure": name, "participant": name.split("_")[0], **record})
    if not records:
        raise FileNotFoundError("No clinical RULA results are available.")
    procedures = pd.DataFrame(records)
    totals = []
    for component, group in procedures.groupby("component", sort=False):
        duration = group.valid_duration_sec.sum()
        record = {"component": component, "valid_duration_sec": duration,
                  "procedure_duration_sec": group.procedure_duration_sec.sum(),
                  "excluded_duration_sec": group.excluded_duration_sec.sum()}
        for key, _, _ in CATEGORIES:
            seconds = group[key + "_sec"].sum()
            record[key + "_sec"] = seconds
            record[key + "_pct"] = seconds / duration * 100 if duration else np.nan
        totals.append(record)
    overall = pd.DataFrame(totals)
    output = project.output("analysis", "temporal_burden")
    write_csv(procedures, output / "procedure_time_proportions.csv")
    write_csv(overall, output / "overall_time_proportions.csv")
    (output / "summary.txt").write_text(
        "Temporal Ergonomic Burden\n\n"
        "Categories: Minimal, Low, Moderate, High.\n"
        "Categories are descriptive normalized RULA percentage categories.\n"
        f"Overall validity mode: {overall_valid_mode}.\n"
        "All-parts requires valid neck and bilateral upper-arm, lower-arm, and wrist scores for overall RULA.\n"
        "Score-only requires a valid overall RULA score alone.\n"
        "Each joint-level component uses its own valid-time denominator.\n\n" + overall.to_string(index=False) + "\n", encoding="utf-8")
    if plots:
        import matplotlib.pyplot as plt
        labels = {c.key: c.label for c in COMPONENTS}
        fig, ax = plt.subplots(figsize=(12, 6))
        left = np.zeros(len(overall))
        for (key, _, _), color in zip(CATEGORIES, ["white", "#ffd6d6", "#ff7478", "#ff0000"]):
            widths = overall[key + "_pct"].fillna(0).to_numpy()
            ax.barh([labels[c] for c in overall.component], widths, left=left,
                    color=color, edgecolor="black", linewidth=0.5, label=key.title())
            left += widths
        ax.set(xlim=(0, 100), xlabel="Time proportion (%)")
        ax.invert_yaxis()
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
        fig.tight_layout()
        fig.savefig(output / "overall_time_proportions.png", dpi=200)
        fig.savefig(output / "overall_time_proportions.svg")
        plt.close(fig)
    return output
