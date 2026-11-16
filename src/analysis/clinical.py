"""Duration-weighted clinical summaries and participant distributions."""

import numpy as np
import pandas as pd

from ..imu.exclusions import affected_components, failures_in_window
from ..imu.quality import detect_sensor_failures
from .scores import COMPONENTS, analyzed_duration, normalized_percent, read_scores, sample_durations, write_csv


def summary_options(project):
    options = project.clinical_summary_options()
    unknown = set(options) - {"source", "procedure_mean_decimals", "excluded_components"}
    if unknown:
        raise ValueError(f"Unknown clinical summary options: {sorted(unknown)}")
    decimals = options.get("procedure_mean_decimals")
    if decimals is not None and (type(decimals) is not int or not 0 <= decimals <= 12):
        raise ValueError("procedure_mean_decimals must be an integer between 0 and 12.")
    exclusions = options.get("excluded_components", {})
    if not isinstance(exclusions, dict):
        raise ValueError("Clinical exclusions must be an object keyed by procedure.")
    available = set(project.input_ids("imu"))
    components = {component.key for component in COMPONENTS}
    for name, keys in exclusions.items():
        if name not in available or not isinstance(keys, list) or any(key not in components for key in keys):
            raise ValueError(f"Invalid clinical summary exclusion: {name}")
    return options


def build_summary(project):
    options = summary_options(project)
    decimals = options.get("procedure_mean_decimals")
    limits = project.time_limits()
    records = []
    for name in project.input_ids("imu"):
        frame = read_scores(project, "imu", name)
        valid_time = limits.get(name)
        if valid_time is not None:
            frame = frame.loc[frame.time.between(*valid_time, inclusive="both")].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No analyzed scores remain inside the configured time range: {name}")
        failures = detect_sensor_failures(project.output("imu", name, "sensor/filtered/by_sensor"))
        active = failures_in_window(failures, float(frame.time.min()), float(frame.time.max()),
                                    manually_validated=valid_time is not None)
        exclusions = affected_components(active)
        durations = sample_durations(frame.time)
        for component in COMPONENTS:
            values = frame[component.column]
            valid = values.notna()
            computed = values.mean()
            mean = float(f"{computed:.{decimals}f}") if decimals is not None else computed
            source = "rounded_procedure_mean" if decimals is not None else "procedure_mean"
            sensors = exclusions.get(component.key, [])
            excluded = component.key in options.get("excluded_components", {}).get(name, [])
            if excluded:
                mean, source = np.nan, "excluded_by_configuration"
            records.append({"procedure": name, "participant": name.split("_")[0],
                            "component": component.key, "computed_raw_mean": computed,
                            "raw_mean": mean, "summary_input_source": source,
                            "detected_sensors": ";".join(sensors),
                            "excluded_sensors": ";".join(sensors) if excluded else "",
                            "normalized_rula_pct": normalized_percent(mean, component.maximum),
                            "valid_duration_sec": float(durations[valid].sum()),
                            "excluded_duration_sec": float(durations[~valid].sum()),
                            "procedure_duration_sec": analyzed_duration(frame.time)})
    if not records:
        raise FileNotFoundError("No clinical RULA results are available.")
    procedures = pd.DataFrame(records)
    participant_records = []
    for (participant, component), group in procedures.groupby(["participant", "component"], sort=False):
        valid = group.raw_mean.notna() & group.procedure_duration_sec.gt(0)
        usable = group.loc[valid]
        weights = usable.procedure_duration_sec
        participant_records.append({"participant": participant, "component": component,
                                    "procedure_count": len(group), "usable_procedure_count": len(usable),
                                    "procedure_duration_sec": weights.sum(),
                                    "valid_duration_sec": usable.valid_duration_sec.sum(),
                                    "raw_mean": np.average(usable.raw_mean, weights=weights) if len(usable) else np.nan,
                                    "normalized_rula_pct": np.average(usable.normalized_rula_pct, weights=weights) if len(usable) else np.nan})
    participants = pd.DataFrame(participant_records)
    return procedures, participants, procedure_weighted_summary(procedures)


def run(project, plots=True):
    procedures, participants, cohort = build_summary(project)
    output = project.output("analysis", "clinical_summary")
    for name, frame in [
        ("procedure_summary", procedures),
        ("participant_summary", participants),
        ("cohort_summary", cohort)
    ]:
        write_csv(frame, output / (name + ".csv"))
    (output / "summary.txt").write_text(
        "IMU-Based Clinical Ergonomic Assessment\n\n"
        "Cohort means and weighted SDs summarize procedure-level means using procedure durations.\n"
        "Weighted SD is the square root of the duration-weighted mean squared deviation.\n"
        "Procedure duration is the last analyzed timestamp minus the first.\n"
        "Participant estimates use the same duration weights and are used for distribution plots.\n"
        f"Clinical reporting source: {project.clinical_summary_options().get('source', 'RULA score CSV files')}.\n"
        "Confirmed procedure-component exclusions are applied to whole-procedure mean summaries.\n"
        "Sensor interruption flags are detected for every procedure and recorded separately.\n"
        "Detection alone does not exclude a whole-procedure mean; sufficient usable coverage may remain.\n"
        "Confirmed manual windows are assessed against actual flat intervals, excluding outside-window failures.\n"
        "Regional scores retain available contralateral data under the existing scoring procedure.\n"
        "Direct score-derived means are retained in procedure_summary.csv as computed_raw_mean.\n"
        "Localized missing data affect only the corresponding component.\n\n" + cohort.to_string(index=False) + "\n", encoding="utf-8")
    if plots:
        import matplotlib.pyplot as plt
        keys = [
            "neck",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_arm",
            "right_lower_arm",
            "left_wrist",
            "right_wrist"
        ]
        labels = {component.key: component.label for component in COMPONENTS}
        distributions = [
            participants.loc[participants.component == key, "normalized_rula_pct"].dropna().to_numpy() for key in keys
        ]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.boxplot(distributions)
        ax.set_xticks(range(1, len(keys) + 1), [labels[key] for key in keys], rotation=20)
        ax.set(
            ylabel="Normalized RULA percentage (%)",
            ylim=(0, 100),
            title="IMU-Based Clinical Ergonomic Assessment"
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output / "participant_rula_distribution.png", dpi=200)
        fig.savefig(output / "participant_rula_distribution.svg")
        plt.close(fig)
    return output


def procedure_weighted_summary(procedures):
    """Combine configured procedure means with duration weights and population SDs."""
    records = []
    maxima = {component.key: component.maximum for component in COMPONENTS}
    for component, group in procedures.groupby("component", sort=False):
        usable = group.loc[group.raw_mean.notna() & group.procedure_duration_sec.gt(0)]
        weights = usable.procedure_duration_sec.to_numpy()
        if len(usable):
            mean = np.average(usable.raw_mean, weights=weights)
            sd = np.sqrt(np.average((usable.raw_mean - mean)**2, weights=weights))
        else:
            mean, sd = np.nan, np.nan
        records.append({"component": component, "participant_count": usable.participant.nunique(),
                        "procedure_count": len(usable), "procedure_duration_sec": weights.sum(),
                        "raw_mean": mean, "raw_sd": sd,
                        "normalized_rula_pct_mean": normalized_percent(mean, maxima[component]),
                        "normalized_rula_pct_sd": sd / (maxima[component] - 1) * 100})
    return pd.DataFrame(records)
