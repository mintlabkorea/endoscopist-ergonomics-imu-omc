"""Participant-level trends across ordered bed and monitor configurations."""

import numpy as np
import pandas as pd
from scipy import stats

from ..paths import natural_key, parse_omc_trial_id
from .scores import COMPONENTS, analyzed_duration, normalized_percent, read_angles, read_scores, write_csv


CONDITIONS = {
    "bed_height": [("low", "M", "VL"), ("intermediate", "M", "L"), ("near_elbow", "M", "M")],
    "monitor_height": [("downward", "L", "M"), ("neutral", "M", "M"), ("upward", "H", "M")],
}
TREND_COMPONENTS = ["neck", "upper_arm", "lower_arm", "wrist"]


def summarize_trial(project, name):
    scores = read_scores(project, "omc", name)
    angles = read_angles(project, "omc", name)
    duration = analyzed_duration(scores.time)
    if len(scores) != len(angles) or not np.allclose(scores.time, angles.time, rtol=0, atol=1e-9):
        raise ValueError(f"Analyzed angle and score timestamps differ: {name}")
    values = {component.key + "_raw": scores[component.column].mean() for component in COMPONENTS}
    if "neck_Flex" not in angles:
        raise ValueError(f"Neck flexion-extension angles are missing: {name}")
    values["neck_extension_deg"] = -pd.to_numeric(angles.neck_Flex, errors="raise").mean()
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError(f"Trial contains an outcome without valid values: {name}")
    values.update({column + "_deg": pd.to_numeric(angles[column], errors="raise").mean()
                   for column in angles.columns if column != "time"})
    return {"trial": name, "duration_sec": duration, **values}


def build_condition_summary(project):
    available = set(project.input_ids("omc"))
    participants = sorted({parse_omc_trial_id(name)["participant"] for name in available}, key=natural_key)
    if not participants:
        raise FileNotFoundError("No OMC height-condition inputs are available.")
    trial_cache, rows = {}, []
    for experiment, conditions in CONDITIONS.items():
        for code, (condition, monitor, bed) in enumerate(conditions, 1):
            for participant in participants:
                names = [f"g_{participant}_{monitor}_{bed}_{trial}" for trial in [1, 2, 3]]
                missing = sorted(set(names) - available)
                if missing:
                    raise FileNotFoundError("Required height-condition inputs are missing: " + ", ".join(missing))
                for name in names:
                    if name not in trial_cache:
                        trial_cache[name] = summarize_trial(project, name)
                trials = pd.DataFrame([trial_cache[name] for name in names])
                weights = trials.duration_sec.to_numpy()
                row = {"participant": participant,
                       "experiment": experiment, "condition": condition, "condition_code": code,
                       "trial_count": len(names), "trial_ids": ";".join(names),
                       "analyzed_duration_sec": float(weights.sum())}
                for component in COMPONENTS:
                    raw = np.average(trials[component.key + "_raw"], weights=weights)
                    row[component.key + "_raw"] = raw
                    row[component.key + "_pct"] = normalized_percent(raw, component.maximum)
                for column in trials.columns:
                    if column.endswith("_deg"):
                        row[column] = np.average(trials[column], weights=weights)
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(trial_cache.values())


def linear_trends(conditions):
    keys = ["participant", "experiment", "condition_code"]
    if conditions.duplicated(keys).any():
        raise ValueError("Duplicate participant-condition rows are not allowed.")
    records, individual = [], []
    outcomes = [(name, name + "_pct", "percentage_points_per_condition_step") for name in TREND_COMPONENTS]
    outcomes.append(("neck_extension", "neck_extension_deg", "degrees_per_condition_step"))
    for experiment in CONDITIONS:
        group = conditions.loc[conditions.experiment == experiment]
        if group.empty:
            raise ValueError(f"No condition means are available for {experiment}.")
        for participant, part in group.groupby("participant"):
            if set(part.condition_code) != {1, 2, 3}:
                raise ValueError(f"Incomplete ordered conditions for {experiment}/{participant}.")
        if set(group.participant) != set(conditions.participant):
            raise ValueError(f"Participant sets differ between experiments: {experiment}")
        for component, column, unit in outcomes:
            wide = group.pivot(index="participant", columns="condition_code", values=column)
            slopes = (wide[3] - wide[1]) / 2.0
            if not np.isfinite(slopes.to_numpy()).all() or len(slopes) < 2:
                raise ValueError(f"At least two valid participant slopes are required for {experiment}/{component}.")
            test = stats.ttest_1samp(slopes, popmean=0.0)
            records.append({"experiment": experiment, "component": component, "unit": unit,
                            "participant_count": len(slopes), "linear_trend": float(slopes.mean()),
                            "slope_sd": float(slopes.std(ddof=1)), "t_statistic": float(test.statistic),
                            "p_value": float(test.pvalue)})
            individual += [{"experiment": experiment, "component": component, "participant": participant,
                            "slope": float(slope), "unit": unit} for participant, slope in slopes.items()]
    return pd.DataFrame(records), pd.DataFrame(individual)


def run(project, plots=True):
    conditions, trials = build_condition_summary(project)
    trends, slopes = linear_trends(conditions)
    output = project.output("analysis", "height_effects")
    cohort = conditions.groupby(["experiment", "condition", "condition_code"], sort=False).agg(
        {column: ["mean", "std", "count"] for column in conditions.columns
         if column.endswith("_raw") or column.endswith("_pct") or column == "neck_extension_deg"})
    cohort.columns = ["_".join(column) for column in cohort.columns]
    for name, frame in [("height_condition_summary", conditions), ("trial_summary", trials),
                        ("condition_cohort_summary", cohort.reset_index()), ("linear_trends", trends),
                        ("participant_slopes", slopes)]:
        write_csv(frame, output / (name + ".csv"))
    (output / "summary.txt").write_text(
        "OMC-Based Laboratory Assessment: Bed- and Monitor-Height Effects\n\n"
        "Three trial means are combined using their own analyzed durations.\n"
        "The near-elbow/neutral reference is shared between the two experiments.\n"
        f"There are {len(trials)} unique trials and {len(conditions)} participant-condition analysis rows.\n"
        "Normalized RULA trends are expressed in percentage points per condition step.\n"
        "Positive cervical values indicate neck extension.\n"
        "Participant slopes are tested against zero using two-sided one-sample t-tests.\n\n"
        + trends.to_string(index=False) + "\n", encoding="utf-8")
    if plots:
        import matplotlib.pyplot as plt
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        for experiment, config in CONDITIONS.items():
            group = conditions.loc[conditions.experiment == experiment]
            for column in [name + "_pct" for name in TREND_COMPONENTS] + ["neck_extension_deg"]:
                fig, ax = plt.subplots(figsize=(7, 5))
                data = [group.loc[group.condition_code == code, column].to_numpy() for code in [1, 2, 3]]
                ax.boxplot(data)
                ax.set_xticks([1, 2, 3], [label.replace("_", " ").title() for label, _, _ in config])
                ax.set(ylabel="Neck extension (degrees)" if column == "neck_extension_deg" else "Normalized RULA percentage (%)",
                       title=column.replace("_", " ").title(), xlabel=experiment.replace("_", " ").title())
                fig.tight_layout()
                fig.savefig(figures / f"{experiment}_{column}.png", dpi=300)
                fig.savefig(figures / f"{experiment}_{column}.svg")
                plt.close(fig)
        _assessment_figures(conditions, figures)
    return output


def _assessment_figures(conditions, folder):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    keys = [
        "neck",
        "left_upper_arm",
        "right_upper_arm",
        "left_lower_arm",
        "right_lower_arm",
        "left_wrist",
        "right_wrist"
    ]
    labels = [
        "Neck",
        "Left upper\narm",
        "Right upper\narm",
        "Left lower\narm",
        "Right lower\narm",
        "Left wrist",
        "Right wrist"
    ]
    colors = ["#d9e6ed", "#78a7bf", "#275d7a"]
    centers = np.arange(len(keys))
    for experiment, config in CONDITIONS.items():
        group = conditions.loc[conditions.experiment == experiment]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for code, ((label, _, _), color) in enumerate(zip(config, colors), 1):
            values = [group.loc[group.condition_code == code, key + "_pct"].to_numpy() for key in keys]
            positions = centers + (code - 2) * 0.24
            boxes = ax.boxplot(values, positions=positions, widths=0.2, patch_artist=True, manage_ticks=False,
                               medianprops={"color": "black"}, showfliers=False)
            for patch in boxes["boxes"]:
                patch.set_facecolor(color)
            for position, samples in zip(positions, values):
                ax.scatter(position + np.linspace(-0.03, 0.03, len(samples)), samples,
                           color="black", s=10, alpha=0.7, zorder=3)
        ax.set_xticks(centers, labels)
        ax.set(ylabel="Normalized RULA percentage (%)", ylim=(0, 105),
               title=experiment.replace("_", " ").title())
        ax.legend(handles=[Patch(facecolor=color, edgecolor="black", label=label.replace("_", " ").title())
                           for (label, _, _), color in zip(config, colors)], ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, 1.15), frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        for extension in ["png", "svg"]:
            fig.savefig(folder / f"{experiment}_bilateral_rula.{extension}", dpi=300)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (experiment, config) in zip(axes, CONDITIONS.items()):
        group = conditions.loc[conditions.experiment == experiment]
        values = [
            group.loc[group.condition_code == code, "neck_extension_deg"].to_numpy() for code in [1, 2, 3]
        ]
        ax.boxplot(values)
        for _, participant in group.groupby("participant"):
            participant = participant.sort_values("condition_code")
            ax.plot(participant.condition_code, participant.neck_extension_deg, "o-", color="#617d8b",
                    markersize=3, linewidth=0.7, alpha=0.65)
        ax.set_xticks([1, 2, 3], [label.replace("_", " ").title() for label, _, _ in config])
        ax.set(title=experiment.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Neck extension (degrees)")
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(folder / f"neck_extension_conditions.{extension}", dpi=300)
    plt.close(fig)
