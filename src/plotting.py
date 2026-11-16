"""Headless figures for existing numerical outputs."""

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def plot_policy(enabled):
    import matplotlib.pyplot as plt
    original = plt.savefig
    if not enabled:
        plt.savefig = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.savefig = original
        plt.close("all")


def regenerate_figures(project, kind, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    root = project.output(kind, name)
    groups = []
    if kind == "imu":
        groups += [(root / "sensor/filtered/by_sensor", root / "sensor/filtered/figures")]
        groups += [(root / "quaternion/by_segment", root / "quaternion/figures")]
    if kind in ["imu", "omc"]:
        for mode in ["full", "valid"]:
            groups.append((root / f"angle/{mode}/by_angle", root / f"angle/{mode}/figures"))
        groups.append((root / "rula/rula_scores.csv", root / "rula/figures"))
    else:
        groups.append((root / "angle_error/angles_and_errors.csv", root / "figures"))
        groups.append((root / "orientation_error", root / "orientation_error/figures"))
    count = 0
    for source, output in groups:
        files = [source] if source.is_file() else sorted(source.glob("*.csv"))
        for path in files:
            df = pd.read_csv(path)
            if len(df.columns) < 2:
                continue
            output.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                for column in df.columns[1:]:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df.iloc[:, 0], df[column])
                    ax.set(xlabel="Time (s)", ylabel=column.replace("_", " "), title=column.replace("_", " "))
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(output / f"{column}.png", dpi=150)
                    plt.close(fig)
                    count += 1
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                for column in df.columns[1:]:
                    ax.plot(df.iloc[:, 0], df[column], label=column)
                ax.set(xlabel="Time (s)", title=path.stem.replace("_", " "))
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output / (path.stem + ".png"), dpi=150)
                plt.close(fig)
                count += 1
    if not count:
        raise FileNotFoundError(f"No numerical results are available for {kind}/{name}.")
    return count
