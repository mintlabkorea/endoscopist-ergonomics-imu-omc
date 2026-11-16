"""Store generated recording outputs in the canonical result layout."""

from pathlib import Path

from .io import copy_folder, copy_file


def export_imu(source, destination, records, resampled=None, overwrite=False):
    source, destination = Path(source), Path(destination)
    if resampled:
        copy_file(Path(resampled) / "sensor_data_100hz.csv",
                      destination / "sensor/resampled/sensor_data_100hz.csv", records, overwrite)
        copy_folder(Path(resampled) / "each", destination / "sensor/resampled/by_sensor", records, overwrite)
    copy_file(source / "0. switch/switch.csv", destination / "sensor/switch.csv", records, overwrite)
    copy_folder(source / "1. imu/each", destination / "sensor/filtered/by_sensor", records, overwrite)
    copy_folder(source / "1. imu/figure", destination / "sensor/filtered/figures", records, overwrite)
    copy_file(
        source / "2. quaternion/combined_q.csv",
        destination / "quaternion/quaternions.csv",
        records,
        overwrite
    )
    copy_folder(source / "2. quaternion/each", destination / "quaternion/by_segment", records, overwrite)
    copy_folder(source / "2. quaternion/figure", destination / "quaternion/figures", records, overwrite)
    angles = source / "3. angle"
    copy_file(angles / "combined_angle.csv", destination / "angle/full/angles.csv", records, overwrite)
    each = angles / "each"
    if each.is_dir():
        for path in sorted(each.glob("*.csv")):
            if path.stem.endswith("_fixed"):
                continue
            copy_file(path, destination / "angle/full/by_angle" / path.name, records, overwrite)
            corrected = path.with_name(path.stem + "_fixed.csv")
            copy_file(corrected if corrected.exists() else path,
                          destination / "angle/valid/by_angle" / path.name, records, overwrite)
            copy_file(angles / "figure" / (path.stem + ".png"),
                          destination / "angle/full/figures" / (path.stem + ".png"), records, overwrite)
            figure = angles / "figure" / (path.stem + ("_fixed.png" if corrected.exists() else ".png"))
            copy_file(figure, destination / "angle/valid/figures" / (path.stem + ".png"), records, overwrite)
    copy_file(angles / "valid_angles.csv", destination / "angle/valid/angles.csv", records, overwrite)
    export_rula(
        source / "4. rula/log",
        source / "4. rula/analysis.txt",
        destination / "rula",
        records,
        overwrite
    )
    for name in ["error_log.txt", "data_modification_log.txt"]:
        copy_file(source / name, destination / "logs" / name, records, overwrite)


def export_rula(folder, report, destination, records, overwrite=False):
    folder, destination = Path(folder), Path(destination)
    copy_file(folder / "rula_conversion.csv", destination / "rula_inputs.csv", records, overwrite)
    copy_file(folder / "rula_score.csv", destination / "rula_scores.csv", records, overwrite)
    copy_file(report, destination / "summary.txt", records, overwrite)
    copy_folder(folder / "plots", destination / "figures", records, overwrite)


def export_omc(source, destination, records, motion=None, overwrite=False, full_angles=None):
    source, destination = Path(source), Path(destination)
    copy_file(source / "combined_angle.csv", destination / "angle/valid/angles.csv", records, overwrite)
    copy_folder(source / "each", destination / "angle/valid/by_angle", records, overwrite)
    copy_folder(source / "figure", destination / "angle/valid/figures", records, overwrite)
    if full_angles:
        copy_file(full_angles, destination / "angle/full/angles.csv", records, overwrite)
    export_rula(source / "rula", source / "rula/analysis.txt", destination / "rula", records, overwrite)
    if motion:
        copy_file(motion, destination / "hand_motion/kinematics.csv", records, overwrite)


def export_validation(source, destination, records, overwrite=False):
    source, destination = Path(source), Path(destination)
    copy_file(
        source / "combined_imu_q.csv",
        destination / "quaternion/imu/quaternions.csv",
        records,
        overwrite
    )
    for kind in ["imu", "omc"]:
        for path in sorted(source.glob(f"*_{kind}_q.csv")):
            if path.name.startswith("combined_"):
                continue
            segment = path.name.removesuffix(f"_{kind}_q.csv")
            copy_file(path, destination / f"quaternion/{kind}/by_segment/{segment}.csv", records, overwrite)
        for path in sorted(source.glob(f"*_{kind}_rel_q.csv")):
            segment = path.name.removesuffix(f"_{kind}_rel_q.csv")
            copy_file(path, destination / f"aligned/{kind}/by_segment/{segment}.csv", records, overwrite)
    for path in sorted((source / "result").glob("*")):
        if path.name == "angle_diff_imu_omc.csv":
            target = destination / "angle_error/angles_and_errors.csv"
        elif path.name.endswith("_quat_angle_error.csv"):
            target = destination / "orientation_error" / path.name.replace("_quat_angle_error", "")
        elif path.name.endswith("_quat_angle_error_stats.txt"):
            target = destination / "orientation_error" / path.name.replace("_quat_angle_error_stats", "_summary")
        else:
            target = destination / "reports" / path.name
        copy_file(path, target, records, overwrite)
    copy_folder(source / "figure", destination / "figures", records, overwrite)
