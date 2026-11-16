"""Processing orchestration with all writable working files under results."""

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import shutil
import traceback

from .common.io import input_inventory, processing_workspace, working_directory, timestamp, write_json
from .common.outputs import export_imu, export_omc, export_validation
from .paths import parse_omc_trial_id


def _copy_required(source, destination, names):
    missing = [name for name in names if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files in {source}: {', '.join(missing)}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / name, destination / name)


def _rula(folder, plots):
    from .common import rula_processing
    rula_processing.rula_convert(str(folder))
    rula_processing.rula_score(str(folder))
    if plots:
        rula_processing.rula_plot(str(folder))
    rula_processing.rula_analysis(str(folder))


def _split_angles(folder):
    import pandas as pd
    frame = pd.read_csv(folder / "combined_angle.csv")
    each = folder / "each"
    each.mkdir(parents=True, exist_ok=True)
    for column in frame.columns[1:]:
        frame[[frame.columns[0], column]].to_csv(each / f"{column}.csv", index=False)


def process_imu(project, name, stage="full", plots=True):
    from .imu import preprocess, filtering, orientation, angles, quality
    from .imu.exclusions import affected_components, failures_in_window, prepare_analysis_angles
    from .plotting import regenerate_figures

    output = project.output("imu", name)
    if stage == "figures":
        return regenerate_figures(project, "imu", name)
    with processing_workspace(project, "imu") as work:
        raw = work / "data/raw" / name
        processed = work / "data/processed" / name
        (work / "data/processed").mkdir(parents=True)
        with working_directory(work / "code"):
            if stage == "full":
                sensors = ["torso", "head", "left_humerus", "left_proximal_radius", "left_distal_radius",
                           "left_hand", "right_humerus", "right_proximal_radius", "right_distal_radius", "right_hand"]
                _copy_required(project.data / "imu" / name, raw,
                               [s + "_imu.csv" for s in sensors] + ["switch.csv", "loading_time.txt"])
                for function in [preprocess.rename, preprocess.create_temp, preprocess.create_csv0,
                                 preprocess.merge_data, preprocess.delete_cache, preprocess.interpolate_data,
                                 preprocess.convert_to_100hz, preprocess.sort_data, preprocess.organize_data,
                                 preprocess.copy_to_processed_data, preprocess.data_verification,
                                 preprocess.remove_extra_clicks]:
                    function(str(raw))
                filtering.create_subdir(str(processed))
                filtering.filter_imu(str(processed))
                filtering.remove_raw(str(processed))
                orientation.cal_quat(str(processed), orientation.get_loading_time(str(processed)))
                orientation.quat_consistency(str(processed))
                orientation.sort_quat(str(processed))
                if plots:
                    orientation.quat_figure(str(processed))
                angles.calculate_angles(str(processed))
                angles.sort_angles(str(processed))
            elif stage == "rula":
                angle_dir = processed / "3. angle"
                _copy_required(output / "angle/full", angle_dir, ["angles.csv"])
                (angle_dir / "angles.csv").rename(angle_dir / "combined_angle.csv")
                _split_angles(angle_dir)
                source = output / "sensor/filtered/by_sensor"
                if not source.is_dir() or not list(source.glob("*.csv")):
                    raise FileNotFoundError(f"Filtered sensor results are missing: {source}")
                shutil.copytree(source, processed / "1. imu/each")
                (processed / "1. imu/figure").mkdir(parents=True)
            else:
                raise ValueError(f"Unsupported IMU processing stage: {stage}")
            quality.write_log(str(processed))
            failures = quality.check_imu_data(str(processed))
            valid_time = project.time_limits().get(name)
            valid = prepare_analysis_angles(processed / "3. angle", failures, valid_time)
            active = failures_in_window(failures, float(valid.time.min()), float(valid.time.max()),
                                        manually_validated=valid_time is not None)
            for failure in active:
                quality.log_on_the_error(str(processed), failure["sensor"] + ".csv", failure["excluded_from_sec"])
            if plots:
                angles.angle_figure(str(processed))
            _rula(processed, plots)
            records = []
            export_imu(processed, output, records, overwrite=True,
                       resampled=raw / "sorted_result" if stage == "full" else None)
            checks = work / "data/processed/log.txt"
            if checks.is_file():
                shutil.copy2(checks, output / "logs/quality_checks.txt")
            write_json(output / "logs/exclusions.json",
                       {"valid_time_sec": valid_time, "automatic_failures": failures,
                        "active_failures": active, "detected_summary_components": affected_components(active),
                        "excluded_summary_components": project.clinical_summary_options().get(
                            "excluded_components", {}).get(name, []),
                        "detector": {"channel": "Acc_X", "minimum_flat_samples": 500,
                                     "sampling_rate_hz": 100, "safety_margin_sec": quality.SAFETY_MARGIN_SEC}})
    return {"files_written": len(records)}


def process_omc(project, name, stage="full", plots=True):
    from .omc import extraction, cropping, angle_files, motion_extraction
    from .plotting import regenerate_figures

    parse_omc_trial_id(name)
    output = project.output("omc", name)
    if stage == "figures":
        return regenerate_figures(project, "omc", name)
    with processing_workspace(project, "omc") as work:
        processed = work / "data/processed" / name
        processed.mkdir(parents=True)
        full_angles = None
        motion = None
        with working_directory(work / "code"):
            if stage == "full":
                raw = work / "data/raw"
                _copy_required(project.data / "omc", raw, [name + ".csv"])
                combined = processed / "combined_angle.csv"
                extraction.process_one(raw / (name + ".csv"), combined)
                full_angles = work / "full_angles.csv"
                shutil.copy2(combined, full_angles)
                cropping.process_file(combined)
                motion_extraction.OUT_DIR = work / "data/accel"
                motion_extraction.OUT_DIR.mkdir(parents=True)
                motion_extraction.process_one(raw / (name + ".csv"))
                motion = motion_extraction.OUT_DIR / (name + ".csv")
            elif stage == "rula":
                _copy_required(output / "angle/valid", processed, ["angles.csv"])
                (processed / "angles.csv").rename(processed / "combined_angle.csv")
            else:
                raise ValueError(f"Unsupported OMC processing stage: {stage}")
            angle_files.sort_angles(str(processed))
            if plots:
                angle_files.angle_figure(str(processed))
            _rula(processed, plots)
            records = []
            export_omc(processed, output, records, motion=motion, overwrite=True, full_angles=full_angles)
    return {"files_written": len(records)}


def process_validation(project, name, stage="full", plots=True):
    from .validation import imu_orientation, omc_orientation, alignment, comparison, joint_angles
    from .plotting import regenerate_figures

    output = project.output("validation", name)
    if stage == "figures":
        return regenerate_figures(project, "validation", name)
    with processing_workspace(project, "validation") as work:
        folder = work / "data" / name
        processed = folder / "processed"
        processed.mkdir(parents=True)
        with working_directory(work / "code"):
            if stage == "full":
                source = project.data / "validation" / name
                names = [p.name for p in source.glob("*.csv")]
                if len(names) < 4:
                    raise FileNotFoundError(f"Validation inputs are incomplete: {source}")
                _copy_required(source, folder, names)
                imu_orientation.cal_quat(str(folder))
                imu_orientation.quat_consistency(str(folder))
                imu_orientation.sort_quat(str(folder))
                omc_orientation.get_module_quaternion(str(folder), omc_orientation.get_offset_of_imu_and_omc(str(folder)))
                alignment.align_offset(str(folder))
                alignment.omc_modify(str(folder))
                alignment.continuity(str(folder))
                if plots:
                    imu_orientation.quat_figure(str(folder))
                    omc_orientation.quat_figure(str(folder))
                    alignment.quat_figure(str(folder))
            elif stage == "compare":
                for kind in ["imu", "omc"]:
                    source = output / "aligned" / kind / "by_segment"
                    files = list(source.glob("*.csv"))
                    if len(files) != 2:
                        raise FileNotFoundError(f"Expected two aligned segment files: {source}")
                    for path in files:
                        shutil.copy2(path, processed / f"{path.stem}_{kind}_rel_q.csv")
            else:
                raise ValueError(f"Unsupported validation processing stage: {stage}")
            comparison.verify(str(folder))
            joint_angles.execute(str(folder))
            records = []
            export_validation(processed, output, records, overwrite=True)
    return {"files_written": len(records)}


def run_processing(project, kind, ids=None, stage="full", plots=True, progress=print):
    from .plotting import plot_policy
    before = input_inventory(project)
    available = project.input_ids(kind)
    selected = list(ids or available)
    if not selected:
        raise FileNotFoundError(f"No {kind} inputs were found under {project.data / kind}.")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate target IDs are not allowed.")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown {kind} targets: {', '.join(unknown)}")
    operation = {"imu": process_imu, "omc": process_omc, "validation": process_validation}[kind]
    results = []
    for index, name in enumerate(selected, 1):
        log = project.output(kind, name, "logs", "processing.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log.open("a", encoding="utf-8") as handle, redirect_stdout(handle), redirect_stderr(handle), plot_policy(plots):
                print(f"Started at {timestamp()}; stage={stage}")
                detail = operation(project, name, stage=stage, plots=plots)
            record = {"id": name, "status": "completed", "stage": stage, "detail": detail}
            progress(f"[{index}/{len(selected)}] {name}: Processing completed.")
        except Exception as exc:
            with log.open("a", encoding="utf-8") as handle:
                traceback.print_exc(file=handle)
            record = {"id": name, "status": "failed", "stage": stage, "error": str(exc)}
            progress(f"[{index}/{len(selected)}] {name}: Failed: {exc}")
        results.append(record)
    if input_inventory(project) != before:
        raise RuntimeError("Input data inventory changed during processing.")
    if kind == "validation" and stage != "figures" and any(record["status"] == "completed" for record in results):
        from .validation.summary import summarize
        summarize(project)
    return results
