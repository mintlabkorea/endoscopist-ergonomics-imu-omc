"""English-only interactive menus and reproducible command-line entry points."""

import argparse
from pathlib import Path
import sys

from .paths import Project


def _configure_runtime():
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", line_buffering=True)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"


def status(project):
    counts = {kind: len(project.input_ids(kind)) for kind in ["imu", "omc", "validation"]}
    print("\nEndoscopist Ergonomics Analysis")
    print("-" * 64)
    print("Available inputs")
    print(f"  IMU-based clinical assessment       {counts['imu']:>3} procedures")
    print(f"  OMC-based laboratory assessment     {counts['omc']:>3} trials")
    print(f"  IMU-OMC joint-angle validation       {counts['validation']:>3} joints")
    print(f"\nProcedures with configured time limits: {len(project.time_limits())}")
    available_results = sum(project.output(kind, name, "rula/rula_scores.csv").is_file()
                            for kind in ["imu", "omc"] for name in project.input_ids(kind))
    print(f"Existing RULA results: {available_results}")
    print("\nInput: data/    Configuration: config/    Output: results/")
    print("-" * 64)
    return counts


def _ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "0"


def _summaries(project, selection, plots=True, overall_valid_mode="all-parts"):
    from .analysis import clinical, temporal, height_effects, hand_motion
    functions = {"clinical": clinical.run, "temporal": temporal.run,
                 "height-effects": height_effects.run, "hand-motion": hand_motion.run}
    selected = list(functions) if selection == "all" else [selection]
    for name in selected:
        print(f"Running summary analysis: {name}")
        options = {"overall_valid_mode": overall_valid_mode} if name == "temporal" else {}
        output = functions[name](project, plots=plots, **options)
        print(f"Output: {output}")


def _process(project, kind, ids=None, stage="full", plots=True):
    from .processing import run_processing
    records = run_processing(project, kind, ids=ids, stage=stage, plots=plots)
    failures = [record for record in records if record["status"] != "completed"]
    print(f"Completed: {len(records) - len(failures)}/{len(records)} targets")
    print(f"Output: {project.output(kind)}")
    if failures:
        raise RuntimeError(f"{len(failures)} target(s) failed. See the recording logs.")


def _processing_menu(project, kind):
    title = {"imu": "IMU-Based Clinical Assessment", "omc": "OMC-Based Laboratory Assessment",
             "validation": "IMU-OMC Joint-Angle Validation"}[kind]
    unit = {"imu": "procedures", "omc": "trials", "validation": "joints"}[kind]
    print(f"\n{title}\n" + "-" * 64)
    print(f"[1] Process all {unit}\n[2] Process selected {unit}\n[0] Back")
    choice = _ask("\nSelect an option: ")
    if choice == "0":
        return
    if choice not in ["1", "2"]:
        raise ValueError("Select 0, 1, or 2.")
    ids = None
    if choice == "2":
        available = project.input_ids(kind)
        print("Available IDs: " + ", ".join(available))
        ids = [part.strip() for part in _ask("Enter IDs, separated by commas: ").split(",") if part.strip()]
        if not ids:
            raise ValueError("At least one target ID is required.")
    print("\nSelect a processing stage:\n[1] Run the complete processing pipeline")
    if kind == "imu":
        print("[2] Apply exclusions and time limits, then update RULA")
    elif kind == "omc":
        print("[2] Update RULA from existing analyzed angles")
    else:
        print("[2] Compare existing aligned IMU and OMC orientations")
    print("[3] Regenerate figures from existing results\n[0] Back")
    stage_choice = _ask("\nSelect an option: ")
    if stage_choice == "0":
        return
    stages = {"1": "full", "2": "compare" if kind == "validation" else "rula", "3": "figures"}
    if stage_choice not in stages:
        raise ValueError("Select a valid processing stage.")
    _process(project, kind, ids, stages[stage_choice])


def _summary_menu(project):
    print("\nSummaries and Statistical Analyses\n" + "-" * 64)
    print("[1] Clinical summary\n[2] Temporal ergonomic burden\n[3] Bed- and monitor-height effects")
    print("[4] Hand motion\n[5] Run all summary analyses\n[0] Back")
    choice = _ask("\nSelect an option: ")
    if choice == "0":
        return
    options = {"1": "clinical", "2": "temporal", "3": "height-effects", "4": "hand-motion", "5": "all"}
    if choice not in options:
        raise ValueError("Select an analysis from 0 to 5.")
    _summaries(project, options[choice])


def interactive(project):
    while True:
        try:
            status(project)
            print("\n[1] IMU-based clinical assessment\n[2] OMC-based laboratory assessment")
            print("[3] IMU-OMC joint-angle validation\n[4] Summaries and statistical analyses")
            print("[5] Recompute all results from input data\n[0] Exit")
            choice = _ask("\nSelect an option: ")
            if choice == "0":
                print("Exiting.")
                return 0
            if choice in ["1", "2", "3"]:
                _processing_menu(project, {"1": "imu", "2": "omc", "3": "validation"}[choice])
            elif choice == "4":
                _summary_menu(project)
            elif choice == "5":
                for kind in ["imu", "omc", "validation"]:
                    _process(project, kind)
                _summaries(project, "all")
            else:
                raise ValueError("Select an option from 0 to 5.")
        except KeyboardInterrupt:
            print("\nOperation interrupted. Completed result files have been retained.")
        except Exception as exc:
            print(f"Error: {exc}")
        try:
            input("\nPress Enter to return to the main menu.")
        except EOFError:
            return 0


def parser():
    result = argparse.ArgumentParser(description="IMU and OMC endoscopist ergonomics analysis.")
    result.add_argument(
        "--root",
        type=Path,
        default=Project.default().root,
        help="Portable project directory."
    )
    result.add_argument(
        "--check",
        action="store_true",
        help="Inspect inputs and configuration without processing."
    )
    commands = result.add_subparsers(dest="command")
    for kind in ["imu", "omc", "validation"]:
        command = commands.add_parser(kind, help=f"Process {kind} inputs.")
        command.add_argument("--ids", nargs="+", help="Procedure, trial, or joint IDs.")
        command.add_argument(
            "--stage",
            default="full",
            choices=["full", "compare" if kind == "validation" else "rula", "figures"]
        )
        command.add_argument("--no-plots", action="store_true", help="Skip optional figure generation.")
    command = commands.add_parser("summary", help="Analyze existing canonical result files.")
    command.add_argument("analysis", choices=["clinical", "temporal", "height-effects", "hand-motion", "all"])
    command.add_argument("--no-plots", action="store_true", help="Skip optional figure generation.")
    command.add_argument("--overall-valid-mode", choices=["all-parts", "score-only"], default="all-parts",
                         help="Overall RULA validity for temporal summaries; default: all-parts.")
    command = commands.add_parser("full", help="Recompute every recording and all summary analyses.")
    command.add_argument("--no-plots", action="store_true", help="Skip optional figure generation.")
    return result


def main(argv=None):
    _configure_runtime()
    args = parser().parse_args(argv)
    project = Project(args.root)
    try:
        project.time_limits()
        from .analysis.clinical import summary_options
        summary_options(project)
        if args.check:
            status(project)
            return 0
        if args.command is None:
            return interactive(project)
        if args.command in ["imu", "omc", "validation"]:
            _process(project, args.command, args.ids, args.stage, not args.no_plots)
        elif args.command == "summary":
            _summaries(project, args.analysis, not args.no_plots, args.overall_valid_mode)
        elif args.command == "full":
            for kind in ["imu", "omc", "validation"]:
                _process(project, kind, plots=not args.no_plots)
            _summaries(project, "all", not args.no_plots)
        return 0
    except (KeyboardInterrupt, Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
