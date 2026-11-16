# Endoscopist Ergonomics Analysis

Python code for continuous ergonomic assessment using inertial measurement units (IMU) and optical motion capture (OMC).

The workflow estimates joint angles, calculates Rapid Upper Limb Assessment (RULA) scores, validates IMU-derived angles against OMC, and summarizes clinical ergonomic burden and workstation-height effects. Numerical outcomes are calculated from supplied inputs. Recordings, generated results, participant lists, and recording-specific configuration are not distributed with the code.

## Repository layout

```text
endo_erg.py
src/
  imu/
  omc/
  validation/
  common/
  analysis/
```

The entry point resolves paths relative to its directory. Local inputs belong under `data/`; generated files and processing workspaces are written under `results/`.

```text
data/
  imu/<procedure>/
  omc/<trial>.csv
  validation/<joint>/
results/
  imu/<procedure>/
  omc/<trial>/
  validation/<joint>/
  analysis/
```

These local directories and the optional `config/` directory are excluded by `.gitignore`.

## Start

```bash
python endo_erg.py
```

The interactive menu provides IMU processing, OMC processing, joint-angle validation, summaries, and statistical analyses. Opening the menu does not start processing.

Inspect available inputs without processing:

```bash
python endo_erg.py --check
```

Use `python endo_erg.py --help` for available commands. To use inputs and results in another directory, place `--root "path/to/project"` before the command.

## Input conventions

### IMU

Procedure folders follow `p<participant>_<procedure>`. Each folder contains ten `<segment>_imu.csv` files, `switch.csv`, and `loading_time.txt`. Required segments are torso, head, and bilateral humerus, proximal radius, distal radius, and hand. Filenames and signal columns follow the readers in `src/imu/`.

### OMC

Trial filenames follow `g_<participant>_<monitor>_<bed>_<repeat>.csv`, with positive integer repetition numbers. Participant IDs are read from the filenames; no participant list or participant count is embedded in the code. Use nonempty participant IDs without underscores. Marker and segment prefixes in the exported CSV must match the participant token in its filename.

The ordered height conditions implement the experimental design:

| Experiment | Ordered conditions | Monitor / bed codes |
|---|---|---|
| Bed height | Low, Intermediate, Near-elbow | M_VL, M_L, M_M |
| Monitor height | Downward, Neutral, Upward | L_M, M_M, H_M |

The first code denotes monitor height and the second denotes bed height. The near-elbow/neutral reference is shared. Height-effect analysis requires repetitions 1, 2, and 3 for every configuration and participant. It combines those three repetitions; additional repetition numbers are not included in that analysis. Missing required trials raise an error. The analysis requires at least two participants for its inferential statistics.

The OMC readers expect the acquisition software's joint-angle and marker export structure. Sensor placement, coordinate conventions, and column names are part of the supported acquisition protocol; other formats require reader and mapping changes.

### Validation

Validation recordings are grouped under `neck`, `shoulder`, `elbow`, and `wrist`. Each group contains the IMU recordings and OMC exports expected by the readers in `src/validation/`.

## Process recordings

```bash
python endo_erg.py imu --stage full
python endo_erg.py omc --stage full
python endo_erg.py validation --stage full
```

IMU processing includes resampling, filtering, calibrated orientation estimation, relative joint-angle calculation, automatic signal exclusions, and RULA scoring. OMC processing extracts and analyzes joint angles and calculates RULA scores. Validation aligns segment orientations and compares joint angles.

To reuse intermediate results already in the corresponding `results/` folders:

```bash
python endo_erg.py imu --stage rula
python endo_erg.py omc --stage rula
python endo_erg.py validation --stage compare
```

Add `--ids` followed by recording IDs to select inputs, or `--no-plots` to omit optional figures. Use `--stage figures` to regenerate figures from existing results.

Recording outputs are grouped by stage, including `sensor`, `quaternion`, `angle`, and `rula`. Full and analyzed angles are stored separately. RULA conversion inputs and scores are saved as CSV files. Input files remain in `data/`.

## Summaries and statistical analyses

```bash
python endo_erg.py summary clinical
python endo_erg.py summary temporal
python endo_erg.py summary height-effects
python endo_erg.py summary hand-motion
python endo_erg.py summary all
```

Summary commands read existing recording results and write outputs under `results/analysis/`. They do not run the complete raw-data processing pipeline.

### Clinical summary

Procedure-level mean scores are combined across procedures using analyzed procedure durations. Duration is the last analyzed timestamp minus the first. Cohort SD is the square root of the duration-weighted mean squared deviation of procedure means from the pooled mean. This implements duration-weighted procedure aggregation, rather than equal weighting of participant means.

Separate duration-weighted estimates within each participant are used for participant distribution plots. Missing procedure-component means are omitted together with their weights. `procedure_summary.csv`, `participant_summary.csv`, and `cohort_summary.csv` contain the corresponding numerical results.

### Temporal ergonomic burden

Normalized RULA percentage is `(score - 1) / (maximum - 1) * 100`. Maxima are neck 6, upper arm 6, lower arm 3, wrist 4, and overall RULA 7. Descriptive categories are Minimal, Low, Moderate, and High, using boundaries at 0, 100/6, 50, 500/6, and 100 percent.

Joint-level proportions use each component's valid samples. The default overall denominator requires valid neck and bilateral upper-arm, lower-arm, and wrist scores (`all-parts`). The optional `--overall-valid-mode score-only` uses all available overall RULA scores instead. Sample durations include the final nominal sampling interval.

### Height effects

Repeated OMC trial means are combined within participant and condition using each trial's own analyzed duration. Condition summaries use arithmetic means and sample SDs across participants. For three equally spaced ordered conditions, each participant's slope is `(condition 3 - condition 1) / 2`; slopes are evaluated using two-sided one-sample t-tests against zero. Neck extension is the negative of signed `neck_Flex`. Trial and participant counts are determined from the inputs.

### Hand motion

IMU acceleration includes gravity and is expressed in m/s^2; OMC acceleration is expressed in mm/s^2. Angular velocity is expressed in degrees/s. IMU samples are restricted to the analyzed procedure interval.

## Scoring and signal quality

RULA angle thresholds, lookup tables, bilateral maximum scoring, and fixed non-kinematic scoring inputs are retained in `src/common/`. The non-kinematic inputs include trunk and leg posture, muscle use, external load, and arm support. These protocol settings must match the observation conditions when the code is applied to another acquisition.

Automatic IMU interruption detection identifies sustained flat filtered Acc_X signals using at least 500 samples at 100 Hz, a five-sample window, and a ten-second safety margin. Relevant intervals are excluded from dependent joint angles during processing. A detected interruption alone does not exclude an entire procedure-component mean from the clinical summary.

## Optional local configuration

The code can read `config/imu_valid_time.json` for reviewed procedure time limits and clinical summary settings. No configuration file or recording-specific values are included in this repository.

Procedure-ID entries contain inclusive `[start, end]` bounds in seconds, using original angle timestamps. The optional `clinical_summary` section supports `procedure_mean_decimals` and `excluded_components`. Exclusions map procedure IDs to component keys defined in `src/analysis/scores.py`. `source` is an optional descriptive label.

Without this file, no additional manual time limits or confirmed whole-procedure component exclusions are applied, and procedure means remain unrounded before aggregation. Automatic signal handling and existing missing scores remain active. For a manually reviewed time window, only actual flat-signal intervals overlapping that window produce active interruption flags; an outside interruption's safety margin does not invalidate the reviewed window.

Recording-specific quality decisions require a corresponding local configuration to reproduce. Such configuration belongs with the locally managed analysis inputs and is excluded from version control.

## Validation outputs

Validation results include aligned orientations, joint angles, and angular errors. `results/validation/metrics_summary.csv` reports axis-specific metrics and pooled target degrees of freedom: neck x/y, shoulder x/y, elbow x, and wrist x/y. Reported error SD is the population SD of signed angular errors.
