# Quantitative Ergonomic Assessment of Endoscopists Using Wearable Inertial Measurement Units and Optical Motion Capture

Sang Hyun Kim<sup>1,†</sup>, Sunin Baek<sup>2,†</sup>, Benjamin Hur<sup>2</sup>, Younghoon Bae<sup>2</sup>, Yuhyeon Lee<sup>2</sup>, Hyuk Soon Choi<sup>3,*</sup>, and Daekyum Kim<sup>2,4,*</sup>

<sup>1</sup> Division of Gastroenterology, Department of Internal Medicine, Seoul St. Mary’s Hospital, College of Medicine, The Catholic University of Korea, Seoul, South Korea

<sup>2</sup> School of Mechanical Engineering, Korea University, Seoul, South Korea

<sup>3</sup> Division of Gastroenterology and Hepatology, Department of Internal Medicine, Korea University College of Medicine, Seoul, South Korea

<sup>4</sup> School of Smart Mobility, Korea University, Seoul, South Korea

<sup>†</sup> These authors contributed equally: Sang Hyun Kim and Sunin Baek.

<sup>*</sup> Correspondence: [Hyuk Soon Choi](mailto:mdkorea@gmail.com) and [Daekyum Kim](mailto:daekyum@korea.ac.kr).

## Abstract

Objective measurements of ergonomic burden during gastrointestinal endoscopy and evidence supporting workstation recommendations remain limited. We characterized region-specific postural exposure during routine screening esophagogastroduodenoscopy (EGD) and evaluated how bed and monitor height affected posture during simulated EGD. Nine endoscopists wore inertial measurement units during routine screening EGD, and 29 procedures were analyzed. Five of these endoscopists also completed 75 simulated EGD trials under controlled bed- and monitor-height conditions using optical motion capture. Joint kinematics were converted into Rapid Upper Limb Assessment (RULA) scores. During routine EGD, the mean overall RULA score was 5.60 (SD 0.30), and scores fell in the moderate or high categories for 94.4% of procedure time. The wrist had the highest normalized joint-level RULA score (71.7%, SD 12.4%), while neck scores remained in the moderate category for 97.3% of procedure time. In simulated EGD, raising the bed toward elbow height reduced the normalized lower-arm RULA score by 6.6 percentage points per level (P = 0.019). Raising the monitor increased neck extension by 5.1° per level (P = 0.002), without a corresponding significant trend in neck RULA scores. Continuous motion measurements can complement standardized ergonomic scores by revealing posture changes that those scores may not capture.

## Code

Python code for IMU and OMC processing, continuous RULA scoring, joint-angle validation, and ergonomic analyses.

```text
endo_erg.py       # Program entry point
src/
  imu/           # IMU processing
  omc/           # OMC processing
  validation/    # IMU–OMC comparison
  common/        # Shared processing and RULA scoring
  analysis/      # Summaries and statistical analyses
```

### Usage

```bash
python endo_erg.py
```

The interactive menu provides processing, validation, and analysis options. For command-line options, run `python endo_erg.py --help`.

### Data

Place compatible inputs under `data/imu/`, `data/omc/`, and `data/validation/`. Generated outputs are saved under `results/`.

Data, generated results, and recording-specific configuration files are not included in this repository.
