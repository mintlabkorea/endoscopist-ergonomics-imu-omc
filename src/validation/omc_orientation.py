"""Estimate validation segment orientations from OMC marker positions."""

import glob
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_first_nonzero_timestamp_from_imu(event_csv_path: str) -> float:
    df = pd.read_csv(event_csv_path)
    THRESH_IMU_SQSUM = 5.0
    required = ['timestamp', 'Acc_X', 'Acc_Y', 'Acc_Z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{event_csv_path} missing columns: {missing}')
    acc_x = df['Acc_X'].to_numpy(dtype=float)
    acc_y = df['Acc_Y'].to_numpy(dtype=float)
    acc_z = df['Acc_Z'].to_numpy(dtype=float)
    sumsq = acc_x ** 2 + acc_y ** 2 + acc_z ** 2
    flags = (sumsq >= THRESH_IMU_SQSUM).astype(int)
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        raise RuntimeError(f'No non-zero detected in IMU condition (sumsq >= {THRESH_IMU_SQSUM}).')
    first_row = int(idx[0])
    return float(df.loc[first_row, 'timestamp'])

def find_first_nonzero_timestamp_from_omc(omc_csv_path: str) -> float:
    df = pd.read_csv(omc_csv_path)
    THRESH_OMC_AVGDIFF = 5.0
    required = ['timestamp', 'sync_x', 'sync_y', 'sync_z']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'{omc_csv_path} missing columns: {missing}')
    sx = df['sync_x'].to_numpy(dtype=float)
    sy = df['sync_y'].to_numpy(dtype=float)
    sz = df['sync_z'].to_numpy(dtype=float)
    dx = np.abs(sx[1:] - sx[:-1])
    dy = np.abs(sy[1:] - sy[:-1])
    dz = np.abs(sz[1:] - sz[:-1])
    avgdiff = (dx + dy + dz) / 3.0
    flags = (avgdiff >= THRESH_OMC_AVGDIFF).astype(int)
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        raise RuntimeError(f'No non-zero detected in OMC condition (avgdiff >= {THRESH_OMC_AVGDIFF}).')
    first_row = int(idx[0])
    return float(df.loc[first_row, 'timestamp'])

def get_offset_of_imu_and_omc(path):
    imu_path = os.path.join(path, 'event_detect.csv')
    if not os.path.isfile(imu_path):
        raise FileNotFoundError(f'event_detect.csv not found in {path}')
    omc_candidates = sorted(glob.glob(os.path.join(path, '*_omc.csv')))
    if len(omc_candidates) == 0:
        raise FileNotFoundError(f"No '*_omc.csv' file found in {path}")
    if len(omc_candidates) > 1:
        print("WARNING: Multiple '*_omc.csv' files found. Using the first one:")
        for p in omc_candidates:
            print('  -', os.path.basename(p))
    omc_path = omc_candidates[0]
    imu_synch_idx = find_first_nonzero_timestamp_from_imu(imu_path)
    omc_synch_idx = find_first_nonzero_timestamp_from_omc(omc_path)
    synch_offset = imu_synch_idx - omc_synch_idx
    print(f'imu_synch_idx  = {imu_synch_idx}')
    print(f'omc_synch_idx  = {omc_synch_idx}')
    print(f'synch_offset   = {synch_offset}   (imu_synch_idx - omc_synch_idx)')
    print(f'used_omc_file  = {os.path.basename(omc_path)}')
    return synch_offset
CANDIDATES = ['head', 'torso', 'humerus', 'p_radius', 'd_radius', 'hand']
CORNERS = ['ur', 'lr', 'll', 'ul']

def kabsch_rotation(P0: np.ndarray, Pt: np.ndarray) -> np.ndarray:
    H = P0.T @ Pt
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R

def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=float)
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q

def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ],
        dtype=float
    )

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 0 else q

def extract_points(df: pd.DataFrame, prefix: str, idx: int) -> np.ndarray:
    pts = []
    for c in CORNERS:
        pts.append([df.loc[idx, f'{prefix}_{c}_x'], df.loc[idx, f'{prefix}_{c}_y'], df.loc[idx, f'{prefix}_{c}_z']])
    return np.asarray(pts, dtype=float)

def has_all_columns(df: pd.DataFrame, prefix: str) -> bool:
    needed = []
    for c in CORNERS:
        for ax in ['x', 'y', 'z']:
            needed.append(f'{prefix}_{c}_{ax}')
    return all((col in df.columns for col in needed))

def get_module_quaternion(path, offset):
    omc_candidates = sorted(glob.glob(os.path.join(path, '*_omc.csv')))
    if len(omc_candidates) == 0:
        raise FileNotFoundError(f"No '*_omc.csv' file found in {path}")
    if len(omc_candidates) > 1:
        print("WARNING: Multiple '*_omc.csv' files found. Using the first one:")
        for p in omc_candidates:
            print('  -', os.path.basename(p))
    omc_path = omc_candidates[0]
    df = pd.read_csv(omc_path)
    present_prefixes = [p for p in CANDIDATES if has_all_columns(df, p)]
    if len(present_prefixes) == 0:
        raise ValueError('No module has complete marker columns.')
    print('Modules to be processed:', present_prefixes)
    n = len(df)
    out_dir = os.path.join(path, 'processed')
    os.makedirs(out_dir, exist_ok=True)
    ts = df['timestamp'].to_numpy(dtype=float) + float(offset)
    for prefix in present_prefixes:
        P0 = extract_points(df, prefix, 0)
        c0 = P0.mean(axis=0)
        P0c = P0 - c0
        if np.linalg.matrix_rank(P0c) < 2:
            print(f'[WARN] {prefix}: Degenerate marker configuration at t=0.')
        quats = np.zeros((n, 4), dtype=float)
        for i in range(n):
            Pt = extract_points(df, prefix, i)
            ct = Pt.mean(axis=0)
            Ptc = Pt - ct
            R_abs = kabsch_rotation(P0c, Ptc)
            q_abs = rotmat_to_quat_wxyz(R_abs)
            if i > 0 and np.dot(quats[i - 1], q_abs) < 0:
                q_abs = -q_abs
            quats[i] = quat_normalize(q_abs)
        out = pd.DataFrame({'timestamp': ts, f'{prefix}_omc_w': quats[:, 0], f'{prefix}_omc_x': quats[:, 1], f'{prefix}_omc_y': quats[:, 2], f'{prefix}_omc_z': quats[:, 3]})
        out_name = os.path.join(out_dir, f'{prefix}_omc_q.csv')
        out.to_csv(out_name, index=False)
        print(f'[OK] saved: {out_name}')

def quat_figure(path):
    input_path = os.path.join(path, 'processed')
    output_dir = os.path.join(path, 'processed', 'figure')
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_path) if f.endswith('_omc_q.csv')]
    if not csv_files:
        print('No CSV files found.')
        return
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(input_path, csv_file))
        if df.shape[1] < 2:
            print(f'Skipping {csv_file}: less than two columns.')
            continue
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label=f'Data from {csv_file}')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion W component')
        plt.title(f'Plot of {csv_file}')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved plot for {csv_file}')
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(input_path, csv_file))
        if df.shape[1] < 2:
            continue
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.plot(x, y, label=os.path.splitext(csv_file)[0])
        plotted_any = True
    if plotted_any:
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion W component')
        plt.title('Overlay plot (W component)')
        plt.legend()
        plt.grid(True)
        combined_path = os.path.join(output_dir, 'combined_omc_q.png')
        plt.savefig(combined_path)
        plt.close()
        print('Saved overlay plot: combined_omc_q.png')
    else:
        plt.close()
        print('No valid CSVs to plot for overlay.')
    print('Plots generated and saved successfully!')
