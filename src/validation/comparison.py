"""Compare aligned segment orientations and summarize angular errors."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return q / n

def quat_conj(q):
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul(q1, q2):
    w1, x1, y1, z1 = (q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3])
    w2, x2, y2, z2 = (q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3])
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ],
        axis=-1
    )

def quat_angle_error_deg(q_est, q_gt):
    q_est = normalize(q_est)
    q_gt = normalize(q_gt)
    q_err = quat_mul(quat_conj(q_gt), q_est)
    w = np.clip(np.abs(q_err[:, 0]), -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return (np.degrees(ang), q_err)

def axis_alignment_report(q_imu_wxyz, q_omc_wxyz):
    Rimu = R.from_quat([q_imu_wxyz[1], q_imu_wxyz[2], q_imu_wxyz[3], q_imu_wxyz[0]]).as_matrix()
    Romc = R.from_quat([q_omc_wxyz[1], q_omc_wxyz[2], q_omc_wxyz[3], q_omc_wxyz[0]]).as_matrix()
    S = Rimu.T @ Romc
    return S

def get_time_col(df):
    for c in ['time', 'timestamp', 'Time', 'Timestamp']:
        if c in df.columns:
            return c
    return df.columns[0]

def get_quat_cols(df, segment, kind):
    cols = [f'{segment}_{kind}_{a}' for a in ['w', 'x', 'y', 'z']]
    if all((c in df.columns for c in cols)):
        return cols
    alt = [f'{kind}_{a}' for a in ['w', 'x', 'y', 'z']]
    if all((c in df.columns for c in alt)):
        return alt
    raise ValueError(f"Quaternion columns not found for segment='{segment}', kind='{kind}'. Tried {cols} and {alt}.")

def verify(path):
    processed = os.path.join(path, 'processed')
    if not os.path.isdir(processed):
        print(f'WARNING processed folder not found: {processed}')
        return
    fig_dir = os.path.join(processed, 'figure')
    res_dir = os.path.join(processed, 'result')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    omc_files = sorted(glob.glob(os.path.join(processed, '*_omc_rel_q.csv')))
    if not omc_files:
        print(f'WARNING No *_omc_rel_q.csv found in {processed}')
        return
    seg_names = [os.path.basename(f).replace('_omc_rel_q.csv', '') for f in omc_files]
    print(f'verify(): segments found = {seg_names}')
    for name in seg_names:
        imu_f = os.path.join(processed, f'{name}_imu_rel_q.csv')
        omc_f = os.path.join(processed, f'{name}_omc_rel_q.csv')
        if not os.path.isfile(imu_f):
            print(f'SKIP verify: IMU file missing for {name}: {imu_f}')
            continue
        if not os.path.isfile(omc_f):
            print(f'SKIP verify: OMC file missing for {name}: {omc_f}')
            continue
        df_imu = pd.read_csv(imu_f)
        df_omc = pd.read_csv(omc_f)
        if len(df_imu) == 0 or len(df_omc) == 0:
            print(f'SKIP verify: empty file for {name}')
            continue
        imu_tcol = get_time_col(df_imu)
        omc_tcol = get_time_col(df_omc)
        imu_qcols = get_quat_cols(df_imu, name, 'imu')
        omc_qcols = get_quat_cols(df_omc, name, 'omc')
        df_imu = df_imu.copy()
        df_omc = df_omc.copy()
        df_imu[imu_tcol] = df_imu[imu_tcol].astype(float)
        df_omc[omc_tcol] = df_omc[omc_tcol].astype(float)
        df_imu = df_imu.sort_values(imu_tcol).reset_index(drop=True)
        df_omc = df_omc.sort_values(omc_tcol).reset_index(drop=True)
        t0 = max(float(df_imu[imu_tcol].min()), float(df_omc[omc_tcol].min()))
        t1 = min(float(df_imu[imu_tcol].max()), float(df_omc[omc_tcol].max()))
        if t1 <= t0:
            print(f'WARNING verify: no overlapping time range for {name} (imu[{df_imu[imu_tcol].min()}, {df_imu[imu_tcol].max()}], omc[{df_omc[omc_tcol].min()}, {df_omc[omc_tcol].max()}])')
            continue
        df_imu_clip = df_imu[(df_imu[imu_tcol] >= t0) & (df_imu[imu_tcol] <= t1)][[imu_tcol] + imu_qcols].rename(columns={imu_tcol: 't'})
        df_omc_clip = df_omc[(df_omc[omc_tcol] >= t0) & (df_omc[omc_tcol] <= t1)][[omc_tcol] + omc_qcols].rename(columns={omc_tcol: 't'})
        df_imu_clip['t'] = df_imu_clip['t'].round(3)
        df_omc_clip['t'] = df_omc_clip['t'].round(3)
        df = pd.merge(df_imu_clip, df_omc_clip, on='t', how='inner')
        if len(df) < 5:
            print(f'WARNING verify: insufficient merged rows for {name} (merged={len(df)}).')
            continue
        q_est = df[imu_qcols].to_numpy(dtype=float)
        q_gt = df[omc_qcols].to_numpy(dtype=float)
        q_est = normalize(q_est)
        q_gt = normalize(q_gt)
        for k in range(1, len(q_gt)):
            if float(np.dot(q_gt[k], q_gt[k - 1])) < 0:
                q_gt[k] = -q_gt[k]
            if float(np.dot(q_est[k], q_est[k - 1])) < 0:
                q_est[k] = -q_est[k]
        ang_err_deg, q_err = quat_angle_error_deg(q_est, q_gt)
        t = df['t'].to_numpy(dtype=float)
        plt.figure(figsize=(10, 5))
        plt.plot(t, ang_err_deg)
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title(f'Quaternion Angle Error (deg) | segment={name}')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle Error [deg]')
        plt.grid(True)
        plt.tight_layout()
        ang_fig_path = os.path.join(fig_dir, f'{name}_quat_angle_error.png')
        plt.savefig(ang_fig_path, dpi=150)
        plt.close()
        mae = float(np.mean(np.abs(ang_err_deg)))
        rmse = float(np.sqrt(np.mean(ang_err_deg ** 2)))
        mean = float(np.mean(ang_err_deg))
        std = float(np.std(ang_err_deg))
        p95 = float(np.percentile(np.abs(ang_err_deg), 95))
        lines = [f'=== Quaternion Angle Error Statistics (deg) | segment={name} ===', f'Matched rows = {len(df)}, overlap=[{t0:.6f}, {t1:.6f}]', f'mean={mean:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}, std={std:.3f}, |err|95%={p95:.3f}']
        log_path = os.path.join(res_dir, f'{name}_quat_angle_error_stats.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        out_csv = pd.DataFrame(
            {
                't': t,
                'angle_err_deg': ang_err_deg,
                'qerr_w': q_err[:, 0],
                'qerr_x': q_err[:, 1],
                'qerr_y': q_err[:, 2],
                'qerr_z': q_err[:, 3]
            }
        )
        out_csv_path = os.path.join(res_dir, f'{name}_quat_angle_error.csv')
        out_csv.to_csv(out_csv_path, index=False)
        print('\n' + '=' * 72)
        print(f'SUMMARY Quaternion Angle Error Summary | segment: {name}')
        print('-' * 72)
        print(f'Mean: {mean:8.3f} deg')
        print(f'MAE : {mae:8.3f} deg')
        print(f'RMSE: {rmse:8.3f} deg')
        print(f'STD : {std:8.3f} deg')
        print(f'|err|95%: {p95:8.3f} deg')
        print('-' * 72)
        print(f'Matched rows : {len(df)}')
        print(f'Time overlap : [{t0:.6f}, {t1:.6f}]')
        print('=' * 72 + '\n')
        print(f'OK verify: {name} -> saved')
        print(f'   FIGURE figure : {os.path.basename(ang_fig_path)}')
        print(f'   REPORT stats  : {os.path.basename(log_path)}')
        print(f'   CSV csv    : {os.path.basename(out_csv_path)}')
        components = ['w', 'x', 'y', 'z']
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), sharex=True)
        t = df['t'].to_numpy()
        for row, comp in enumerate(components):
            ax = axes[row]
            ax.plot(t, q_est[:, row], label='IMU', alpha=0.9)
            ax.plot(t, q_gt[:, row], label='OMC', alpha=0.7)
            ax.set_ylabel(comp)
            ax.set_ylim(-1.0, 1.0)
            if row == 0:
                ax.set_title(f'Quaternion Comparison: IMU vs OMC (segment={name})')
                ax.legend(loc='upper right')
            if row == 3:
                ax.set_xlabel('Time (s)')
            ax.grid(True)
        plt.tight_layout()
        q_fig_path = os.path.join(fig_dir, f'{name}_q_comparison.png')
        plt.savefig(q_fig_path, dpi=150)
        plt.close()
        print(f'   - {os.path.basename(q_fig_path)}')
    print('OK verify(): done for all segments.')
