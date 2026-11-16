"""Align IMU and OMC segment orientations in a common reference frame."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return q
    return q / n

def quat_mult(q1, q2):
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

def quat_conj(q):
    w, x, y, z = q
    return [w, -x, -y, -z]

def quat_sign_inverse(q):
    w, x, y, z = q
    return [-w, -x, -y, -z]

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

def quat_from_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    rad = np.deg2rad(deg)
    s = np.sin(rad / 2.0)
    return normalize(np.array([np.cos(rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float))

def align_offset(path):
    processed = os.path.join(path, 'processed')
    if not os.path.isdir(processed):
        print(f'WARNING processed folder not found: {processed}')
        return
    omc_files = sorted(glob.glob(os.path.join(processed, '*_omc_q.csv')))
    if not omc_files:
        print(f'WARNING No *_omc_q.csv found (run align_coordinate first): {processed}')
        return
    for omc_f in omc_files:
        seg = os.path.basename(omc_f).replace('_omc_q.csv', '')
        imu_f = os.path.join(processed, f'{seg}_imu_q.csv')
        if not os.path.isfile(imu_f):
            print(f'SKIP align_offset: IMU file missing for {seg}: {imu_f}')
            continue
        df_imu = pd.read_csv(imu_f)
        df_omc = pd.read_csv(omc_f)
        if len(df_imu) == 0 or len(df_omc) == 0:
            print(f'SKIP align_offset: empty file for {seg}')
            continue
        imu_tcol = get_time_col(df_imu)
        omc_tcol = get_time_col(df_omc)
        imu_qcols = get_quat_cols(df_imu, seg, 'imu')
        omc_qcols = get_quat_cols(df_omc, seg, 'omc')
        df_imu = df_imu.copy()
        df_omc = df_omc.copy()
        df_imu[imu_tcol] = df_imu[imu_tcol].astype(float)
        df_omc[omc_tcol] = df_omc[omc_tcol].astype(float)
        df_imu = df_imu.sort_values(imu_tcol).reset_index(drop=True)
        df_omc = df_omc.sort_values(omc_tcol).reset_index(drop=True)
        imu_small = df_imu[[imu_tcol] + imu_qcols].rename(columns={imu_tcol: 't'})
        omc_small = df_omc[[omc_tcol] + omc_qcols].rename(columns={omc_tcol: 't'})
        merged = pd.merge(imu_small, omc_small, on='t', how='inner')
        if len(merged) < 2:
            print(f'WARNING align_offset: insufficient matched timestamps for {seg} (matched={len(merged)}).')
            continue
        q_imu0 = normalize(merged.loc[0, imu_qcols].to_numpy(dtype=float))
        name = os.path.basename(os.path.normpath(path))
        if name == 'wrist':
            axis_rot = 180
        else:
            axis_rot = -90.0
        q_align = quat_from_axis_angle([0, 0, 1], axis_rot)
        q0_aligned = normalize(quat_mult(q_align, q_imu0))
        Qimu = merged[imu_qcols].to_numpy(dtype=float)
        Qimu_rel = np.zeros_like(Qimu)
        for i in range(len(Qimu)):
            qi = normalize(Qimu[i])
            qi_aligned = normalize(quat_mult(q_align, qi))
            Qimu_rel[i] = normalize(quat_mult(qi_aligned, quat_conj(q0_aligned)))
        out_imu = pd.DataFrame({'timestamp': merged['t'].to_numpy(dtype=float)})
        for j, c in enumerate(imu_qcols):
            out_imu[c] = Qimu_rel[:, j]
        out_imu_path = os.path.join(processed, f'{seg}_imu_rel_q.csv')
        out_imu.to_csv(out_imu_path, index=False)
        print(f'   saved: {os.path.basename(out_imu_path)}')

def _sign_continuity(Q):
    Q = Q.copy()
    for k in range(1, len(Q)):
        if np.dot(Q[k], Q[k - 1]) < 0:
            Q[k:] *= -1.0
    return Q

def _remove_spikes(Q, threshold=0.8):
    Q = Q.copy()
    for k in range(1, len(Q)):
        dot = float(np.dot(Q[k], Q[k - 1]))
        if abs(dot) < threshold:
            Q[k] = Q[k - 1]
    return Q

def continuity(path, spike_threshold=0.8):
    continuity_one(path, suffix='_imu_rel_q.csv', kind='imu', spike_threshold=spike_threshold)
    continuity_one(path, suffix='_omc_rel_q.csv', kind='omc', spike_threshold=spike_threshold)

def continuity_one(path, suffix, kind, spike_threshold=0.8):
    processed = os.path.join(path, 'processed')
    if not os.path.isdir(processed):
        print(f'WARNING processed folder not found: {processed}')
        return
    files = sorted(glob.glob(os.path.join(processed, f'*{suffix}')))
    if not files:
        print(f'WARNING No *{suffix} found: {processed}')
        return
    for f in files:
        seg = os.path.basename(f).replace(suffix, '')
        df = pd.read_csv(f)
        qcols = get_quat_cols(df, seg, kind)
        Q = df[qcols].to_numpy(dtype=float)
        if len(Q) < 2:
            print(f'SKIP continuity: too short, skip {seg} ({os.path.basename(f)})')
            continue
        for i in range(len(Q)):
            Q[i] = normalize(Q[i])
        Q = _sign_continuity(Q)
        Q = _remove_spikes(Q, threshold=float(spike_threshold))
        for i in range(len(Q)):
            Q[i] = normalize(Q[i])
        df[qcols] = Q
        df.to_csv(f, index=False)
        print(f'OK continuity: {seg} -> overwritten {os.path.basename(f)} (threshold={spike_threshold})')

def quat_figure(path):
    quat_figure_one(path, suffix='_imu_rel_q.csv', tag='imu_rel_q')
    quat_figure_one(path, suffix='_omc_rel_q.csv', tag='omc_rel_q')

def quat_figure_one(path, suffix, tag):
    input_path = os.path.join(path, 'processed')
    output_dir = os.path.join(path, 'processed', 'figure')
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_path) if f.endswith(suffix)]
    if not csv_files:
        print(f'No CSV files found for {suffix}.')
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
        plt.title(f'{tag}: {csv_file}')
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
        plt.title(f'Overlay plot (W component) - {tag}')
        plt.legend()
        plt.grid(True)
        combined_path = os.path.join(output_dir, f'combined_{tag}.png')
        plt.savefig(combined_path)
        plt.close()
        print(f'Saved overlay plot: combined_{tag}.png')
    else:
        plt.close()
        print(f'No valid CSVs to plot for overlay ({tag}).')

def omc_modify(path, min_static_len_frames=100, motion_thresh_deg=1.0):
    processed = os.path.join(path, 'processed')
    if not os.path.isdir(processed):
        print(f'WARNING processed folder not found: {processed}')
        return
    omc_files = sorted(glob.glob(os.path.join(processed, '*_omc_q.csv')))
    if not omc_files:
        print(f'WARNING No *_omc_q.csv found (run omc2quat first): {processed}')
        return
    for omc_f in omc_files:
        seg = os.path.basename(omc_f).replace('_omc_q.csv', '')
        imu_rel_f = os.path.join(processed, f'{seg}_imu_rel_q.csv')
        if not os.path.isfile(imu_rel_f):
            print(f'SKIP omc_modify: IMU_REL file missing for {seg}: {imu_rel_f}')
            continue
        df_imu = pd.read_csv(imu_rel_f)
        df_omc = pd.read_csv(omc_f)
        if len(df_imu) == 0 or len(df_omc) == 0:
            print(f'SKIP omc_modify: empty file for {seg}')
            continue
        imu_tcol = get_time_col(df_imu)
        omc_tcol = get_time_col(df_omc)
        imu_qcols = get_quat_cols(df_imu, seg, 'imu')
        omc_qcols = get_quat_cols(df_omc, seg, 'omc')
        df_imu = df_imu.copy()
        df_omc = df_omc.copy()
        df_imu[imu_tcol] = df_imu[imu_tcol].astype(float)
        df_omc[omc_tcol] = df_omc[omc_tcol].astype(float)
        df_imu = df_imu.sort_values(imu_tcol).reset_index(drop=True)
        df_omc = df_omc.sort_values(omc_tcol).reset_index(drop=True)
        imu_small = df_imu[[imu_tcol] + imu_qcols].rename(columns={imu_tcol: 't'})
        omc_small = df_omc[[omc_tcol] + omc_qcols].rename(columns={omc_tcol: 't'})
        merged = pd.merge(imu_small, omc_small, on='t', how='inner')
        if len(merged) < 2:
            print(f'WARNING omc_modify: insufficient matched rows for {seg}')
            continue
        Qimu_raw = merged[imu_qcols].to_numpy(dtype=float)
        Qomc_raw = merged[omc_qcols].to_numpy(dtype=float)
        N = len(Qimu_raw)
        Qimu = np.zeros_like(Qimu_raw)
        Qomc = np.zeros_like(Qomc_raw)
        for i in range(N):
            Qimu[i] = normalize(Qimu_raw[i])
            Qomc[i] = normalize(Qomc_raw[i])
        motion_thresh_rad = np.radians(motion_thresh_deg)
        is_static = np.zeros(N, dtype=bool)
        is_static[0] = False
        for i in range(1, N):
            dot_imu = float(np.dot(Qimu[i], Qimu[i - 1]))
            dot_imu = abs(max(min(dot_imu, 1.0), -1.0))
            dtheta_imu = 2.0 * np.arccos(dot_imu)
            dot_omc = float(np.dot(Qomc[i], Qomc[i - 1]))
            dot_omc = abs(max(min(dot_omc, 1.0), -1.0))
            dtheta_omc = 2.0 * np.arccos(dot_omc)
            if dtheta_imu <= motion_thresh_rad and dtheta_omc <= motion_thresh_rad:
                is_static[i] = True
        static_indices = []
        start = None
        for i in range(N):
            if is_static[i]:
                if start is None:
                    start = i
            elif start is not None:
                if i - start >= min_static_len_frames:
                    static_indices.extend(range(start, i))
                start = None
        if start is not None and N - start >= min_static_len_frames:
            static_indices.extend(range(start, N))
        static_indices = np.array(static_indices, dtype=int)
        if static_indices.size == 0:
            print(f'WARNING omc_modify[{seg}]: no static segment >= {min_static_len_frames} frames -> fallback to first matched frame for q_corr.')
            static_indices = np.array([0], dtype=int)
        q_corr_list = []
        for idx in static_indices:
            q_imu_i = Qimu[idx]
            q_omc_i = Qomc[idx]
            q_corr_i = quat_mult(q_imu_i, quat_conj(q_omc_i))
            q_corr_list.append(normalize(q_corr_i))
        q_corr_arr = np.stack(q_corr_list, axis=0)
        ref = q_corr_arr[0]
        for i in range(1, len(q_corr_arr)):
            if float(np.dot(q_corr_arr[i], ref)) < 0.0:
                q_corr_arr[i] = -q_corr_arr[i]
        q_corr_mean = normalize(q_corr_arr.mean(axis=0))
        Qomc_full = df_omc[omc_qcols].to_numpy(dtype=float)
        Qomc_rel = np.zeros_like(Qomc_full)
        for i in range(len(Qomc_full)):
            qi = normalize(Qomc_full[i])
            Qomc_rel[i] = normalize(quat_mult(q_corr_mean, qi))
        df_out = df_omc.copy()
        for j, c in enumerate(omc_qcols):
            df_out[c] = Qomc_rel[:, j]
        out_path = os.path.join(processed, f'{seg}_omc_rel_q.csv')
        df_out.to_csv(out_path, index=False)
        print(f'omc_modify[{seg}]: N={N}, static_frames={static_indices.size}, q_corr_mean={q_corr_mean} -> saved {os.path.basename(out_path)}')
