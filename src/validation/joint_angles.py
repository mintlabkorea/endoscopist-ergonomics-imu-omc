"""Calculate relative joint angles and IMU-OMC angle differences."""

import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def find_two_files(workdir: str, pattern: str):
    files = [
        f for f in glob.glob(os.path.join(workdir, pattern)) if 'combined' not in os.path.basename(f).lower()
    ]
    files = sorted(files)
    if len(files) != 2:
        raise RuntimeError(f"Expected exactly 2 non-combined files for '{pattern}' in '{workdir}', found {len(files)}: {files}")
    return files

def detect_time_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'No time column found. Tried {candidates}. Columns: {list(df.columns)[:30]}')

def detect_quat_cols(df: pd.DataFrame, time_col: str):
    cols = [c for c in df.columns if c != time_col]
    w_cols = [c for c in cols if c.endswith('_w')]
    for w in w_cols:
        prefix = w[:-2]
        x, y, z = (f'{prefix}_x', f'{prefix}_y', f'{prefix}_z')
        if x in df.columns and y in df.columns and (z in df.columns):
            return [w, x, y, z]
    raise RuntimeError(f'Quaternion columns not found. Need <prefix>_w/_x/_y/_z. Found *_w candidates: {w_cols[:10]}')

def normalize_quat(q):
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n == 0, 1.0, n)

def quat_conj(q):
    qc = q.copy()
    qc[:, 1:] *= -1.0
    return qc

def quat_mul(qa, qb):
    w1, x1, y1, z1 = (qa[:, 0], qa[:, 1], qa[:, 2], qa[:, 3])
    w2, x2, y2, z2 = (qb[:, 0], qb[:, 1], qb[:, 2], qb[:, 3])
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=1)

def quat_angle_deg(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    dot = np.sum(q1 * q2, axis=1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return np.degrees(2 * np.arccos(np.abs(dot)))

def quat_rel_to_euler_xyz(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    q_rel = quat_mul(q2, quat_conj(q1))
    q_rel = normalize_quat(q_rel)
    q_xyzw = np.stack([q_rel[:, 1], q_rel[:, 2], q_rel[:, 3], q_rel[:, 0]], axis=1)
    r = R.from_quat(q_xyzw)
    euler_deg = r.as_euler('xyz', degrees=True)
    return euler_deg

def load_pair_angle(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    t1 = detect_time_col(df1, ['time', 'timestamp'])
    t2 = detect_time_col(df2, ['time', 'timestamp'])
    qcols1 = detect_quat_cols(df1, t1)
    qcols2 = detect_quat_cols(df2, t2)
    df1 = df1.rename(columns={t1: 't'})
    df2 = df2.rename(columns={t2: 't'})
    df1 = df1[['t'] + qcols1].rename(
        columns={qcols1[0]: 'q1_w', qcols1[1]: 'q1_x', qcols1[2]: 'q1_y', qcols1[3]: 'q1_z'}
    )
    df2 = df2[['t'] + qcols2].rename(
        columns={qcols2[0]: 'q2_w', qcols2[1]: 'q2_x', qcols2[2]: 'q2_y', qcols2[3]: 'q2_z'}
    )
    df = pd.merge(df1, df2, on='t', how='inner')
    q1 = df[['q1_w', 'q1_x', 'q1_y', 'q1_z']].to_numpy()
    q2 = df[['q2_w', 'q2_x', 'q2_y', 'q2_z']].to_numpy()
    angle_scalar = quat_angle_deg(q1, q2)
    euler_deg = quat_rel_to_euler_xyz(q1, q2)
    return (df['t'].to_numpy(), angle_scalar, euler_deg, os.path.basename(file1), os.path.basename(file2))

def execute(path):
    workdir = os.path.join(path, 'processed')
    if not os.path.isdir(workdir):
        print(f'WARNING processed folder not found: {workdir}')
        return
    result_dir = os.path.join(workdir, 'result')
    os.makedirs(result_dir, exist_ok=True)
    imu1, imu2 = find_two_files(workdir, '*_imu_rel_q.csv')
    omc1, omc2 = find_two_files(workdir, '*_omc_rel_q.csv')
    t_imu, angle_imu, euler_imu, imu_fn1, imu_fn2 = load_pair_angle(imu1, imu2)
    t_omc, angle_omc, euler_omc, omc_fn1, omc_fn2 = load_pair_angle(omc1, omc2)
    df_imu = pd.DataFrame(
        {
            't': t_imu,
            'angle_diff_imu_deg': angle_imu,
            'angle_x_imu_deg': euler_imu[:, 0],
            'angle_y_imu_deg': euler_imu[:, 1],
            'angle_z_imu_deg': euler_imu[:, 2]
        }
    )
    df_omc = pd.DataFrame(
        {
            't': t_omc,
            'angle_diff_omc_deg': angle_omc,
            'angle_x_omc_deg': euler_omc[:, 0],
            'angle_y_omc_deg': euler_omc[:, 1],
            'angle_z_omc_deg': euler_omc[:, 2]
        }
    )
    out = pd.merge(df_imu, df_omc, on='t', how='inner').rename(columns={'t': 'time'})

    def angle_wrap(diff):
        return (diff + 180.0) % 360.0 - 180.0
    out['err_x_deg'] = angle_wrap(out['angle_x_imu_deg'] - out['angle_x_omc_deg'])
    out['err_y_deg'] = angle_wrap(out['angle_y_imu_deg'] - out['angle_y_omc_deg'])
    out['err_z_deg'] = angle_wrap(out['angle_z_imu_deg'] - out['angle_z_omc_deg'])
    out_path = os.path.join(result_dir, 'angle_diff_imu_omc.csv')
    out.to_csv(out_path, index=False)
    diff_scalar = out['angle_diff_imu_deg'] - out['angle_diff_omc_deg']
    mean_error = diff_scalar.mean()
    mae_scalar = diff_scalar.abs().mean()
    std_scalar = diff_scalar.std(ddof=0)
    mean_x = out['err_x_deg'].mean()
    mean_y = out['err_y_deg'].mean()
    mean_z = out['err_z_deg'].mean()
    mae_x = out['err_x_deg'].abs().mean()
    mae_y = out['err_y_deg'].abs().mean()
    mae_z = out['err_z_deg'].abs().mean()
    std_x = out['err_x_deg'].std(ddof=0)
    std_y = out['err_y_deg'].std(ddof=0)
    std_z = out['err_z_deg'].std(ddof=0)
    print(f'OK Saved: {out_path}')
    print(f'IMU files: {imu_fn1} vs {imu_fn2}')
    print(f'OMC files: {omc_fn1} vs {omc_fn2}')
    print(f'Rows saved: {len(out)}')
    print(f'[Scalar] Mean Error (IMU - OMC) [deg]: {mean_error:.4f}')
    print(f'[Scalar] MAE [deg]: {mae_scalar:.4f}')
    print(f'[Scalar] STD [deg]: {std_scalar:.4f}')
    print('Per-axis angle error (IMU - OMC) [deg]:')
    print(f'  X: mean = {mean_x:.4f}, MAE = {mae_x:.4f}, STD = {std_x:.4f}')
    print(f'  Y: mean = {mean_y:.4f}, MAE = {mae_y:.4f}, STD = {std_y:.4f}')
    print(f'  Z: mean = {mean_z:.4f}, MAE = {mae_z:.4f}, STD = {std_z:.4f}')

def _format_session_block(session_name, df: pd.DataFrame):
    diff_scalar = df['angle_diff_imu_deg'] - df['angle_diff_omc_deg']
    mean_error = diff_scalar.mean()
    mae_scalar = diff_scalar.abs().mean()
    std_scalar = diff_scalar.std(ddof=0)
    mean_x = df['err_x_deg'].mean()
    mean_y = df['err_y_deg'].mean()
    mean_z = df['err_z_deg'].mean()
    mae_x = df['err_x_deg'].abs().mean()
    mae_y = df['err_y_deg'].abs().mean()
    mae_z = df['err_z_deg'].abs().mean()
    std_x = df['err_x_deg'].std(ddof=0)
    std_y = df['err_y_deg'].std(ddof=0)
    std_z = df['err_z_deg'].std(ddof=0)
    lines = []
    lines.append(f'[Session] {session_name}')
    lines.append('------------------------------------------------------------')
    lines.append(f'Rows              : {len(df)}')
    lines.append('Scalar angle error (IMU - OMC) [deg]')
    lines.append(f'  Mean            : {mean_error:8.4f}')
    lines.append(f'  MAE             : {mae_scalar:8.4f}')
    lines.append(f'  STD             : {std_scalar:8.4f}')
    lines.append('Per-axis angle error (IMU - OMC) [deg]')
    lines.append(f'  X: mean = {mean_x:8.4f}, MAE = {mae_x:8.4f}, STD = {std_x:8.4f}')
    lines.append(f'  Y: mean = {mean_y:8.4f}, MAE = {mae_y:8.4f}, STD = {std_y:8.4f}')
    lines.append(f'  Z: mean = {mean_z:8.4f}, MAE = {mae_z:8.4f}, STD = {std_z:8.4f}')
    lines.append('------------------------------------------------------------')
    return lines

def _parse_existing_summary(summary_path):
    if not os.path.isfile(summary_path):
        return {}
    with open(summary_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    blocks = {}
    current_name = None
    buffer = []
    for line in lines:
        if line.startswith('[Session] '):
            if current_name is not None:
                blocks[current_name] = buffer
            current_name = line[len('[Session] '):].strip()
            buffer = [line]
        elif current_name is not None:
            buffer.append(line)
    if current_name is not None:
        blocks[current_name] = buffer
    return blocks

def result_analysis(target_session=None):
    base_path = '../data'
    if not os.path.exists(base_path):
        print(f'WARNING Base path not found for result analysis: {base_path}')
        return
    summary_path = os.path.join(base_path, 'angle_diff_summary.txt')
    existing_blocks = _parse_existing_summary(summary_path)
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    subdirs = sorted(subdirs)
    new_blocks = {}
    if target_session is not None:
        if target_session not in subdirs:
            print(f"WARNING target_session '{target_session}' not found under {base_path}")
            return
        candidate_subdirs = [target_session]
    else:
        candidate_subdirs = subdirs
    required_cols = ['angle_diff_imu_deg', 'angle_diff_omc_deg', 'err_x_deg', 'err_y_deg', 'err_z_deg']
    for subdir in candidate_subdirs:
        result_csv = os.path.join(base_path, subdir, 'processed', 'result', 'angle_diff_imu_omc.csv')
        if not os.path.isfile(result_csv):
            continue
        df = pd.read_csv(result_csv)
        if not all((c in df.columns for c in required_cols)):
            print(f'WARNING Missing columns in {result_csv}, skipping.')
            continue
        new_blocks[subdir] = _format_session_block(subdir, df)
    for name, block in new_blocks.items():
        existing_blocks[name] = block
    if not existing_blocks:
        print('WARNING No sessions to summarize. Nothing to write.')
        return
    lines = []
    lines.append('============================================================')
    lines.append(' Joint Angle Error Summary (IMU vs OMC)')
    lines.append('============================================================')
    lines.append('')
    for name in sorted(existing_blocks.keys()):
        lines.extend(existing_blocks[name])
        lines.append('')
    dfs_global = []
    dfs_by_session = {}
    for name in sorted(existing_blocks.keys()):
        csv_path = os.path.join(base_path, name, 'processed', 'result', 'angle_diff_imu_omc.csv')
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if not all((c in df.columns for c in required_cols)):
            continue
        dfs_global.append(df)
        dfs_by_session[name] = df
    if dfs_global:
        df_all = pd.concat(dfs_global, axis=0, ignore_index=True)
        lines.append('***************** Global Summary (all sessions) *****************')
        global_block = _format_session_block('GLOBAL (all sessions)', df_all)
        lines.extend(global_block)
        lines.append('')
    target_axis_map = {'neck': ['x', 'y'], 'shoulder': ['x', 'y'], 'elbow': ['x'], 'wrist': ['x', 'y']}
    axis_col_map = {'x': 'err_x_deg', 'y': 'err_y_deg', 'z': 'err_z_deg'}
    pooled_vals = []
    pooled_x = []
    pooled_y = []
    pooled_z = []
    for sess_name, df in dfs_by_session.items():
        if sess_name not in target_axis_map:
            continue
        axes = target_axis_map[sess_name]
        for ax in axes:
            col = axis_col_map[ax]
            if col not in df.columns:
                continue
            vals = df[col].to_numpy()
            pooled_vals.append(vals)
            if ax == 'x':
                pooled_x.append(vals)
            elif ax == 'y':
                pooled_y.append(vals)
            elif ax == 'z':
                pooled_z.append(vals)
    if pooled_vals:
        v_all = np.concatenate(pooled_vals)
        mean_target = v_all.mean()
        mae_target = np.mean(np.abs(v_all))
        std_target = v_all.std(ddof=0)

        def _stat_or_nan(arr_list):
            if not arr_list:
                return (np.nan, np.nan, np.nan)
            v = np.concatenate(arr_list)
            return (v.mean(), np.mean(np.abs(v)), v.std(ddof=0))
        mean_x, mae_x, std_x = _stat_or_nan(pooled_x)
        mean_y, mae_y, std_y = _stat_or_nan(pooled_y)
        mean_z, mae_z, std_z = _stat_or_nan(pooled_z)
        lines.append('******** Target-DOF Summary (functionally used axes only) ********')
        lines.append('Included DOFs per joint:')
        lines.append('  neck    : x, y')
        lines.append('  shoulder: x, y')
        lines.append('  elbow   : x')
        lines.append('  wrist   : x, y')
        lines.append('')
        lines.append('Pooled target-DOF angle error (IMU - OMC) [deg]')
        lines.append(f'  Mean            : {mean_target:8.4f}')
        lines.append(f'  MAE             : {mae_target:8.4f}')
        lines.append(f'  STD             : {std_target:8.4f}')
        lines.append('Per-axis (within target DOFs only)')
        lines.append(f'  X: mean = {mean_x:8.4f}, MAE = {mae_x:8.4f}, STD = {std_x:8.4f}')
        lines.append(f'  Y: mean = {mean_y:8.4f}, MAE = {mae_y:8.4f}, STD = {std_y:8.4f}')
        lines.append(f'  Z: mean = {mean_z:8.4f}, MAE = {mae_z:8.4f}, STD = {std_z:8.4f}')
        lines.append('------------------------------------------------------------')
        lines.append('')
    summary_text = '\n'.join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print('\n' + summary_text)
    print(f'\nOK Summary saved to: {summary_path}')
