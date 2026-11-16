"""Estimate IMU orientations for joint-angle validation recordings."""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..common.fusion import IMUFusion

def mean_vec(vectors):
    sum_vector = np.sum(vectors, axis=0)
    magnitude = np.linalg.norm(sum_vector)
    if magnitude != 0:
        mean_vector = sum_vector / magnitude
    else:
        mean_vector = np.zeros_like(sum_vector)
    return mean_vector

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

class cal_quat:

    def __init__(self, path):
        self.base_path = path
        self.data_list = ['torso_imu', 'head_imu', 'humerus_imu', 'p_radius_imu', 'd_radius_imu', 'hand_imu']
        self.total_q = []
        self.execute()

    def execute(self):
        segment_quaternion_map = {
            'torso_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'head_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'humerus_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'p_radius_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'd_radius_imu': np.array([0.5, 0.5, 0.5, -0.5]),
            'hand_imu': np.array([0.5, 0.5, 0.5, -0.5])
        }
        for i in range(len(self.data_list)):
            file_path = os.path.join(self.base_path, self.data_list[i] + '.csv')
            try:
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            except IOError:
                print(f'File {self.data_list[i]} does not exist. Skipping to the next file.')
                continue
            timestamp = data[:, 0]
            accel = data[:, 1:4]
            gyro = data[:, 4:7]
            gyro_bias_from_data = data[599:1100, 4:7].mean(axis=0)
            dir_name = os.path.basename(os.path.normpath(self.base_path))
            if dir_name == 'neck':
                start_idx = 8000
                end_idx = 8500
            elif dir_name == 'shoulder':
                start_idx = 9000
                end_idx = 9500
            elif dir_name == 'elbow':
                start_idx = 9500
                end_idx = 10000
            elif dir_name == 'wrist':
                start_idx = 6800
                end_idx = 7300
            else:
                print(f'Unknown folder name: {dir_name} (expected neck/shoulder/elbow/wrist)')
                break
            accel_segment = accel[start_idx:end_idx]
            accel_mean_vec = mean_vec(accel_segment)
            gravity_vector = np.array([-1, 0, 0])
            v = np.cross(gravity_vector, accel_mean_vec)
            s = np.linalg.norm(v)
            c = np.dot(gravity_vector, accel_mean_vec)
            segment_name = self.data_list[i]
            if s == 0:
                initial_q = segment_quaternion_map[segment_name]
            else:
                v = v / s
                angle = np.arccos(c)
                w = np.cos(angle / 2)
                x, y, z = v * np.sin(angle / 2)
                calculated_q = np.array([w, x, y, z])
                predefined_q = segment_quaternion_map[segment_name]
                initial_q = quaternion_multiply(calculated_q, predefined_q)
                initial_q = initial_q / np.linalg.norm(initial_q)
                ref_q = predefined_q
                dot = abs(np.dot(initial_q, ref_q))
                offset_deg = 2 * np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
            n = len(timestamp) - end_idx
            quat = np.empty((n, 5))
            fusion = IMUFusion(initial_quaternion=initial_q, gyro_bias=gyro_bias_from_data)
            for k in range(n):
                idx = end_idx + k
                fusion.update(accel[idx], gyro[idx])
                q = fusion.get_quaternion()
                quat[k] = [timestamp[idx], q[0], q[1], q[2], q[3]]
            quat_df = pd.DataFrame(quat, columns=['time', f'{segment_name}_w', f'{segment_name}_x', f'{segment_name}_y', f'{segment_name}_z'])
            quat_df = quat_df.drop_duplicates(subset=['time'], keep='first')
            self.total_q.append(quat_df)
        if len(self.total_q) == 0:
            print('No quaternion dataframes to merge.')
            return
        base = self.total_q[0].sort_values('time').reset_index(drop=True)
        combined_data = base[['time']].copy()
        for df in self.total_q:
            df = df.sort_values('time').reset_index(drop=True)
            if len(df) != len(combined_data):
                raise ValueError(f'Length mismatch: base={len(combined_data)} vs df={len(df)}')
            if not np.allclose(df['time'].to_numpy(), combined_data['time'].to_numpy(), rtol=0, atol=1e-09):
                raise ValueError('Time vectors are not identical across sensors. Cannot simple-concat.')
            combined_data = pd.concat([combined_data, df.drop(columns=['time'])], axis=1)
        print('combined rows:', len(combined_data), 'unique time:', combined_data['time'].nunique())
        out_dir = os.path.join(self.base_path, 'processed')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, 'combined_imu_q.csv')
        combined_data.to_csv(save_path, index=False)
        print('Quat file has been created successfully and saved as combined_imu_q.csv.')

def quat_consistency(path):
    input_file = os.path.join(path, 'processed', 'combined_imu_q.csv')
    if not os.path.exists(input_file):
        print(f'WARNING combined_imu_q.csv not found: {input_file}')
        return
    df = pd.read_csv(input_file)
    segments = ['torso', 'head', 'humerus', 'p_radius', 'd_radius', 'hand']
    total_flips = 0
    for seg in segments:
        cols = [f'{seg}_imu_w', f'{seg}_imu_x', f'{seg}_imu_y', f'{seg}_imu_z']
        if not all((c in df.columns for c in cols)):
            print(f'SKIP Skipping {seg}: columns missing in combined_imu_q.csv')
            continue
        q = df[cols].to_numpy(dtype=float)
        if q.shape[0] < 2:
            continue
        flips = 0
        for k in range(1, len(q)):
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] *= -1.0
                flips += 1
        df.loc[:, cols] = q
        total_flips += flips
        print(f'OK {seg}: sign continuity fixed (flips = {flips})')
    df.to_csv(input_file, index=False)
    print(f'WRITE Overwritten with sign-consistent quaternions: {input_file} (total flips = {total_flips})')

def sort_quat(path):
    input_file = os.path.join(path, 'processed', 'combined_imu_q.csv')
    output_dir = os.path.join(path, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    segments = ['torso', 'head', 'humerus', 'p_radius', 'd_radius', 'hand']
    for segment in segments:
        segment_columns = ['time'] + [f'{segment}_imu_{axis}' for axis in ['w', 'x', 'y', 'z']]
        missing = [c for c in segment_columns if c not in df.columns]
        if missing:
            print(f'SKIP Skipping {segment}: columns missing: {missing}')
            continue
        segment_df = df[segment_columns]
        output_path = os.path.join(output_dir, f'{segment}_imu_q.csv')
        segment_df.to_csv(output_path, index=False)
    print(f'Segment CSV files have been saved to {output_dir}.')

def quat_figure(path):
    input_path = os.path.join(path, 'processed')
    output_dir = os.path.join(path, 'processed', 'figure')
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_path) if f.endswith('_imu_q.csv') and f != 'combined_imu_q.csv']
    if not csv_files:
        print('No per-segment CSV files found (excluding combined_imu_q.csv).')
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
        combined_path = os.path.join(output_dir, 'combined_imu_q.png')
        plt.savefig(combined_path)
        plt.close()
        print('Saved overlay plot: combined_imu_q.png')
    else:
        plt.close()
        print('No valid CSVs to plot for overlay.')
    print('Plots generated and saved successfully!')
