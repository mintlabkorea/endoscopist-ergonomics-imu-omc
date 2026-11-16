"""Estimate calibrated segment orientations from IMU recordings."""

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

def get_loading_time(path):
    base_name = os.path.basename(path)
    raw_dir = os.path.join('..', 'data', 'raw', base_name)
    log_path = os.path.join(raw_dir, 'loading_time.txt')
    with open(log_path, 'r') as loading_time_file:
        lines = loading_time_file.readlines()
        if len(lines) < 2:
            raise ValueError('The loading_time.txt file does not have enough lines.')
        load_start = float(lines[0].strip())
        load_end = float(lines[1].strip())
    difference = load_end - load_start
    load_timestamp = math.ceil(difference * 100) / 100.0
    start_index = int(load_timestamp * 100 + 2)
    return start_index

class cal_quat:

    def __init__(self, path, idx):
        self.base_path = path
        self.start_index = idx
        self.data_list = [
            'switch',
            'torso_imu',
            'head_imu',
            'left_humerus_imu',
            'left_proximal_radius_imu',
            'left_distal_radius_imu',
            'left_hand_imu',
            'right_humerus_imu',
            'right_proximal_radius_imu',
            'right_distal_radius_imu',
            'right_hand_imu'
        ]
        self.total_q = []
        self.sw = None
        self.sw_first_one = None
        self.sw_last_one = None
        self.execute()

    def execute(self):
        segment_quaternion_map = {
            'torso_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'head_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'left_humerus_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'left_proximal_radius_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'left_distal_radius_imu': np.array([0.5, 0.5, 0.5, -0.5]),
            'left_hand_imu': np.array([0.5, 0.5, 0.5, -0.5]),
            'right_humerus_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'right_proximal_radius_imu': np.array([math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2, 0.0]),
            'right_distal_radius_imu': np.array([0.5, -0.5, 0.5, 0.5]),
            'right_hand_imu': np.array([0.5, -0.5, 0.5, 0.5])
        }
        for i in range(len(self.data_list)):
            if i == 0:
                file_path = os.path.join(self.base_path, '0. switch', self.data_list[i] + '.csv')
            else:
                file_path = os.path.join(self.base_path, '1. imu/each', self.data_list[i] + '.csv')
            try:
                data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                if len(data) <= self.start_index:
                    print(f'File {self.data_list[i]} does not have enough data rows after blank timestamp. Skipping to the next file.')
                    continue
                data = data[self.start_index:]
                print(f'Data from {self.data_list[i]} processed successfully.')
            except IOError:
                print(f'File {self.data_list[i]} does not exist. Skipping to the next file.')
                continue
            timestamp = data[:, 0]
            if i == 0:
                sw = data[:, 1]
                self.sw = sw
                one_indices = np.where(self.sw == 1)[0]
                if len(one_indices) > 0:
                    self.sw_first_one = one_indices[0]
                    self.sw_last_one = one_indices[-1]
                else:
                    self.sw_first_one = None
                    self.sw_last_one = None
                df_switch = pd.DataFrame({'time': timestamp, 'switch': sw})
                continue
            quat = np.full((len(timestamp), 5), np.nan)
            accel = data[:, 1:4]
            gyro = data[:, 4:7]
            gyro_bias_from_data = data[599:1100, 4:7].mean(axis=0)
            start_idx = max(self.sw_first_one - 1000, 0)
            end_idx = self.sw_first_one
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
            fusion = IMUFusion(initial_quaternion=initial_q, gyro_bias=gyro_bias_from_data)
            for index in range(len(timestamp) - self.sw_last_one):
                idx = index + self.sw_last_one
                fusion.update(accel[idx], gyro[idx])
                quaternion = fusion.get_quaternion()
                quat[idx] = [timestamp[idx], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
            quat_df = pd.DataFrame(quat, columns=['time', f'{segment_name}_w', f'{segment_name}_x', f'{segment_name}_y', f'{segment_name}_z'])
            self.total_q.append(quat_df.dropna(subset=['time']))
        combined_data = df_switch
        for df in self.total_q:
            combined_data = pd.merge(combined_data, df, on='time')
        out_dir = os.path.join(self.base_path, '2. quaternion')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, 'combined_q.csv')
        combined_data.to_csv(save_path, index=False)
        print('Quat file has been created successfully and saved as combined_q.csv.')

def quat_consistency(path):
    input_file = os.path.join(path, '2. quaternion', 'combined_q.csv')
    if not os.path.exists(input_file):
        print(f'WARNING combined_q.csv not found: {input_file}')
        return
    df = pd.read_csv(input_file)
    segments = [
        'torso',
        'head',
        'left_humerus',
        'left_proximal_radius',
        'left_distal_radius',
        'left_hand',
        'right_humerus',
        'right_proximal_radius',
        'right_distal_radius',
        'right_hand'
    ]
    total_flips = 0
    for seg in segments:
        cols = [f'{seg}_imu_w', f'{seg}_imu_x', f'{seg}_imu_y', f'{seg}_imu_z']
        if not all((c in df.columns for c in cols)):
            print(f'SKIP Skipping {seg}: columns missing in combined_q.csv')
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
    input_file = os.path.join(path, '2. quaternion/combined_q.csv')
    output_dir = os.path.join(path, '2. quaternion/each')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    segments = [
        'torso',
        'head',
        'left_humerus',
        'left_proximal_radius',
        'left_distal_radius',
        'left_hand',
        'right_humerus',
        'right_proximal_radius',
        'right_distal_radius',
        'right_hand'
    ]
    for segment in segments:
        segment_columns = ['time'] + [f'{segment}_imu_{axis}' for axis in ['w', 'x', 'y', 'z']]
        segment_df = df[segment_columns]
        output_path = os.path.join(output_dir, f'{segment}_q.csv')
        segment_df.to_csv(output_path, index=False)
    print(f'Segment CSV files have been saved to {output_dir}.')

def quat_figure(path):
    input_path = os.path.join(path, '2. quaternion', 'each')
    output_dir = os.path.join(path, '2. quaternion', 'figure')
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
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
    print('Plots generated and saved successfully!')
