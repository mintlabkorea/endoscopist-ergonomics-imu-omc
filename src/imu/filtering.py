"""Filter IMU signals and organize sensor channels."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import glob

def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist_rate = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def create_subdir(path):
    if not path or not os.path.isdir(path):
        print(f'Invalid path: {path}')
        return
    imu_dir = os.path.join(path, '1. imu')
    if not os.path.exists(imu_dir):
        print(f'Directory not found: {imu_dir}')
        return
    subdirs = ['each', 'figure']
    for sub in subdirs:
        sub_path = os.path.join(imu_dir, sub)
        os.makedirs(sub_path, exist_ok=True)
        print(f'Created: {sub_path}')
    print("Subdirectories 'each' and 'figure' created successfully under '1. imu'.")

def filter_imu(path):
    data_list = [
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
    output_figure_dir = os.path.join(path, '1. imu/figure')
    os.makedirs(output_figure_dir, exist_ok=True)
    output_imu_dir = os.path.join(path, '1. imu/each')
    os.makedirs(output_imu_dir, exist_ok=True)
    for i in range(len(data_list)):
        file_path = os.path.join(path, '1. imu', data_list[i] + '.csv')
        try:
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            print(f'Data from {data_list[i]} processed successfully.')
        except IOError:
            print(f'File {data_list[i]} does not exist. Skipping to the next file.')
            continue
        timestamp = data[:, 0]
        gyroscope = data[:, 4:7]
        accelerometer = data[:, 1:4]
        Acc_X = accelerometer[:, 0]
        Acc_Y = accelerometer[:, 1]
        Acc_Z = accelerometer[:, 2]
        Gyr_X = gyroscope[:, 0]
        Gyr_Y = gyroscope[:, 1]
        Gyr_Z = gyroscope[:, 2]
        sample_rate = 100.0
        lowpass_cutoff_freq = 0.3
        Ft_Acc_X = lowpass_filter(Acc_X, lowpass_cutoff_freq, sample_rate)
        Ft_Acc_Y = lowpass_filter(Acc_Y, lowpass_cutoff_freq, sample_rate)
        Ft_Acc_Z = lowpass_filter(Acc_Z, lowpass_cutoff_freq, sample_rate)
        filtered_data = pd.DataFrame(
            {
                'timestamp': timestamp,
                'Acc_X': Ft_Acc_X,
                'Acc_Y': Ft_Acc_Y,
                'Acc_Z': Ft_Acc_Z,
                'Gyr_X': Gyr_X,
                'Gyr_Y': Gyr_Y,
                'Gyr_Z': Gyr_Z
            }
        )
        output_file_path = os.path.join(output_imu_dir, f'{data_list[i]}.csv')
        filtered_data.to_csv(output_file_path, index=False)
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(timestamp, Acc_X, label='Accel X (Raw)', color='r', alpha=0.5)
        plt.plot(timestamp, Ft_Acc_X, label='Accel X (Lowpass Filtered)', color='b', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Accel X (g)')
        plt.title('Accel X vs Time')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(timestamp, Acc_Y, label='Accel Y (Raw)', color='g', alpha=0.5)
        plt.plot(timestamp, Ft_Acc_Y, label='Accel Y (Lowpass Filtered)', color='b', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Accel Y (g)')
        plt.title('Accel Y vs Time')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(timestamp, Acc_Z, label='Accel Z (Raw)', color='b', alpha=0.5)
        plt.plot(timestamp, Ft_Acc_Z, label='Accel Z (Lowpass Filtered)', color='purple', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Accel Z (g)')
        plt.title('Accel Z vs Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_figure_path = os.path.join(output_figure_dir, f'accel_{data_list[i]}.png')
        plt.savefig(acc_figure_path)
        plt.close()
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(timestamp, Gyr_X, label='Gyro X (Raw)', color='r', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Gyro X (deg/s)')
        plt.title('Gyro X vs Time')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(timestamp, Gyr_Y, label='Gyro Y (Raw)', color='g', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Gyro Y (deg/s)')
        plt.title('Gyro Y vs Time')
        plt.legend()
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(timestamp, Gyr_Z, label='Gyro Z (Raw)', color='b', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Gyro Z (deg/s)')
        plt.title('Gyro Z vs Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        gyro_figure_path = os.path.join(output_figure_dir, f'gyro_{data_list[i]}.png')
        plt.savefig(gyro_figure_path)
        plt.close()

def remove_raw(path):
    if not path or not os.path.isdir(path):
        print(f'Invalid path: {path}')
        return
    imu_dir = os.path.join(path, '1. imu')
    if not os.path.exists(imu_dir):
        print(f'Directory not found: {imu_dir}')
        return
    csv_files = glob.glob(os.path.join(imu_dir, '*.csv'))
    if not csv_files:
        print("No CSV files found in '1. imu'.")
        return
    for file in csv_files:
        try:
            os.remove(file)
            print(f'Removed: {file}')
        except Exception as e:
            print(f'Failed to remove {file}: {e}')
    print("All .csv files in '1. imu' have been removed successfully.")
