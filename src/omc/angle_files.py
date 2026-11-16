"""Organize and plot individual OMC joint angles."""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def sort_angles(path):
    input_file = os.path.join(path, 'combined_angle.csv')
    output_dir = os.path.join(path, 'each')
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_file):
        print(f'WARNING combined_angle.csv not found in {path}')
        return
    data = pd.read_csv(input_file)
    if data.shape[1] < 2:
        print(f'WARNING {input_file} has insufficient columns.')
        return
    first_column_header = data.columns[0]
    for col in data.columns[1:]:
        new_df = data[[first_column_header, col]]
        output_file = os.path.join(output_dir, f'{col}.csv')
        new_df.to_csv(output_file, index=False)
    print(f'OK Split done for {os.path.basename(path)}')

def angle_figure(path):
    input_path = os.path.join(path, 'each')
    output_dir = os.path.join(path, 'figure')
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    if not csv_files:
        print(f'WARNING No individual CSVs in {input_path}')
        return
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(input_path, csv_file))
        if df.shape[1] < 2:
            print(f'Skipping {csv_file}: less than two columns.')
            continue
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label=f'{csv_file}')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.title(f'Plot of {csv_file}')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.png')
        plt.savefig(output_path)
        plt.close()
    print(f'FIGURE Figures saved for {os.path.basename(path)}')
