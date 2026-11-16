"""Synchronize IMU channels and resample recordings to 100 Hz."""

import os
import pandas as pd
from decimal import Decimal, getcontext
import re
import glob
import shutil
import csv

def rename(selected_path):
    if not selected_path:
        print('No path provided to rename().')
        return
    if not os.path.isdir(selected_path):
        print(f'Not a directory: {selected_path}')
        return
    for filename in os.listdir(selected_path):
        if not filename.lower().endswith('.csv'):
            continue
        old_file = os.path.join(selected_path, filename)
        base, ext = os.path.splitext(filename)
        new_base = base
        if new_base.endswith(('_IMU', '_switch')):
            new_base = new_base.rsplit('_', 1)[0]
        if new_base.endswith('_IMU'):
            new_base = new_base[:-4] + '_imu'
        if 'Hand' in new_base:
            new_base = new_base.replace('Hand', 'hand')
        if 'Upper_Forearm' in new_base:
            new_base = new_base.replace('Upper_Forearm', 'proximal_radius')
        if 'Forearm' in new_base:
            new_base = new_base.replace('Forearm', 'distal_radius')
        if 'Upper_Arm' in new_base:
            new_base = new_base.replace('Upper_Arm', 'humerus')
        if '_(L)' in new_base:
            new_base = new_base.replace('_(L)', '')
            new_base = 'left_' + new_base
        if '_(R)' in new_base:
            new_base = new_base.replace('_(R)', '')
            new_base = 'right_' + new_base
        new_filename = new_base + ext
        if new_filename == filename:
            continue
        new_file = os.path.join(selected_path, new_filename)
        if os.path.exists(new_file):
            print(f'SKIP (exists): {filename} -> {new_filename}')
            continue
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

def create_temp(selected_path):
    if not selected_path or not os.path.isdir(selected_path):
        print(f'Invalid directory: {selected_path}')
        return
    file_paths = [
        os.path.join(selected_path, f) for f in os.listdir(selected_path) if f.lower().endswith('.csv')
    ]
    if not file_paths:
        print('No CSV files found in the selected directory.')
        return
    data_frames = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        base = os.path.splitext(os.path.basename(fp))[0]
        new_cols = [c if c == 'timestamps' else f'{base}_{c}' for c in df.columns]
        df.columns = new_cols
        data_frames.append(df)
    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on='timestamps', how='outer')
    merged_df.sort_values(by='timestamps', inplace=True)
    min_timestamps = merged_df['timestamps'].min()
    merged_df['timestamps'] = merged_df['timestamps'] - min_timestamps
    merged_df.fillna(0, inplace=True)
    timestamps_col = merged_df.pop('timestamps')
    merged_df.insert(0, 'timestamps', timestamps_col)
    out_path = os.path.join(selected_path, 'csv_temp.csv')
    merged_df.to_csv(out_path, index=False)
    print(f"CSV files have been successfully merged and saved as '{out_path}'.")

def create_csv0(selected_path):
    getcontext().prec = 10
    file_names = [os.path.join(selected_path, 'csv_temp.csv')]
    timestamps = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        timestamps.extend(df['timestamps'].values)
    min_timestamps = Decimal(min(timestamps))
    max_timestamps = Decimal(max(timestamps))
    new_timestamps = []
    current_timestamps = min_timestamps
    while current_timestamps < max_timestamps:
        new_timestamps.append(float(current_timestamps))
        current_timestamps += Decimal('0.0001')
    df_new = pd.DataFrame(new_timestamps, columns=['timestamps'])
    for i in range(1, 7):
        df_new[f'data{i}'] = 0
    out_path = os.path.join(selected_path, 'csv0.csv')
    df_new.to_csv(out_path, index=False)
    print("The file 'csv0.csv' has been created with precise timestamps and data columns filled with 0.")

def merge_data(selected_path):
    file_names = [os.path.join(selected_path, 'csv0.csv'), os.path.join(selected_path, 'csv_temp.csv')]
    data_frames = []
    for i, file_name in enumerate(file_names):
        df = pd.read_csv(file_name)
        new_column_names = [f'{col}' if col != 'timestamps' else 'timestamps' for col in df.columns]
        df.columns = new_column_names
        data_frames.append(df)
    merged_df = data_frames[0]
    for df in data_frames[1:]:
        merged_df = pd.merge(merged_df, df, on='timestamps', how='outer')
    merged_df.sort_values(by='timestamps', inplace=True)
    min_timestamps = merged_df['timestamps'].min()
    merged_df['timestamps'] = merged_df['timestamps'] - min_timestamps
    merged_df.fillna(0, inplace=True)
    timestamps_col = merged_df.pop('timestamps')
    merged_df.insert(0, 'timestamps', timestamps_col)
    out_path = os.path.join(selected_path, 'csv_merged.csv')
    merged_df.to_csv(out_path, index=False)
    print("CSV files have been successfully merged and saved as 'csv_merged.csv'.")

def delete_cache(selected_path):
    files_to_delete = [os.path.join(selected_path, 'csv0.csv'), os.path.join(selected_path, 'csv_temp.csv')]
    for file_name in files_to_delete:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f'File {file_name} has been deleted successfully.')
        else:
            print(f'File {file_name} does not exist and cannot be deleted.')
    file_to_clean = os.path.join(selected_path, 'csv_merged.csv')
    if not os.path.exists(file_to_clean):
        print(f'{file_to_clean} not found; skip column cleanup.')
        return
    df = pd.read_csv(file_to_clean)
    if df.shape[1] >= 7:
        cols_to_drop = df.columns[1:7]
        df.drop(columns=cols_to_drop, inplace=True)
        df.to_csv(file_to_clean, index=False)
        print("Columns B to G have been deleted from 'csv_merged.csv'.")
    else:
        print('csv_merged.csv has fewer than 7 columns; skip dropping B~G.')

def interpolate_data(selected_path):
    merged_path = os.path.join(selected_path, 'csv_merged.csv')
    if not os.path.exists(merged_path):
        print(f'{merged_path} not found; cannot interpolate.')
        return
    df = pd.read_csv(merged_path)
    data_columns = df.columns[1:]
    if 'switch_channel_1' in df.columns:
        df.rename(columns={'switch_channel_1': 'switch'}, inplace=True)
        switch_column = 'switch'
        data_columns = df.columns[1:]
    else:
        switch_column = None
        print("Warning: 'switch_channel_1' column not found in the CSV file.")
    for column in data_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df[data_columns] = df[data_columns].replace(0, pd.NA)
    df[data_columns] = df[data_columns].apply(pd.to_numeric, errors='coerce')
    if switch_column:
        data_columns = data_columns.drop(switch_column)
    df[data_columns] = df[data_columns].interpolate(method='linear')
    df[data_columns] = df[data_columns].bfill()
    df[data_columns] = df[data_columns].ffill()
    if switch_column:
        switch_values = df[switch_column].tolist()
        start_index = next((i for i, x in enumerate(switch_values) if pd.notna(x)), None)
        end_index = next((i for i, x in enumerate(reversed(switch_values)) if pd.notna(x)), None)
        end_index = len(switch_values) - end_index - 1 if end_index is not None else None
        if start_index is not None:
            for i in range(start_index):
                switch_values[i] = 0
        if end_index is not None:
            for i in range(end_index + 1, len(switch_values)):
                switch_values[i] = 0
        one_indices = [i for i, value in enumerate(switch_values) if value == 1]
        for idx in one_indices:
            start = max(idx - 200, 0)
            end = min(idx + 200, len(switch_values) - 1)
            for j in range(start, end + 1):
                switch_values[j] = 1
        switch_values = [0 if pd.isna(x) else x for x in switch_values]
        df[switch_column] = switch_values
    out_path = os.path.join(selected_path, 'csv_merged_interpolated.csv')
    df.to_csv(out_path, index=False)
    print(
        "Linear interpolation has been performed on the data columns and the file 'csv_merged_interpolated.csv' has been saved."
    )

def convert_to_100hz(selected_path):
    output_dir = os.path.join(selected_path, 'sorted_result')
    os.makedirs(output_dir, exist_ok=True)
    merged_interp_path = os.path.join(selected_path, 'csv_merged_interpolated.csv')
    if not os.path.exists(merged_interp_path):
        print(f'{merged_interp_path} not found; cannot downsample.')
        return
    timestamps = pd.read_csv(merged_interp_path, usecols=['timestamps'])
    epsilon = 1e-06
    indices_100hz = timestamps[
        (timestamps['timestamps'] % 0.01 < epsilon) | (timestamps['timestamps'] % 0.01 > 0.01 - epsilon)
    ].index
    filtered_timestamps_100hz = timestamps.loc[indices_100hz]
    filtered_df_100hz = pd.DataFrame()
    all_columns = pd.read_csv(merged_interp_path, nrows=0).columns
    data_columns = [col for col in all_columns if col != 'timestamps']
    for col in data_columns:
        print(f'Processing column: {col}')
        column_data = pd.read_csv(merged_interp_path, usecols=[col])
        filtered_col_100hz = column_data.loc[indices_100hz]
        if filtered_df_100hz.empty:
            filtered_df_100hz = filtered_timestamps_100hz.copy()
        filtered_df_100hz[col] = filtered_col_100hz[col].values
    out_path = os.path.join(output_dir, 'sensor_data_100hz.csv')
    filtered_df_100hz.to_csv(out_path, index=False)
    print("Filtered data saved to 'sorted_result/sensor_data_100hz.csv'.")
    df = pd.read_csv(out_path)
    df['timestamps'] = df['timestamps'].round(2)
    df = df.groupby('timestamps', as_index=False).mean()
    df.to_csv(out_path, index=False)
    print(
        'Timestamps have been rounded and duplicate rows merged by averaging. The updated file has been saved.'
    )
    files_to_delete = [
        os.path.join(selected_path, 'csv_merged_interpolated.csv'),
        os.path.join(selected_path, 'csv_merged.csv')
    ]
    for file_name in files_to_delete:
        if os.path.exists(file_name):
            os.remove(file_name)
            print(f'File {file_name} has been deleted successfully.')
        else:
            print(f'File {file_name} does not exist and cannot be deleted.')

def sort_data(selected_path):
    input_dir = os.path.join(selected_path, 'sorted_result')
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    output_dir = os.path.join(selected_path, 'sorted_result/each')
    os.makedirs(output_dir, exist_ok=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        base_filename = os.path.splitext(os.path.basename(csv_file))[0]
        column_groups = {}
        for col in df.columns:
            if col != 'timestamps':
                base_name = re.sub('_\\d+$', '', col)
                if base_name not in column_groups:
                    column_groups[base_name] = []
                column_groups[base_name].append(col)
        for base_name, columns in column_groups.items():
            group_df = df[['timestamps'] + columns]
            group_filename = f'{base_filename}_{base_name}.csv'
            group_file_path = os.path.join(output_dir, group_filename)
            group_df.to_csv(group_file_path, index=False)
            print(f'Saved {group_file_path}')
    print('All files have been processed and saved successfully.')

def organize_data(selected_path):
    base_path = os.path.join(selected_path, 'sorted_result/each')
    for file_name in os.listdir(base_path):
        if file_name.endswith('.csv'):
            new_name = file_name.replace('sensor_data_100hz_', '').replace('_channel', '')
            os.rename(os.path.join(base_path, file_name), os.path.join(base_path, new_name))
    print('Files have been renamed successfully.')

def copy_to_processed_data(selected_path):
    if not selected_path or not os.path.isdir(selected_path):
        print(f'Invalid selected_path: {selected_path}')
        return
    data_dir = os.path.dirname(os.path.dirname(selected_path))
    processed_root = os.path.join(data_dir, 'processed')
    leaf_name = os.path.basename(selected_path)
    dest_root = os.path.join(processed_root, leaf_name)
    subdirs = ['0. switch', '1. imu', '2. quaternion', '3. angle', '4. rula']
    for sub in subdirs:
        os.makedirs(os.path.join(dest_root, sub), exist_ok=True)
    sorted_dir = os.path.join(selected_path, 'sorted_result/each')
    src_switch = os.path.join(sorted_dir, 'switch.csv')
    dst_switch = os.path.join(dest_root, '0. switch', 'switch.csv')
    if os.path.exists(src_switch):
        shutil.copy2(src_switch, dst_switch)
        print(f'Copied: {src_switch} -> {dst_switch}')
    else:
        print(f'Warning: {src_switch} not found; skipping.')
    imu_dest_dir = os.path.join(dest_root, '1. imu')
    imu_candidates = [f for f in os.listdir(sorted_dir)] if os.path.isdir(sorted_dir) else []
    imu_files = [f for f in imu_candidates if f.lower().endswith('_imu.csv')]
    if not imu_files:
        print(f'No *_imu.csv files found in {sorted_dir}.')
    else:
        for fname in imu_files:
            src = os.path.join(sorted_dir, fname)
            dst = os.path.join(imu_dest_dir, fname)
            shutil.copy2(src, dst)
            print(f'Copied: {src} -> {dst}')
    print(f'Processed data copied to: {dest_root}')

def data_verification(selected_path):
    base_path = selected_path
    output_file = os.path.join(base_path, 'output.txt')
    summary_file = os.path.join(base_path, 'loading_time.txt')
    csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
    smallest_second_row_value = float('inf')
    largest_second_row_value = float('-inf')
    with open(output_file, 'w') as outfile:
        for csv_file in csv_files:
            try:
                with open(os.path.join(base_path, csv_file), 'r') as infile:
                    reader = list(csv.reader(infile))
                    if len(reader) < 2:
                        print(f'File {csv_file} does not have enough rows.')
                        continue
                    second_row_value = float(reader[1][-1])
                    if second_row_value < smallest_second_row_value:
                        smallest_second_row_value = second_row_value
                    if second_row_value > largest_second_row_value:
                        largest_second_row_value = second_row_value
                    last_row_value = float(reader[-1][-1])
                    difference = last_row_value - second_row_value
                    if difference != 0:
                        division_result = len(reader) / difference
                    else:
                        division_result = 'undefined'
                    outfile.write(f'{csv_file}: {second_row_value}, {last_row_value}, Difference = {difference}, Hz = {division_result}\n')
                    print(f'Processed {csv_file} successfully.')
            except Exception as e:
                print(f'Error processing {csv_file}: {e}')
    with open(summary_file, 'w') as summary:
        summary.write(f'{smallest_second_row_value}\n')
        summary.write(f'{largest_second_row_value}\n')
    print(f'All files have been processed. Output saved to {output_file}. Summary saved to {summary_file}.')

def remove_extra_clicks(selected_path):
    parent_dir = os.path.dirname(selected_path)
    grandparent_dir = os.path.dirname(parent_dir)
    lowest_dir = os.path.basename(selected_path)
    switch_dir = os.path.join(grandparent_dir, 'processed', lowest_dir, '0. switch')
    switch_file = os.path.join(switch_dir, 'switch.csv')
    df = pd.read_csv(switch_file)
    df['diff'] = df['switch'].diff()
    rising = df[df['diff'] > 0].index
    falling = df[df['diff'] < 0].index
    clicks = []
    for r in rising:
        f_candidates = falling[falling > r]
        if not f_candidates.empty:
            clicks.append((r, f_candidates[0]))
    for i, (start, end) in enumerate(clicks):
        if i >= 2:
            df.loc[start:end, 'switch'] = 0
    df.drop(columns=['diff'], inplace=True)
    df.to_csv(switch_file, index=False)
    print(f'OK Processed: {switch_file} | Clicks kept: 2 | Removed: {(len(clicks) - 2 if len(clicks) > 2 else 0)}')
