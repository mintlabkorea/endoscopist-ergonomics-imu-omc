"""Convert joint angles into continuous RULA scores and recording reports."""

import os
import re
import datetime
import pandas as pd
import numpy as np
from . import rula
import matplotlib.pyplot as plt

def rula_convert(path):
    input_dir = angle_input(path)
    output_dir = rula_output(path)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'rula_conversion.csv')
    input_files = [
        'neck_Flex.csv',
        'neck_Twist.csv',
        'neck_Bend.csv',
        'r_shoulder_Flex.csv',
        'r_shoulder_Abd.csv',
        'r_shoulder_Twist.csv',
        'l_shoulder_Flex.csv',
        'l_shoulder_Abd.csv',
        'l_shoulder_Twist.csv',
        'r_elbow_Flex.csv',
        'l_elbow_Flex.csv',
        'r_wrist_Flex.csv',
        'r_wrist_Abd.csv',
        'r_wrist_Twist.csv',
        'l_wrist_Flex.csv',
        'l_wrist_Abd.csv',
        'l_wrist_Twist.csv'
    ]
    for file in input_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            print(f'The file {file} does not exist.')
    updated_files = []
    for file in input_files:
        base_name, ext = os.path.splitext(file)
        fixed_file = f'{base_name}_fixed{ext}'
        fixed_path = os.path.join(input_dir, fixed_file)
        if os.path.exists(fixed_path):
            updated_files.append(fixed_file)
        else:
            updated_files.append(file)
    input_files = updated_files
    dfs = []
    for file in input_files:
        file_path = os.path.join(input_dir, file)
        if not os.path.exists(file_path):
            continue
        base_name = os.path.splitext(file)[0]
        df_temp = pd.read_csv(file_path)
        df_temp.columns = ['timestamps', base_name]
        dfs.append(df_temp)
    if dfs:
        df = dfs[0]
        for df_temp in dfs[1:]:
            df = pd.merge(df, df_temp, on='timestamps', how='outer')
        df.sort_values(by='timestamps', inplace=True)
    else:
        df = pd.DataFrame()
    df.columns = df.columns.str.replace('_fixed$', '', regex=True)
    upper_arm_position_left = []
    is_upper_arm_abducted_left = []
    lower_arm_position_left = []
    wrist_position_left = []
    is_wrist_bent_left = []
    upper_arm_position_right = []
    is_upper_arm_abducted_right = []
    lower_arm_position_right = []
    wrist_position_right = []
    is_wrist_bent_right = []
    is_shoulder_raised = []
    is_upper_arm_supported = []
    is_lower_arm_crossed_or_outside = []
    is_wrist_twisted_mid = []
    is_wrist_twisted_max = []
    is_muscle_static_A = []
    is_muscle_repeated_A = []
    load_value_A = []
    is_load_constant_A = []
    neck_position = []
    is_neck_twisted = []
    is_neck_bent = []
    trunk_position = []
    is_trunk_twisted = []
    is_trunk_bent = []
    is_leg_supported = []
    is_muscle_static_B = []
    is_muscle_repeated_B = []
    load_value_B = []
    is_load_constant_B = []
    for index, row in df.iterrows():
        upper_arm_position_left.append(row['l_shoulder_Flex'])
        upper_arm_position_right.append(row['r_shoulder_Flex'])
        is_shoulder_raised.append(False)
        is_upper_arm_abducted_left.append(row['l_shoulder_Abd'] > 20)
        is_upper_arm_abducted_right.append(row['r_shoulder_Abd'] > 20)
        is_upper_arm_supported.append(False)
        lower_arm_position_left.append(180 - row['l_elbow_Flex'])
        lower_arm_position_right.append(180 - row['r_elbow_Flex'])
        is_lower_arm_crossed_or_outside.append(False)
        wrist_position_left.append(abs(row['l_wrist_Flex']))
        wrist_position_right.append(abs(row['r_wrist_Flex']))
        is_wrist_bent_left.append(abs(row['l_wrist_Abd']) > 30)
        is_wrist_bent_right.append(abs(row['r_wrist_Abd']) > 30)
        is_wrist_twisted_mid.append(True)
        is_wrist_twisted_max.append(False)
        neck_position.append(row['neck_Flex'])
        is_neck_twisted.append(False)
        is_neck_bent.append(abs(row['neck_Bend']) > 30)
        is_muscle_static_A.append(True)
        is_muscle_repeated_A.append(True)
        load_value_A.append(1)
        is_load_constant_A.append(False)
        trunk_position.append(5)
        is_trunk_twisted.append(False)
        is_trunk_bent.append(False)
        is_leg_supported.append(True)
        is_muscle_static_B.append(False)
        is_muscle_repeated_B.append(False)
        load_value_B.append(0)
        is_load_constant_B.append(False)
    rula_score_df = pd.DataFrame(
        {
            'time': df['timestamps'],
            'upper_arm_position_left': upper_arm_position_left,
            'upper_arm_position_right': upper_arm_position_right,
            'is_shoulder_raised': is_shoulder_raised,
            'is_upper_arm_abducted_left': is_upper_arm_abducted_left,
            'is_upper_arm_abducted_right': is_upper_arm_abducted_right,
            'is_upper_arm_supported': is_upper_arm_supported,
            'lower_arm_position_left': lower_arm_position_left,
            'lower_arm_position_right': lower_arm_position_right,
            'is_lower_arm_crossed_or_outside': is_lower_arm_crossed_or_outside,
            'wrist_position_left': wrist_position_left,
            'wrist_position_right': wrist_position_right,
            'is_wrist_bent_left': is_wrist_bent_left,
            'is_wrist_bent_right': is_wrist_bent_right,
            'is_wrist_twisted_mid': is_wrist_twisted_mid,
            'is_wrist_twisted_max': is_wrist_twisted_max,
            'is_muscle_static_A': is_muscle_static_A,
            'is_muscle_repeated_A': is_muscle_repeated_A,
            'load_value_A': load_value_A,
            'is_load_constant_A': is_load_constant_A,
            'neck_position': neck_position,
            'is_neck_twisted': is_neck_twisted,
            'is_neck_bent': is_neck_bent,
            'trunk_position': trunk_position,
            'is_trunk_twisted': is_trunk_twisted,
            'is_trunk_bent': is_trunk_bent,
            'is_leg_supported': is_leg_supported,
            'is_muscle_static_B': is_muscle_static_B,
            'is_muscle_repeated_B': is_muscle_repeated_B,
            'load_value_B': load_value_B,
            'is_load_constant_B': is_load_constant_B
        }
    )
    rula_score_df.to_csv(output_file, index=False)
    print(f'RULA score data has been written to {output_file}')

def rula_score(path):
    input_dir = rula_output(path)
    input_file = os.path.join(input_dir, 'rula_conversion.csv')
    output_dir = rula_output(path)
    output_file = os.path.join(output_dir, 'rula_score.csv')
    if not os.path.exists(input_file):
        print(f'  rula_conversion.csv does not exist.')
    else:
        df = pd.read_csv(input_file)
        results = []
        for index, row in df.iterrows():
            try:
                upper_arm_position_left = row['upper_arm_position_left']
                upper_arm_position_right = row['upper_arm_position_right']
                is_shoulder_raised = row['is_shoulder_raised']
                is_upper_arm_abducted_left = row['is_upper_arm_abducted_left']
                is_upper_arm_abducted_right = row['is_upper_arm_abducted_right']
                is_upper_arm_supported = row['is_upper_arm_supported']
                lower_arm_position_left = row['lower_arm_position_left']
                lower_arm_position_right = row['lower_arm_position_right']
                is_lower_arm_crossed_or_outside = row['is_lower_arm_crossed_or_outside']
                wrist_position_left = row['wrist_position_left']
                wrist_position_right = row['wrist_position_right']
                is_wrist_bent_left = row['is_wrist_bent_left']
                is_wrist_bent_right = row['is_wrist_bent_right']
                is_wrist_twisted_mid = row['is_wrist_twisted_mid']
                is_wrist_twisted_max = row['is_wrist_twisted_max']
                is_muscle_static_A = row['is_muscle_static_A']
                is_muscle_repeated_A = row['is_muscle_repeated_A']
                load_value_A = row['load_value_A']
                is_load_constant_A = row['is_load_constant_A']
                neck_position = row['neck_position']
                is_neck_twisted = row['is_neck_twisted']
                is_neck_bent = row['is_neck_bent']
                trunk_position = row['trunk_position']
                is_trunk_twisted = row['is_trunk_twisted']
                is_trunk_bent = row['is_trunk_bent']
                is_leg_supported = row['is_leg_supported']
                is_muscle_static_B = row['is_muscle_static_B']
                is_muscle_repeated_B = row['is_muscle_repeated_B']
                load_value_B = row['load_value_B']
                is_load_constant_B = row['is_load_constant_B']
                upper_arm_score_left = rula.upper_arm_score(
                    upper_arm_position_left,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_supported
                )
                upper_arm_score_right = rula.upper_arm_score(
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported
                )
                upper_arm_score = np.nanmax([upper_arm_score_left, upper_arm_score_right])
                lower_arm_score_left = rula.lower_arm_score(lower_arm_position_left, is_lower_arm_crossed_or_outside)
                lower_arm_score_right = rula.lower_arm_score(lower_arm_position_right, is_lower_arm_crossed_or_outside)
                lower_arm_score = np.nanmax([lower_arm_score_left, lower_arm_score_right])
                wrist_score_left = rula.wrist_score(wrist_position_left, is_wrist_bent_left)
                wrist_score_right = rula.wrist_score(wrist_position_right, is_wrist_bent_right)
                wrist_score = np.nanmax([wrist_score_left, wrist_score_right])
                wrist_twist_score = rula.wrist_twist_score(is_wrist_twisted_mid, is_wrist_twisted_max)
                posture_score_A_left = rula.posture_score_A(
                    upper_arm_position_left,
                    upper_arm_position_left,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_abducted_left,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_left,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_left,
                    wrist_position_left,
                    is_wrist_bent_left,
                    is_wrist_bent_left,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max
                )
                posture_score_A_right = rula.posture_score_A(
                    upper_arm_position_right,
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_right,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported,
                    lower_arm_position_right,
                    lower_arm_position_right,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_right,
                    wrist_position_right,
                    is_wrist_bent_right,
                    is_wrist_bent_right,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max
                )
                posture_score_A = rula.posture_score_A(
                    upper_arm_position_left,
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_right,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_left,
                    wrist_position_right,
                    is_wrist_bent_left,
                    is_wrist_bent_right,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max
                )
                muscle_use_score_A = rula.muscle_use_score_A(is_muscle_static_A, is_muscle_repeated_A)
                force_load_score_A = rula.force_load_score_A(load_value_A, is_load_constant_A)
                wrist_and_arm_score_left = rula.wrist_and_arm_score(
                    upper_arm_position_left,
                    upper_arm_position_left,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_abducted_left,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_left,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_left,
                    wrist_position_left,
                    is_wrist_bent_left,
                    is_wrist_bent_left,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max,
                    is_muscle_static_A,
                    is_muscle_repeated_A,
                    load_value_A,
                    is_load_constant_A
                )
                wrist_and_arm_score_right = rula.wrist_and_arm_score(
                    upper_arm_position_right,
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_right,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_right,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_right,
                    wrist_position_right,
                    is_wrist_bent_right,
                    is_wrist_bent_right,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max,
                    is_muscle_static_A,
                    is_muscle_repeated_A,
                    load_value_A,
                    is_load_constant_A
                )
                wrist_and_arm_score = rula.wrist_and_arm_score(
                    upper_arm_position_left,
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_right,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_left,
                    wrist_position_right,
                    is_wrist_bent_left,
                    is_wrist_bent_right,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max,
                    is_muscle_static_A,
                    is_muscle_repeated_A,
                    load_value_A,
                    is_load_constant_A
                )
                neck_score = rula.neck_score(neck_position, is_neck_twisted, is_neck_bent)
                trunk_score = rula.trunk_score(trunk_position, is_trunk_twisted, is_trunk_bent)
                leg_score = rula.leg_score(is_leg_supported)
                posture_score_B = rula.posture_score_B(
                    neck_position,
                    is_neck_twisted,
                    is_neck_bent,
                    trunk_position,
                    is_trunk_twisted,
                    is_trunk_bent,
                    is_leg_supported
                )
                muscle_use_score_B = rula.muscle_use_score_B(is_muscle_static_B, is_muscle_repeated_B)
                force_load_score_B = rula.force_load_score_B(load_value_B, is_load_constant_B)
                neck_trunk_leg_score = rula.neck_trunk_leg_score(
                    neck_position,
                    is_neck_twisted,
                    is_neck_bent,
                    trunk_position,
                    is_trunk_twisted,
                    is_trunk_bent,
                    is_leg_supported,
                    is_muscle_static_B,
                    is_muscle_repeated_B,
                    load_value_B,
                    is_load_constant_B
                )
                rula_score_left = rula.rula_score(
                    upper_arm_position_left,
                    upper_arm_position_left,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_abducted_left,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_left,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_left,
                    wrist_position_left,
                    is_wrist_bent_left,
                    is_wrist_bent_left,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max,
                    is_muscle_static_A,
                    is_muscle_repeated_A,
                    load_value_A,
                    is_load_constant_A,
                    neck_position,
                    is_neck_twisted,
                    is_neck_bent,
                    trunk_position,
                    is_trunk_twisted,
                    is_trunk_bent,
                    is_leg_supported,
                    is_muscle_static_B,
                    is_muscle_repeated_B,
                    load_value_B,
                    is_load_constant_B
                )
                rula_score_right = rula.rula_score(
                    upper_arm_position_right,
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_right,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported,
                    lower_arm_position_right,
                    lower_arm_position_right,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_right,
                    wrist_position_right,
                    is_wrist_bent_right,
                    is_wrist_bent_right,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max,
                    is_muscle_static_A,
                    is_muscle_repeated_A,
                    load_value_A,
                    is_load_constant_A,
                    neck_position,
                    is_neck_twisted,
                    is_neck_bent,
                    trunk_position,
                    is_trunk_twisted,
                    is_trunk_bent,
                    is_leg_supported,
                    is_muscle_static_B,
                    is_muscle_repeated_B,
                    load_value_B,
                    is_load_constant_B
                )
                rula_score = rula.rula_score(
                    upper_arm_position_left,
                    upper_arm_position_right,
                    is_shoulder_raised,
                    is_upper_arm_abducted_left,
                    is_upper_arm_abducted_right,
                    is_upper_arm_supported,
                    lower_arm_position_left,
                    lower_arm_position_right,
                    is_lower_arm_crossed_or_outside,
                    wrist_position_left,
                    wrist_position_right,
                    is_wrist_bent_left,
                    is_wrist_bent_right,
                    is_wrist_twisted_mid,
                    is_wrist_twisted_max,
                    is_muscle_static_A,
                    is_muscle_repeated_A,
                    load_value_A,
                    is_load_constant_A,
                    neck_position,
                    is_neck_twisted,
                    is_neck_bent,
                    trunk_position,
                    is_trunk_twisted,
                    is_trunk_bent,
                    is_leg_supported,
                    is_muscle_static_B,
                    is_muscle_repeated_B,
                    load_value_B,
                    is_load_constant_B
                )
                results.append(
                    [
                        row['time'],
                        upper_arm_score_left,
                        upper_arm_score_right,
                        upper_arm_score,
                        lower_arm_score_left,
                        lower_arm_score_right,
                        lower_arm_score,
                        wrist_score_left,
                        wrist_score_right,
                        wrist_score,
                        wrist_twist_score,
                        posture_score_A_left,
                        posture_score_A_right,
                        posture_score_A,
                        muscle_use_score_A,
                        force_load_score_A,
                        wrist_and_arm_score_left,
                        wrist_and_arm_score_right,
                        wrist_and_arm_score,
                        neck_score,
                        trunk_score,
                        leg_score,
                        posture_score_B,
                        muscle_use_score_B,
                        force_load_score_B,
                        neck_trunk_leg_score,
                        rula_score_left,
                        rula_score_right,
                        rula_score
                    ]
                )
            except Exception as e:
                print(f'Error processing row {index}: {e}')
        scores_df = pd.DataFrame(
            results,
            columns=[
                'time',
                'upper_arm_score_left',
                'upper_arm_score_right',
                'upper_arm_score',
                'lower_arm_score_left',
                'lower_arm_score_right',
                'lower_arm_score',
                'wrist_score_left',
                'wrist_score_right',
                'wrist_score',
                'wrist_twist_score',
                'posture_score_A_left',
                'posture_score_A_right',
                'posture_score_A',
                'muscle_use_score_A',
                'force_load_score_A',
                'wrist_and_arm_score_left',
                'wrist_and_arm_score_right',
                'wrist_and_arm_score',
                'neck_score',
                'trunk_score',
                'leg_score',
                'posture_score_B',
                'muscle_use_score_B',
                'force_load_score_B',
                'neck_trunk_leg_score',
                'rula_score_left',
                'rula_score_right',
                'rula_score'
            ]
        )
        scores_df.to_csv(output_file, index=False)
        print(f'The RULA scores have been saved to {output_file}.')

def rula_plot(path):
    input_dir = rula_output(path)
    input_file = os.path.join(input_dir, 'rula_score.csv')
    output_dir = os.path.join(input_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(input_file)
    time_column = data.columns[0]
    for i in range(1, len(data.columns)):
        header = data.columns[i]
        time = data[time_column]
        score = data[header]
        plt.figure(figsize=(20, 10))
        plt.plot(time, score, linestyle='-', label=header)
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.title(f'{header}')
        plt.grid(True)
        plt.legend()
        output_file = os.path.join(output_dir, f'{header}.png')
        plt.savefig(output_file)
        plt.close()
    print(f'All plots saved to {output_dir}')

def _process_error_block(block_lines, sink_set):
    import re
    failure_time = None
    affected = []
    in_affected = False
    for ln in block_lines:
        if 'Failure time' in ln:
            m = re.search(r'Failure time\s*:\s*([-+]?[0-9.]+)', ln)
            if m:
                failure_time = m.group(1).strip()
        if 'Affected angles' in ln:
            in_affected = True
            continue
        if in_affected:
            m = re.search('^\\s*-\\s*([A-Za-z0-9_]+)\\.csv', ln)
            if m:
                affected.append(m.group(1))
            if ln.strip().startswith('='):
                break
    if failure_time is None:
        failure_time = 'N/A'
    for seg in affected:
        sink_set.add((seg, failure_time))

def _is_float(s):
    try:
        float(s)
        return True
    except:
        return False

def rula_analysis(path):
    input_dir = rula_output(path)
    input_file = os.path.join(input_dir, 'rula_score.csv')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'rula_score.csv not found: {input_file}')
    output_dir = report_output(path)
    output_file = os.path.join(output_dir, 'analysis.txt')
    df = pd.read_csv(input_file)
    if df.empty:
        raise ValueError(f'rula_score.csv is empty: {input_file}')
    leaf_name = os.path.basename(os.path.normpath(path))

    def _mean_for(colname):
        if colname is None:
            return None
        vals = pd.to_numeric(df[colname], errors='coerce')
        if vals.notna().any():
            return float(vals.mean(skipna=True))
        return None

    def _ratio_for(colname, low=1, high=7):
        if colname is None:
            return 'N/A'
        vals = pd.to_numeric(df[colname], errors='coerce').dropna()
        if vals.empty:
            return 'N/A'
        ivals = vals.round().astype(int)
        ivals = ivals[(ivals >= low) & (ivals <= high)]
        total = len(ivals)
        if total == 0:
            return 'N/A'
        parts = []
        for k in range(low, high + 1):
            cnt = int((ivals == k).sum())
            ratio = cnt / total
            parts.append(f'{k},{ratio:.3f}')
        return ';'.join(parts)

    def _fmt_mean(v):
        return f'{v:.3f}' if isinstance(v, (int, float)) and v is not None else 'N/A'
    time_col = df.columns[0]
    time_series = pd.to_numeric(df[time_col], errors='coerce')
    time_series = time_series.dropna()
    if time_series.empty:
        duration = None
    else:
        duration = float(time_series.iloc[-1] - time_series.iloc[0])
    err_log_path = os.path.join(path, 'error_log.txt')
    error_items = set()
    if os.path.exists(err_log_path):
        with open(err_log_path, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f]
        block = []
        in_block = False
        for ln in lines:
            if ln.strip().startswith('=') and 'IMU ERROR' in ln:
                if in_block and block:
                    _process_error_block(block, error_items)
                block = [ln]
                in_block = True
            elif in_block:
                block.append(ln)
        if in_block and block:
            _process_error_block(block, error_items)
    avg_upper = _mean_for('upper_arm_score')
    avg_upper_L = _mean_for('upper_arm_score_left')
    avg_upper_R = _mean_for('upper_arm_score_right')
    avg_lower = _mean_for('lower_arm_score')
    avg_lower_L = _mean_for('lower_arm_score_left')
    avg_lower_R = _mean_for('lower_arm_score_right')
    avg_wrist = _mean_for('wrist_score')
    avg_wrist_L = _mean_for('wrist_score_left')
    avg_wrist_R = _mean_for('wrist_score_right')
    avg_wna = _mean_for('wrist_and_arm_score')
    avg_wna_L = _mean_for('wrist_and_arm_score_left')
    avg_wna_R = _mean_for('wrist_and_arm_score_right')
    avg_neck = _mean_for('neck_score')
    avg_rula = _mean_for('rula_score')
    ratio_upper = _ratio_for('upper_arm_score')
    ratio_upper_L = _ratio_for('upper_arm_score_left')
    ratio_upper_R = _ratio_for('upper_arm_score_right')
    ratio_lower = _ratio_for('lower_arm_score')
    ratio_lower_L = _ratio_for('lower_arm_score_left')
    ratio_lower_R = _ratio_for('lower_arm_score_right')
    ratio_wrist = _ratio_for('wrist_score')
    ratio_wrist_L = _ratio_for('wrist_score_left')
    ratio_wrist_R = _ratio_for('wrist_score_right')
    ratio_wna = _ratio_for('wrist_and_arm_score')
    ratio_wna_L = _ratio_for('wrist_and_arm_score_left')
    ratio_wna_R = _ratio_for('wrist_and_arm_score_right')
    ratio_neck = _ratio_for('neck_score')
    ratio_rula = _ratio_for('rula_score')
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f'[ RULA Score Analysis of {leaf_name} ]\n')
        f.write('\n')
        f.write(' ※ General Information ※\n')
        f.write(f'  > Logged at {now_str}\n')
        if duration is not None:
            f.write(f'  > Duration : {duration}\n')
        else:
            f.write(f'  > Duration : N/A (invalid or missing time values)\n')
        if os.path.exists(err_log_path) and error_items:
            f.write(f'  > Error Segments :\n')
            for seg, tval in sorted(
                list(error_items),
                key=lambda x: (float(x[1]) if _is_float(x[1]) else 1e+30, x[0])
            ):
                if _is_float(tval) and (not time_series.empty):
                    try:
                        t0 = float(time_series.iloc[0])
                        t_error = float(tval) - t0
                        f.write(f'        - {seg} from {t_error:.2f}\n')
                    except Exception:
                        f.write(f'        - {seg} from {tval}\n')
                else:
                    f.write(f'        - {seg} from {tval}\n')
        elif os.path.exists(err_log_path):
            f.write(f'  > Error Segments : (present but none parsed)\n')
        else:
            f.write(f'  > Error Segments : None (no error_log.txt)\n')
        f.write('\n ※ Average Score ※\n')
        f.write(f'  > Upper Arm : {_fmt_mean(avg_upper)}\n')
        f.write(f'       - (L) : {_fmt_mean(avg_upper_L)}\n')
        f.write(f'       - (R) : {_fmt_mean(avg_upper_R)}\n')
        f.write(f'  > Lower Arm : {_fmt_mean(avg_lower)}\n')
        f.write(f'       - (L) : {_fmt_mean(avg_lower_L)}\n')
        f.write(f'       - (R) : {_fmt_mean(avg_lower_R)}\n')
        f.write(f'  > Wrist : {_fmt_mean(avg_wrist)}\n')
        f.write(f'       - (L) : {_fmt_mean(avg_wrist_L)}\n')
        f.write(f'       - (R) : {_fmt_mean(avg_wrist_R)}\n')
        f.write(f'  > Wrist & Arm : {_fmt_mean(avg_wna)}\n')
        f.write(f'       - (L) : {_fmt_mean(avg_wna_L)}\n')
        f.write(f'       - (R) : {_fmt_mean(avg_wna_R)}\n')
        f.write(f'  > Neck : {_fmt_mean(avg_neck)}\n')
        f.write(f'  > RULA : {_fmt_mean(avg_rula)}\n')
        f.write('\n ※ Score Ratio ※\n')
        f.write(f'  > Upper Arm : {ratio_upper}\n')
        f.write(f'       - (L) : {ratio_upper_L}\n')
        f.write(f'       - (R) : {ratio_upper_R}\n')
        f.write(f'  > Lower Arm : {ratio_lower}\n')
        f.write(f'       - (L) : {ratio_lower_L}\n')
        f.write(f'       - (R) : {ratio_lower_R}\n')
        f.write(f'  > Wrist : {ratio_wrist}\n')
        f.write(f'       - (L) : {ratio_wrist_L}\n')
        f.write(f'       - (R) : {ratio_wrist_R}\n')
        f.write(f'  > Wrist & Arm : {ratio_wna}\n')
        f.write(f'       - (L) : {ratio_wna_L}\n')
        f.write(f'       - (R) : {ratio_wna_R}\n')
        f.write(f'  > Neck : {ratio_neck}\n')
        f.write(f'  > RULA : {ratio_rula}\n')
    print(f'OK Wrote analysis to {output_file}')


def angle_input(path):
    from pathlib import Path
    root = Path(path)
    return str(root / "3. angle" / "each" if (root / "3. angle").is_dir() else root / "each")

def rula_output(path):
    from pathlib import Path
    root = Path(path)
    return str(root / "4. rula" / "log" if (root / "3. angle").is_dir() else root / "rula")

def report_output(path):
    from pathlib import Path
    root = Path(path)
    return str(root / "4. rula" if (root / "3. angle").is_dir() else root / "rula")
