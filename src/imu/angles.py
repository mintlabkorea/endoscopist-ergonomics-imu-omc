"""Derive neck and upper-limb angles from relative segment orientations."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def w2a(w):
    theta_b = 2 * np.arccos(w)
    if theta_b > np.pi:
        theta = theta_b - 2 * np.pi
    elif theta_b <= np.pi:
        theta = theta_b
    degree = np.degrees(theta)
    return degree

def mean_vec(vectors):
    sum_vector = np.sum(vectors, axis=0)
    magnitude = np.linalg.norm(sum_vector)
    if magnitude != 0:
        mean_vector = sum_vector / magnitude
    else:
        mean_vector = np.zeros_like(sum_vector)
    return mean_vector

def qm(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w3, x3, y3, z3])

def qi(q):
    w, x, y, z = q
    q_conjugate = np.array([w, -x, -y, -z])
    norm_sq = w ** 2 + x ** 2 + y ** 2 + z ** 2
    q_inverse = q_conjugate / norm_sq
    return q_inverse

def q2a(quat):
    w, x, y, z = quat
    sin_theta_over_2 = np.sqrt(max(0.0, 1.0 - w * w))
    if abs(np.arcsin(sin_theta_over_2)) < np.pi / 90:
        return np.array([0, 0, 0])
    else:
        if w >= 0:
            v_x = x / sin_theta_over_2
            v_y = y / sin_theta_over_2
            v_z = z / sin_theta_over_2
        elif w < 0:
            v_x = -x / sin_theta_over_2
            v_y = -y / sin_theta_over_2
            v_z = -z / sin_theta_over_2
        return np.array([v_x, v_y, v_z])

def twist_removed_deg(q, second_axis):
    q = q / np.linalg.norm(q)
    s = q[1] * second_axis[0] + q[2] * second_axis[1] + q[3] * second_axis[2]
    twist = 2.0 * np.arctan2(s, q[0])
    q_f = np.array(
        [
            np.cos(twist / 2.0),
            np.sin(twist / 2.0) * second_axis[0],
            np.sin(twist / 2.0) * second_axis[1],
            np.sin(twist / 2.0) * second_axis[2]
        ]
    )
    q_s = qm(qi(q_f), q)
    q_s = q_s / np.linalg.norm(q_s)
    theta = 2.0 * np.arccos(np.clip(q_s[0], -1.0, 1.0))
    theta = np.minimum(theta, np.pi)
    return np.degrees(theta)

def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist_rate = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def raa(q, part):
    if part == 'neck':
        normal = np.array([0, 1, 0])
        second_axis = np.array([1, 0, 0])
        third_axis = np.array([0, 0, 1])
        s = q[1] * second_axis[0] + q[2] * second_axis[1] + q[3] * second_axis[2]
        twist = 2 * np.arctan2(s, q[0])
        q_f = np.array(
            [
                np.cos(twist / 2),
                np.sin(twist / 2) * second_axis[0],
                np.sin(twist / 2) * second_axis[1],
                np.sin(twist / 2) * second_axis[2]
            ]
        )
        q_s = qm(qi(q_f), q)
        q_s /= np.linalg.norm(q_s)
        f = q_s[1] * normal[0] + q_s[2] * normal[1] + q_s[3] * normal[2]
        flex = 2 * np.arctan2(f, q_s[0])
        q_t = np.array(
            [
                np.cos(flex / 2),
                np.sin(flex / 2) * normal[0],
                np.sin(flex / 2) * normal[1],
                np.sin(flex / 2) * normal[2]
            ]
        )
        q_fs = qm(qi(q_t), q_s)
        q_fs /= np.linalg.norm(q_fs)
        if q_fs[1] * third_axis[0] + q_fs[2] * third_axis[1] + q_fs[3] * third_axis[2] >= 0:
            bend = 2 * np.arccos(q_fs[0])
        elif q_fs[1] * third_axis[0] + q_fs[2] * third_axis[1] + q_fs[3] * third_axis[2] < 0:
            bend = -2 * np.arccos(q_fs[0])
        flex = np.degrees(flex)
        twist = np.degrees(twist)
        bend = np.degrees(bend)
        euler_angles = np.array([flex, twist, bend])
        return euler_angles
    elif part == 'shoulder':
        normal = np.array([0, 1, 0])
        second_axis = np.array([0, 0, 1])
        third_axis = np.array([1, 0, 0])
        s = q[1] * third_axis[0] + q[2] * third_axis[1] + q[3] * third_axis[2]
        twist = 2 * np.arctan2(s, q[0])
        q_f = np.array(
            [
                np.cos(twist / 2),
                np.sin(twist / 2) * third_axis[0],
                np.sin(twist / 2) * third_axis[1],
                np.sin(twist / 2) * third_axis[2]
            ]
        )
        q_s = qm(qi(q_f), q)
        q_s /= np.linalg.norm(q_s)
        f = q_s[1] * normal[0] + q_s[2] * normal[1] + q_s[3] * normal[2]
        flex = 2 * np.arctan2(f, q_s[0])
        q_t = np.array(
            [
                np.cos(flex / 2),
                np.sin(flex / 2) * normal[0],
                np.sin(flex / 2) * normal[1],
                np.sin(flex / 2) * normal[2]
            ]
        )
        q_fs = qm(qi(q_t), q_s)
        q_fs /= np.linalg.norm(q_fs)
        if q_fs[1] * second_axis[0] + q_fs[2] * second_axis[1] + q_fs[3] * second_axis[2] >= 0:
            bend = 2 * np.arccos(q_fs[0])
        elif q_fs[1] * second_axis[0] + q_fs[2] * second_axis[1] + q_fs[3] * second_axis[2] < 0:
            bend = -2 * np.arccos(q_fs[0])
        flex = np.degrees(flex)
        twist = np.degrees(twist)
        bend = np.degrees(bend)
        euler_angles = np.array([flex, bend, twist])
        return euler_angles
    elif part == 'elbow':
        normal = np.array([0, 0, 1])
        second_axis = np.array([1, 0, 0])
        third_axis = np.array([0, 1, 0])
        s = q[1] * second_axis[0] + q[2] * second_axis[1] + q[3] * second_axis[2]
        twist = 2 * np.arctan2(s, q[0])
        q_f = np.array(
            [
                np.cos(twist / 2),
                np.sin(twist / 2) * second_axis[0],
                np.sin(twist / 2) * second_axis[1],
                np.sin(twist / 2) * second_axis[2]
            ]
        )
        q_s = qm(qi(q_f), q)
        q_s /= np.linalg.norm(q_s)
        f = q_s[1] * normal[0] + q_s[2] * normal[1] + q_s[3] * normal[2]
        flex = 2 * np.arctan2(f, q_s[0])
        q_t = np.array(
            [
                np.cos(flex / 2),
                np.sin(flex / 2) * normal[0],
                np.sin(flex / 2) * normal[1],
                np.sin(flex / 2) * normal[2]
            ]
        )
        q_fs = qm(qi(q_t), q_s)
        q_fs /= np.linalg.norm(q_fs)
        if q_fs[1] * third_axis[0] + q_fs[2] * third_axis[1] + q_fs[3] * third_axis[2] >= 0:
            bend = 2 * np.arccos(q_fs[0])
        elif q_fs[1] * third_axis[0] + q_fs[2] * third_axis[1] + q_fs[3] * third_axis[2] < 0:
            bend = -2 * np.arccos(q_fs[0])
        flex = np.degrees(flex)
        twist = np.degrees(twist)
        bend = np.degrees(bend)
        euler_angles = np.array([flex, twist, bend])
        return euler_angles
    elif part == 'wrist':
        normal = np.array([0, 1, 0])
        second_axis = np.array([1, 0, 0])
        third_axis = np.array([0, 0, 1])
        s = q[1] * second_axis[0] + q[2] * second_axis[1] + q[3] * second_axis[2]
        twist = 2 * np.arctan2(s, q[0])
        q_f = np.array(
            [
                np.cos(twist / 2),
                np.sin(twist / 2) * second_axis[0],
                np.sin(twist / 2) * second_axis[1],
                np.sin(twist / 2) * second_axis[2]
            ]
        )
        q_s = qm(qi(q_f), q)
        q_s /= np.linalg.norm(q_s)
        f = q_s[1] * normal[0] + q_s[2] * normal[1] + q_s[3] * normal[2]
        flex = 2 * np.arctan2(f, q_s[0])
        q_t = np.array(
            [
                np.cos(flex / 2),
                np.sin(flex / 2) * normal[0],
                np.sin(flex / 2) * normal[1],
                np.sin(flex / 2) * normal[2]
            ]
        )
        q_fs = qm(qi(q_t), q_s)
        q_fs /= np.linalg.norm(q_fs)
        if q_fs[1] * third_axis[0] + q_fs[2] * third_axis[1] + q_fs[3] * third_axis[2] >= 0:
            bend = 2 * np.arccos(q_fs[0])
        elif q_fs[1] * third_axis[0] + q_fs[2] * third_axis[1] + q_fs[3] * third_axis[2] < 0:
            bend = -2 * np.arccos(q_fs[0])
        flex = np.degrees(flex)
        twist = np.degrees(twist)
        bend = np.degrees(bend)
        euler_angles = np.array([flex, bend, twist])
        return euler_angles

def calculate_angles(path):
    file_path = os.path.join(path, '2. quaternion', 'combined_q.csv')
    combined_data = pd.read_csv(file_path)
    output_dir = os.path.join(path, '3. angle')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'combined_angle.csv')
    quats = {}

    def joint_angle(start_idx, end_idx, offsets):
        for index in range(start_idx, end_idx):
            row = combined_data.iloc[index]
            for part in [
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
            ]:
                quat = row[[f'{part}_w', f'{part}_x', f'{part}_y', f'{part}_z']].values.reshape(4)
                quats[part] = quat
            inv_q_torso = qi(quats['torso_imu'])
            q_head = quats['head_imu']
            q_neck = qm(inv_q_torso, q_head)
            w1, w2, w3 = raa(q_neck, 'neck')
            neck_Flex[index - start_idx] = -w1 - offsets[0]
            neck_Twist[index - start_idx] = -w2 - offsets[1]
            neck_Bend[index - start_idx] = w3 - offsets[2]
            inv_q_torso = qi(quats['torso_imu'])
            q_arm_r = quats['right_humerus_imu']
            q_shoulder_r = qm(inv_q_torso, q_arm_r)
            q_arm_l = quats['left_humerus_imu']
            q_shoulder_l = qm(inv_q_torso, q_arm_l)
            w4, w5, w6 = raa(q_shoulder_r, 'shoulder')
            w7, w8, w9 = raa(q_shoulder_l, 'shoulder')
            r_shoulder_Flex[index - start_idx] = w4 - offsets[3]
            r_shoulder_Abd[index - start_idx] = w5 - offsets[4]
            r_shoulder_Twist[index - start_idx] = -w6 - offsets[5]
            l_shoulder_Flex[index - start_idx] = w7 - offsets[6]
            l_shoulder_Abd[index - start_idx] = -w8 - offsets[7]
            l_shoulder_Twist[index - start_idx] = w9 - offsets[8]
            inv_q_humerus_r = qi(quats['right_humerus_imu'])
            q_radius_r = quats['right_proximal_radius_imu']
            q_elbow_r = qm(inv_q_humerus_r, q_radius_r)
            inv_q_humerus_l = qi(quats['left_humerus_imu'])
            q_radius_l = quats['left_proximal_radius_imu']
            q_elbow_l = qm(inv_q_humerus_l, q_radius_l)
            second_axis = np.array([1.0, 0.0, 0.0])
            theta_r = twist_removed_deg(q_elbow_r, second_axis)
            theta_l = twist_removed_deg(q_elbow_l, second_axis)
            r_elbow_Flex[index - start_idx] = 180.0 - theta_r
            l_elbow_Flex[index - start_idx] = 180.0 - theta_l
            inv_q_radius_r = qi(quats['right_proximal_radius_imu'])
            q_sonmok_r = quats['right_distal_radius_imu']
            q_wrist_r = qm(inv_q_radius_r, q_sonmok_r)
            w22, w23, w24 = raa(q_wrist_r, 'wrist')
            r_wrist_Twist[index - start_idx] = -w24 - offsets[10]
            inv_q_radius_l = qi(quats['left_proximal_radius_imu'])
            q_sonmok_l = quats['left_distal_radius_imu']
            q_wrist_l = qm(inv_q_radius_l, q_sonmok_l)
            w25, w26, w27 = raa(q_wrist_l, 'wrist')
            l_wrist_Twist[index - start_idx] = -w27 - offsets[12]
            inv_q_sonmok_r = qi(quats['right_distal_radius_imu'])
            q_hand_r = quats['right_hand_imu']
            q_wrist_r = qm(inv_q_sonmok_r, q_hand_r)
            w16, w17, w18 = raa(q_wrist_r, 'wrist')
            r_wrist_Flex[index - start_idx] = w16 - offsets[13]
            r_wrist_Abd[index - start_idx] = w17 - offsets[14]
            inv_q_sonmok_l = qi(quats['left_distal_radius_imu'])
            q_hand_l = quats['left_hand_imu']
            q_wrist_l = qm(inv_q_sonmok_l, q_hand_l)
            w19, w20, w21 = raa(q_wrist_l, 'wrist')
            l_wrist_Flex[index - start_idx] = w19 - offsets[15]
            l_wrist_Abd[index - start_idx] = -w20 - offsets[16]

    def static_cal(start_idx, end_idx):
        offsets = np.zeros((17, end_idx - start_idx + 1))
        for index in range(start_idx, end_idx):
            row = combined_data.iloc[index]
            for part in [
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
            ]:
                quat = row[[f'{part}_w', f'{part}_x', f'{part}_y', f'{part}_z']].values.reshape(4)
                quats[part] = quat
            inv_q_torso = qi(quats['torso_imu'])
            q_head = quats['head_imu']
            q_neck = qm(inv_q_torso, q_head)
            w1, w2, w3 = raa(q_neck, 'neck')
            offsets[0, index - start_idx] = -w1
            offsets[1, index - start_idx] = -w2
            offsets[2, index - start_idx] = w3
            inv_q_torso = qi(quats['torso_imu'])
            q_arm_r = quats['right_humerus_imu']
            q_shoulder_r = qm(inv_q_torso, q_arm_r)
            q_arm_l = quats['left_humerus_imu']
            q_shoulder_l = qm(inv_q_torso, q_arm_l)
            w4, w5, w6 = raa(q_shoulder_r, 'shoulder')
            w7, w8, w9 = raa(q_shoulder_l, 'shoulder')
            offsets[3, index - start_idx] = w4
            offsets[4, index - start_idx] = w5
            offsets[5, index - start_idx] = -w6
            offsets[6, index - start_idx] = w7
            offsets[7, index - start_idx] = -w8
            offsets[8, index - start_idx] = w9
            inv_q_humerus_r = qi(quats['right_humerus_imu'])
            q_radius_r = quats['right_proximal_radius_imu']
            q_elbow_r = qm(inv_q_humerus_r, q_radius_r)
            inv_q_humerus_l = qi(quats['left_humerus_imu'])
            q_radius_l = quats['left_proximal_radius_imu']
            q_elbow_l = qm(inv_q_humerus_l, q_radius_l)
            w10, w11, w12 = raa(q_elbow_r, 'elbow')
            w13, w14, w15 = raa(q_elbow_l, 'elbow')
            offsets[9, index - start_idx] = w10
            offsets[11, index - start_idx] = w13
            inv_q_radius_r = qi(quats['right_proximal_radius_imu'])
            q_sonmok_r = quats['right_distal_radius_imu']
            q_wrist_r = qm(inv_q_radius_r, q_sonmok_r)
            w22, w23, w24 = raa(q_wrist_r, 'wrist')
            offsets[10, index - start_idx] = -w24
            inv_q_radius_l = qi(quats['left_proximal_radius_imu'])
            q_sonmok_l = quats['left_distal_radius_imu']
            q_wrist_l = qm(inv_q_radius_l, q_sonmok_l)
            w25, w26, w27 = raa(q_wrist_l, 'wrist')
            offsets[12, index - start_idx] = -w27
            inv_q_radius_r = qi(quats['right_distal_radius_imu'])
            q_hand_r = quats['right_hand_imu']
            q_wrist_r = qm(inv_q_radius_r, q_hand_r)
            inv_q_radius_l = qi(quats['left_distal_radius_imu'])
            q_hand_l = quats['left_hand_imu']
            q_wrist_l = qm(inv_q_radius_l, q_hand_l)
            w16, w17, w18 = raa(q_wrist_r, 'wrist')
            w19, w20, w21 = raa(q_wrist_l, 'wrist')
            offsets[13, index - start_idx] = w16
            offsets[14, index - start_idx] = w17
            offsets[15, index - start_idx] = w19
            offsets[16, index - start_idx] = -w20
        offsets = np.mean(offsets, axis=1)
        return offsets
    timestamp = combined_data['time'].values
    switch = combined_data['switch'].values
    t_len = len(timestamp)
    df = pd.DataFrame(switch, columns=['value'])
    df['block'] = (df['value'].diff() != 0).cumsum()
    tmp = df.loc[df['value'] == 1].assign(i=lambda x: x.index)
    blocks = tmp.groupby('block', sort=False)['i'].agg(start_index='min', end_index='max').reset_index().rename(columns={'block': 'block_index'})
    blocks['block_index'] = range(1, len(blocks) + 1)
    block_list = [
        {
            'block_index': row['block_index'],
            'start_index': row['start_index'],
            'end_index': row['end_index'] + 1
        } for _,
        row in blocks.iterrows()
    ]
    for block in block_list:
        block_index = block['block_index']
        start_index = block['start_index']
        end_index = block['end_index']
        if block_index == 1:
            offsets = static_cal(start_index, end_index)
        else:
            print('No Calibration')
    neck_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    neck_Twist = np.empty((t_len - block_list[0]['end_index'], 1))
    neck_Bend = np.empty((t_len - block_list[0]['end_index'], 1))
    r_shoulder_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    l_shoulder_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    r_shoulder_Abd = np.empty((t_len - block_list[0]['end_index'], 1))
    l_shoulder_Abd = np.empty((t_len - block_list[0]['end_index'], 1))
    r_shoulder_Twist = np.empty((t_len - block_list[0]['end_index'], 1))
    l_shoulder_Twist = np.empty((t_len - block_list[0]['end_index'], 1))
    r_elbow_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    l_elbow_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    r_wrist_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    l_wrist_Flex = np.empty((t_len - block_list[0]['end_index'], 1))
    r_wrist_Abd = np.empty((t_len - block_list[0]['end_index'], 1))
    l_wrist_Abd = np.empty((t_len - block_list[0]['end_index'], 1))
    r_wrist_Twist = np.empty((t_len - block_list[0]['end_index'], 1))
    l_wrist_Twist = np.empty((t_len - block_list[0]['end_index'], 1))
    joint_angle(block_list[0]['end_index'], t_len, offsets)
    joint_dict = {
        'time': timestamp[block_list[0]['end_index']:t_len],
        'neck_Flex': neck_Flex.flatten(),
        'neck_Twist': neck_Twist.flatten(),
        'neck_Bend': neck_Bend.flatten(),
        'r_shoulder_Flex': r_shoulder_Flex.flatten(),
        'r_shoulder_Abd': r_shoulder_Abd.flatten(),
        'r_shoulder_Twist': r_shoulder_Twist.flatten(),
        'l_shoulder_Flex': l_shoulder_Flex.flatten(),
        'l_shoulder_Abd': l_shoulder_Abd.flatten(),
        'l_shoulder_Twist': l_shoulder_Twist.flatten(),
        'r_elbow_Flex': r_elbow_Flex.flatten(),
        'l_elbow_Flex': l_elbow_Flex.flatten(),
        'r_wrist_Flex': r_wrist_Flex.flatten(),
        'r_wrist_Abd': r_wrist_Abd.flatten(),
        'r_wrist_Twist': r_wrist_Twist.flatten(),
        'l_wrist_Flex': l_wrist_Flex.flatten(),
        'l_wrist_Abd': l_wrist_Abd.flatten(),
        'l_wrist_Twist': l_wrist_Twist.flatten()
    }
    df = pd.DataFrame(joint_dict)
    sample_rate = 100.0
    cutoff_freq = 1.0
    time_column = df.columns[0]
    data_columns = df.columns[1:]
    for col in data_columns:
        df[col] = lowpass_filter(df[col].values, cutoff_freq, sample_rate)
    df.to_csv(output_file_path, index=False)
    print(f'Data has been saved to {output_file_path}')

def sort_angles(path):
    input_file = os.path.join(path, '3. angle/combined_angle.csv')
    output_dir = os.path.join(path, '3. angle/each')
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(input_file)
    first_column_header = data.columns[0]
    for col in data.columns[1:]:
        new_df = data[[first_column_header, col]]
        output_file = os.path.join(output_dir, f'{col}.csv')
        new_df.to_csv(output_file, index=False)
    print('CSV files have been saved successfully.')

def angle_figure(path):
    input_path = os.path.join(path, '3. angle', 'each')
    output_dir = os.path.join(path, '3. angle', 'figure')
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
        plt.ylabel('Angle (degrees)')
        plt.title(f'Plot of {csv_file}')
        plt.legend()
        plt.grid(True)
        output_path = os.path.join(output_dir, f'{os.path.splitext(csv_file)[0]}.png')
        plt.savefig(output_path)
        plt.close()
        print(f'Saved plot for {csv_file}')
    print('Plots generated and saved successfully!')
