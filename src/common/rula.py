"""RULA component scores and posture lookup tables."""

import numpy as np
import pandas as pd

def _any_nan(*args):
    return any((pd.isna(a) for a in args))

def upper_arm_score(upper_arm_position, is_shoulder_raised, is_upper_arm_abducted, is_upper_arm_supported):
    if _any_nan(upper_arm_position):
        return np.nan
    score = 0
    if -20 <= upper_arm_position <= 20:
        score += 1
    elif -20 > upper_arm_position:
        score += 2
    elif 20 < upper_arm_position <= 45:
        score += 2
    elif 45 < upper_arm_position <= 90:
        score += 3
    elif 90 < upper_arm_position:
        score += 4
    if is_shoulder_raised:
        score += 1
    if is_upper_arm_abducted:
        score += 1
    if is_upper_arm_supported:
        score -= 1
    return max(1, min(score, 6))

def lower_arm_score(lower_arm_position, is_lower_arm_crossed_or_outside):
    if _any_nan(lower_arm_position):
        return np.nan
    score = 0
    if lower_arm_position <= 60:
        score += 2
    elif 60 < lower_arm_position <= 100:
        score += 1
    elif 100 < lower_arm_position:
        score += 2
    if is_lower_arm_crossed_or_outside:
        score += 1
    return max(1, min(score, 4))

def wrist_score(wrist_position, is_wrist_bent):
    if _any_nan(wrist_position):
        return np.nan
    score = 0
    if -5 <= wrist_position <= 5:
        score += 1
    elif -25 <= wrist_position <= 25:
        score += 2
    elif -25 > wrist_position:
        score += 3
    elif 25 < wrist_position:
        score += 3
    if is_wrist_bent:
        score += 1
    return max(1, min(score, 4))

def wrist_twist_score(is_wrist_twisted_mid, is_wrist_twisted_max):
    score = 0
    if is_wrist_twisted_mid:
        score += 1
    if is_wrist_twisted_max:
        score += 2
    return max(1, min(score, 2))

def posture_score_A(
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
):
    if _any_nan(
        upper_arm_position_left,
        upper_arm_position_right,
        is_upper_arm_abducted_left,
        is_upper_arm_abducted_right,
        lower_arm_position_left,
        lower_arm_position_right,
        wrist_position_left,
        wrist_position_right,
        is_wrist_bent_left,
        is_wrist_bent_right
    ):
        if pd.isna(upper_arm_position_left):
            upper_arm_position_left = upper_arm_position_right
            is_upper_arm_abducted_left = is_upper_arm_abducted_right
        elif pd.isna(upper_arm_position_right):
            upper_arm_position_right = upper_arm_position_left
            is_upper_arm_abducted_right = is_upper_arm_abducted_left
        if pd.isna(lower_arm_position_left):
            lower_arm_position_left = lower_arm_position_right
        elif pd.isna(lower_arm_position_right):
            lower_arm_position_right = lower_arm_position_left
        if pd.isna(wrist_position_left):
            wrist_position_left = wrist_position_right
            is_wrist_bent_left = is_wrist_bent_right
        elif pd.isna(wrist_position_right):
            wrist_position_right = wrist_position_left
            is_wrist_bent_right = is_wrist_bent_left
        if _any_nan(
            upper_arm_position_left,
            upper_arm_position_right,
            is_upper_arm_abducted_left,
            is_upper_arm_abducted_right,
            lower_arm_position_left,
            lower_arm_position_right,
            wrist_position_left,
            wrist_position_right,
            is_wrist_bent_left,
            is_wrist_bent_right
        ):
            return np.nan
    upper_arm = max(
        upper_arm_score(
            upper_arm_position_left,
            is_shoulder_raised,
            is_upper_arm_abducted_left,
            is_upper_arm_supported
        ),
        upper_arm_score(
            upper_arm_position_right,
            is_shoulder_raised,
            is_upper_arm_abducted_right,
            is_upper_arm_supported
        )
    )
    lower_arm = max(
        lower_arm_score(lower_arm_position_left, is_lower_arm_crossed_or_outside),
        lower_arm_score(lower_arm_position_right, is_lower_arm_crossed_or_outside)
    )
    wrist = max(
        wrist_score(wrist_position_left, is_wrist_bent_left),
        wrist_score(wrist_position_right, is_wrist_bent_right)
    )
    wrist_twist = wrist_twist_score(is_wrist_twisted_mid, is_wrist_twisted_max)
    posture_score_A = 0
    if upper_arm == 1:
        if lower_arm == 1:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 1
                elif wrist_twist == 2:
                    posture_score_A = 2
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 2
                elif wrist_twist == 2:
                    posture_score_A = 2
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 2
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
        elif lower_arm == 2:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 2
                elif wrist_twist == 2:
                    posture_score_A = 2
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 2
                elif wrist_twist == 2:
                    posture_score_A = 2
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
        elif lower_arm == 3:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 2
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
    elif upper_arm == 2:
        if lower_arm == 1:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 2
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
        elif lower_arm == 2:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
        elif lower_arm == 3:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
    elif upper_arm == 3:
        if lower_arm == 1:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 3
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
        elif lower_arm == 2:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 3
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
        elif lower_arm == 3:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
    elif upper_arm == 4:
        if lower_arm == 1:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
        elif lower_arm == 2:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
        elif lower_arm == 3:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 4
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 4
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 6
                elif wrist_twist == 2:
                    posture_score_A = 6
    elif upper_arm == 5:
        if lower_arm == 1:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 5
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 6
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 6
                elif wrist_twist == 2:
                    posture_score_A = 7
        elif lower_arm == 2:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 5
                elif wrist_twist == 2:
                    posture_score_A = 6
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 6
                elif wrist_twist == 2:
                    posture_score_A = 6
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 6
                elif wrist_twist == 2:
                    posture_score_A = 7
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 7
                elif wrist_twist == 2:
                    posture_score_A = 7
        elif lower_arm == 3:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 6
                elif wrist_twist == 2:
                    posture_score_A = 6
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 6
                elif wrist_twist == 2:
                    posture_score_A = 7
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 7
                elif wrist_twist == 2:
                    posture_score_A = 7
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 7
                elif wrist_twist == 2:
                    posture_score_A = 8
    elif upper_arm == 6:
        if lower_arm == 1:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 7
                elif wrist_twist == 2:
                    posture_score_A = 7
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 7
                elif wrist_twist == 2:
                    posture_score_A = 7
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 7
                elif wrist_twist == 2:
                    posture_score_A = 8
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 8
                elif wrist_twist == 2:
                    posture_score_A = 9
        elif lower_arm == 2:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 8
                elif wrist_twist == 2:
                    posture_score_A = 8
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 8
                elif wrist_twist == 2:
                    posture_score_A = 8
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 8
                elif wrist_twist == 2:
                    posture_score_A = 9
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 9
                elif wrist_twist == 2:
                    posture_score_A = 9
        elif lower_arm == 3:
            if wrist == 1:
                if wrist_twist == 1:
                    posture_score_A = 9
                elif wrist_twist == 2:
                    posture_score_A = 9
            elif wrist == 2:
                if wrist_twist == 1:
                    posture_score_A = 9
                elif wrist_twist == 2:
                    posture_score_A = 9
            elif wrist == 3:
                if wrist_twist == 1:
                    posture_score_A = 9
                elif wrist_twist == 2:
                    posture_score_A = 9
            elif wrist == 4:
                if wrist_twist == 1:
                    posture_score_A = 9
                elif wrist_twist == 2:
                    posture_score_A = 9
    return posture_score_A

def muscle_use_score_A(is_muscle_static_A, is_muscle_repeated_A):
    score = 0
    if is_muscle_static_A or is_muscle_repeated_A:
        score += 1
    return score

def force_load_score_A(load_value_A, is_load_constant_A):
    score = 0
    if load_value_A < 2:
        score += 0
    elif load_value_A < 10 and (not is_load_constant_A):
        score += 1
    elif load_value_A < 10 and is_load_constant_A:
        score += 2
    else:
        score += 3
    return score

def wrist_and_arm_score(
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
):
    if _any_nan(
        upper_arm_position_left,
        upper_arm_position_right,
        is_upper_arm_abducted_left,
        is_upper_arm_abducted_right,
        lower_arm_position_left,
        lower_arm_position_right,
        wrist_position_left,
        wrist_position_right,
        is_wrist_bent_left,
        is_wrist_bent_right
    ):
        if pd.isna(upper_arm_position_left):
            upper_arm_position_left = upper_arm_position_right
            is_upper_arm_abducted_left = is_upper_arm_abducted_right
        elif pd.isna(upper_arm_position_right):
            upper_arm_position_right = upper_arm_position_left
            is_upper_arm_abducted_right = is_upper_arm_abducted_left
        if pd.isna(lower_arm_position_left):
            lower_arm_position_left = lower_arm_position_right
        elif pd.isna(lower_arm_position_right):
            lower_arm_position_right = lower_arm_position_left
        if pd.isna(wrist_position_left):
            wrist_position_left = wrist_position_right
            is_wrist_bent_left = is_wrist_bent_right
        elif pd.isna(wrist_position_right):
            wrist_position_right = wrist_position_left
            is_wrist_bent_right = is_wrist_bent_left
        if _any_nan(
            upper_arm_position_left,
            upper_arm_position_right,
            is_upper_arm_abducted_left,
            is_upper_arm_abducted_right,
            lower_arm_position_left,
            lower_arm_position_right,
            wrist_position_left,
            wrist_position_right,
            is_wrist_bent_left,
            is_wrist_bent_right
        ):
            return np.nan
    posture_A = posture_score_A(
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
    muscle_use_A = muscle_use_score_A(is_muscle_static_A, is_muscle_repeated_A)
    force_load_A = force_load_score_A(load_value_A, is_load_constant_A)
    wrist_and_arm_score = posture_A + muscle_use_A + force_load_A
    return wrist_and_arm_score

def neck_score(neck_position, is_neck_twisted, is_neck_bent):
    if _any_nan(neck_position):
        return np.nan
    score = 0
    if 0 <= neck_position < 10:
        score += 1
    elif 10 <= neck_position < 20:
        score += 2
    elif neck_position >= 20:
        score += 3
    elif neck_position < 0:
        score += 4
    if is_neck_twisted:
        score += 1
    if is_neck_bent:
        score += 1
    return score

def trunk_score(trunk_position, is_trunk_twisted, is_trunk_bent):
    score = 0
    if -5 <= trunk_position <= 5:
        score += 1
    elif 5 < trunk_position <= 20:
        score += 2
    elif 20 < trunk_position <= 60:
        score += 3
    elif trunk_position > 60:
        score += 4
    if is_trunk_twisted:
        score += 1
    if is_trunk_bent:
        score += 1
    return score

def leg_score(is_leg_supported):
    score = 0
    if is_leg_supported:
        score += 1
    else:
        score += 2
    return score

def posture_score_B(
    neck_position,
    is_neck_twisted,
    is_neck_bent,
    trunk_position,
    is_trunk_twisted,
    is_trunk_bent,
    is_leg_supported
):
    if _any_nan(neck_position):
        return np.nan
    neck = neck_score(neck_position, is_neck_twisted, is_neck_bent)
    trunk = trunk_score(trunk_position, is_trunk_twisted, is_trunk_bent)
    leg = leg_score(is_leg_supported)
    posture_score_B = 0
    if neck == 1:
        if trunk == 1:
            if leg == 1:
                posture_score_B += 1
            elif leg == 2:
                posture_score_B += 3
        elif trunk == 2:
            if leg == 1:
                posture_score_B += 2
            elif leg == 2:
                posture_score_B += 3
        elif trunk == 3:
            if leg == 1:
                posture_score_B += 3
            elif leg == 2:
                posture_score_B += 4
        elif trunk == 4:
            if leg == 1:
                posture_score_B += 5
            elif leg == 2:
                posture_score_B += 5
        elif trunk == 5:
            if leg == 1:
                posture_score_B += 6
            elif leg == 2:
                posture_score_B += 6
        elif trunk == 6:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
    elif neck == 2:
        if trunk == 1:
            if leg == 1:
                posture_score_B += 2
            elif leg == 2:
                posture_score_B += 3
        elif trunk == 2:
            if leg == 1:
                posture_score_B += 2
            elif leg == 2:
                posture_score_B += 3
        elif trunk == 3:
            if leg == 1:
                posture_score_B += 4
            elif leg == 2:
                posture_score_B += 5
        elif trunk == 4:
            if leg == 1:
                posture_score_B += 5
            elif leg == 2:
                posture_score_B += 5
        elif trunk == 5:
            if leg == 1:
                posture_score_B += 6
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 6:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
    elif neck == 3:
        if trunk == 1:
            if leg == 1:
                posture_score_B += 3
            elif leg == 2:
                posture_score_B += 3
        elif trunk == 2:
            if leg == 1:
                posture_score_B += 3
            elif leg == 2:
                posture_score_B += 4
        elif trunk == 3:
            if leg == 1:
                posture_score_B += 4
            elif leg == 2:
                posture_score_B += 5
        elif trunk == 4:
            if leg == 1:
                posture_score_B += 5
            elif leg == 2:
                posture_score_B += 6
        elif trunk == 5:
            if leg == 1:
                posture_score_B += 6
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 6:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
    elif neck == 4:
        if trunk == 1:
            if leg == 1:
                posture_score_B += 5
            elif leg == 2:
                posture_score_B += 5
        elif trunk == 2:
            if leg == 1:
                posture_score_B += 5
            elif leg == 2:
                posture_score_B += 6
        elif trunk == 3:
            if leg == 1:
                posture_score_B += 6
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 4:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 5:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 6:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
    elif neck == 5:
        if trunk == 1:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 2:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 7
        elif trunk == 3:
            if leg == 1:
                posture_score_B += 7
            elif leg == 2:
                posture_score_B += 8
        elif trunk == 4:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
        elif trunk == 5:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
        elif trunk == 6:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
    elif neck == 6:
        if trunk == 1:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
        elif trunk == 2:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
        elif trunk == 3:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 8
        elif trunk == 4:
            if leg == 1:
                posture_score_B += 8
            elif leg == 2:
                posture_score_B += 9
        elif trunk == 5:
            if leg == 1:
                posture_score_B += 9
            elif leg == 2:
                posture_score_B += 9
        elif trunk == 6:
            if leg == 1:
                posture_score_B += 9
            elif leg == 2:
                posture_score_B += 9
    return posture_score_B

def muscle_use_score_B(is_muscle_static_B, is_muscle_repeated_B):
    score = 0
    if is_muscle_static_B or is_muscle_repeated_B:
        score += 1
    return score

def force_load_score_B(load_value_B, is_load_constant_B):
    score = 0
    if load_value_B < 2:
        score += 0
    elif load_value_B < 10 and (not is_load_constant_B):
        score += 1
    elif load_value_B < 10 and is_load_constant_B:
        score += 2
    else:
        score += 3
    return score

def neck_trunk_leg_score(
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
):
    if _any_nan(neck_position):
        return np.nan
    posture_B = posture_score_B(
        neck_position,
        is_neck_twisted,
        is_neck_bent,
        trunk_position,
        is_trunk_twisted,
        is_trunk_bent,
        is_leg_supported
    )
    muscle_use_B = muscle_use_score_B(is_muscle_static_B, is_muscle_repeated_B)
    force_load_B = force_load_score_B(load_value_B, is_load_constant_B)
    neck_trunk_leg_score = posture_B + muscle_use_B + force_load_B
    return neck_trunk_leg_score

def rula_score(
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
):
    wrist_and_arm = wrist_and_arm_score(
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
    neck_trunk_leg = neck_trunk_leg_score(
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
    if _any_nan(
        upper_arm_position_left,
        upper_arm_position_right,
        is_upper_arm_abducted_left,
        is_upper_arm_abducted_right,
        lower_arm_position_left,
        lower_arm_position_right,
        wrist_position_left,
        wrist_position_right,
        is_wrist_bent_left,
        is_wrist_bent_right
    ):
        if pd.isna(upper_arm_position_left):
            upper_arm_position_left = upper_arm_position_right
            is_upper_arm_abducted_left = is_upper_arm_abducted_right
        elif pd.isna(upper_arm_position_right):
            upper_arm_position_right = upper_arm_position_left
            is_upper_arm_abducted_right = is_upper_arm_abducted_left
        if pd.isna(lower_arm_position_left):
            lower_arm_position_left = lower_arm_position_right
        elif pd.isna(lower_arm_position_right):
            lower_arm_position_right = lower_arm_position_left
        if pd.isna(wrist_position_left):
            wrist_position_left = wrist_position_right
            is_wrist_bent_left = is_wrist_bent_right
        elif pd.isna(wrist_position_right):
            wrist_position_right = wrist_position_left
            is_wrist_bent_right = is_wrist_bent_left
        if _any_nan(
            upper_arm_position_left,
            upper_arm_position_right,
            is_upper_arm_abducted_left,
            is_upper_arm_abducted_right,
            lower_arm_position_left,
            lower_arm_position_right,
            wrist_position_left,
            wrist_position_right,
            is_wrist_bent_left,
            is_wrist_bent_right,
            neck_position
        ):
            return np.nan
    rula_score = 0
    if wrist_and_arm == 1:
        if neck_trunk_leg == 1:
            rula_score += 1
        elif neck_trunk_leg == 2:
            rula_score += 2
        elif neck_trunk_leg == 3:
            rula_score += 3
        elif neck_trunk_leg == 4:
            rula_score += 3
        elif neck_trunk_leg == 5:
            rula_score += 4
        elif neck_trunk_leg == 6:
            rula_score += 5
        elif neck_trunk_leg >= 7:
            rula_score += 5
    elif wrist_and_arm == 2:
        if neck_trunk_leg == 1:
            rula_score += 2
        elif neck_trunk_leg == 2:
            rula_score += 2
        elif neck_trunk_leg == 3:
            rula_score += 3
        elif neck_trunk_leg == 4:
            rula_score += 4
        elif neck_trunk_leg == 5:
            rula_score += 4
        elif neck_trunk_leg == 6:
            rula_score += 5
        elif neck_trunk_leg >= 7:
            rula_score += 5
    elif wrist_and_arm == 3:
        if neck_trunk_leg == 1:
            rula_score += 3
        elif neck_trunk_leg == 2:
            rula_score += 3
        elif neck_trunk_leg == 3:
            rula_score += 3
        elif neck_trunk_leg == 4:
            rula_score += 4
        elif neck_trunk_leg == 5:
            rula_score += 4
        elif neck_trunk_leg == 6:
            rula_score += 5
        elif neck_trunk_leg >= 7:
            rula_score += 6
    elif wrist_and_arm == 4:
        if neck_trunk_leg == 1:
            rula_score += 3
        elif neck_trunk_leg == 2:
            rula_score += 3
        elif neck_trunk_leg == 3:
            rula_score += 3
        elif neck_trunk_leg == 4:
            rula_score += 4
        elif neck_trunk_leg == 5:
            rula_score += 5
        elif neck_trunk_leg == 6:
            rula_score += 6
        elif neck_trunk_leg >= 7:
            rula_score += 6
    elif wrist_and_arm == 5:
        if neck_trunk_leg == 1:
            rula_score += 4
        elif neck_trunk_leg == 2:
            rula_score += 4
        elif neck_trunk_leg == 3:
            rula_score += 4
        elif neck_trunk_leg == 4:
            rula_score += 5
        elif neck_trunk_leg == 5:
            rula_score += 6
        elif neck_trunk_leg == 6:
            rula_score += 7
        elif neck_trunk_leg >= 7:
            rula_score += 7
    elif wrist_and_arm == 6:
        if neck_trunk_leg == 1:
            rula_score += 4
        elif neck_trunk_leg == 2:
            rula_score += 4
        elif neck_trunk_leg == 3:
            rula_score += 5
        elif neck_trunk_leg == 4:
            rula_score += 6
        elif neck_trunk_leg == 5:
            rula_score += 6
        elif neck_trunk_leg == 6:
            rula_score += 7
        elif neck_trunk_leg >= 7:
            rula_score += 7
    elif wrist_and_arm == 7:
        if neck_trunk_leg == 1:
            rula_score += 5
        elif neck_trunk_leg == 2:
            rula_score += 5
        elif neck_trunk_leg == 3:
            rula_score += 6
        elif neck_trunk_leg == 4:
            rula_score += 6
        elif neck_trunk_leg == 5:
            rula_score += 7
        elif neck_trunk_leg == 6:
            rula_score += 7
        elif neck_trunk_leg >= 7:
            rula_score += 7
    elif wrist_and_arm >= 8:
        if neck_trunk_leg == 1:
            rula_score += 5
        elif neck_trunk_leg == 2:
            rula_score += 5
        elif neck_trunk_leg == 3:
            rula_score += 6
        elif neck_trunk_leg == 4:
            rula_score += 7
        elif neck_trunk_leg == 5:
            rula_score += 7
        elif neck_trunk_leg == 6:
            rula_score += 7
        elif neck_trunk_leg >= 7:
            rula_score += 7
    return rula_score
