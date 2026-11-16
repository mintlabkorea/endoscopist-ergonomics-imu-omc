"""Extract neck and upper-limb angles from OMC exports."""

import csv
from pathlib import Path
from statistics import median
import pandas as pd
import numpy as np
import traceback
RAW_DIR = Path('../data/raw')
PROCESSED_DIR = Path('../data/processed')
TOKENS_DROP_BY_ROW2 = ('TX', 'TY', 'TZ', 'SX', 'SY', 'SZ', "'")
TOKENS_KEEP_TRIPLE_BY_ROW1 = ('ELBOW', 'NECK', 'SHOULDER', 'WRIST', 'THORAX')
DELIMS = [',', '\t', ';', '|']
SCORE_SAMPLE_ROWS = 200

def read_with_delim(path: Path, delim: str):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f, delimiter=delim))

def choose_best_delim(path: Path) -> str:
    best = DELIMS[0]
    best_score = (-1, -1.0)
    for d in DELIMS:
        try:
            rows = read_with_delim(path, d)
        except Exception:
            continue
        if not rows:
            continue
        counts = [len(r) for r in rows[:SCORE_SAMPLE_ROWS] if r]
        if not counts:
            continue
        uniq = len(set(counts))
        med = median(counts)
        score = (-uniq, med)
        if score > best_score:
            best_score = score
            best = d
    return best

def reparsed_by_comma(cell: str):
    txt = cell or ''
    if txt.startswith('"') and txt.endswith('"') or (txt.startswith("'") and txt.endswith("'")):
        txt = txt[1:-1]
    return next(csv.reader([txt], delimiter=','))

def normalize(v: str) -> str:
    return (v or '').strip().upper()

def fix_rows_if_single_column(rows):
    if not rows:
        return (rows, False)
    counts = [len(r) for r in rows[:SCORE_SAMPLE_ROWS] if r]
    if not counts:
        return (rows, False)
    if median(counts) == 1:
        fixed, changed = ([], False)
        for r in rows:
            if len(r) == 1 and ',' in r[0]:
                fixed.append(reparsed_by_comma(r[0]))
                changed = True
            else:
                fixed.append(r)
        return (fixed, changed)
    return (rows, False)

def to_num(s):
    try:
        return float(str(s).strip())
    except Exception:
        return None

def to_numeric_series(s):
    return pd.to_numeric(s, errors='coerce')

def rename_by_substring(df, sub, new_name):
    hit = [c for c in df.columns if sub in c]
    for c in hit:
        df = df.rename(columns={c: new_name})
    return df

def mean_neck_signed(df, substr):
    cols_pos = [c for c in df.columns if substr in c and 'LNeckAngles' not in c]
    cols_neg = [c for c in df.columns if substr in c and 'LNeckAngles' in c]
    data_pos = df[cols_pos].apply(to_numeric_series) if cols_pos else pd.DataFrame()
    data_neg = df[cols_neg].apply(to_numeric_series) * -1 if cols_neg else pd.DataFrame()
    all_data = pd.concat([data_pos, data_neg], axis=1)
    if all_data.empty:
        return pd.Series([np.nan] * len(df))
    return all_data.mean(axis=1)

def mean_neck(df, substr):
    cols = [c for c in df.columns if substr in c]
    if not cols:
        return pd.Series([np.nan] * len(df))
    return df[cols].apply(to_numeric_series).mean(axis=1)

def process_one(src_path: Path, out_path: Path):
    delim = choose_best_delim(src_path)
    rows = read_with_delim(src_path, delim)
    rows, repaired = fix_rows_if_single_column(rows)
    if not rows:
        raise ValueError('CSV is empty.')
    first_col = [r[0].strip() if r else '' for r in rows]
    try:
        N_1b = first_col.index('Model Outputs') + 1
    except ValueError:
        raise ValueError('Model Outputs was not found in the first column.')
    try:
        M_1b = first_col.index('Segments') + 1
    except ValueError:
        raise ValueError('Segments was not found in the first column.')
    if M_1b <= N_1b:
        raise ValueError('Segments appears before Model Outputs.')
    start_1b = N_1b + 2
    end_1b = M_1b - 1
    if start_1b > end_1b:
        raise ValueError('The selected section is empty. Check the input file structure.')
    cropped = rows[start_1b - 1:end_1b]
    if not cropped:
        raise ValueError('The cropped section is empty.')
    max_len = max((len(r) for r in cropped))
    cropped = [r + [''] * (max_len - len(r)) for r in cropped]
    if len(cropped) < 2:
        raise ValueError('The second header row is missing from the cropped section.')
    row2 = cropped[1]
    drop_idx = [j for j, v in enumerate(row2) if any((tok in normalize(v) for tok in TOKENS_DROP_BY_ROW2))]
    keep_idx = [j for j in range(max_len) if j not in drop_idx] or list(range(max_len))
    filtered = [[row[j] for j in keep_idx] for row in cropped]
    max_len2 = max((len(r) for r in filtered))
    filtered = [r + [''] * (max_len2 - len(r)) for r in filtered]
    row1 = filtered[0]
    candidate_k = [
        j for j in range(2, max_len2) if any((tok in normalize(row1[j]) for tok in TOKENS_KEEP_TRIPLE_BY_ROW1))
    ]
    if candidate_k:
        keep_set = {0, 1}
        for k in candidate_k:
            if k < max_len2:
                keep_set.add(k)
            if k + 1 < max_len2:
                keep_set.add(k + 1)
            if k + 2 < max_len2:
                keep_set.add(k + 2)
        keep_sorted = sorted(keep_set)
        final_rows = [[r[j] for j in keep_sorted] for r in filtered]
    else:
        final_rows = filtered
    if len(final_rows) >= 3:
        final_rows.pop(2)
    for r in final_rows:
        if len(r) >= 2:
            del r[1]
    max_len3 = max((len(r) for r in final_rows))
    final_rows = [r + [''] * (max_len3 - len(r)) for r in final_rows]
    if len(final_rows) < 2:
        raise ValueError('At least two rows are required to reconstruct the header.')
    group_start = 1
    if max_len3 - 1 > 0:
        n_groups = (max_len3 - 1 + 2) // 3
        for g in range(n_groups):
            base = group_start + g * 3
            if base >= max_len3:
                break
            group_name_raw = (final_rows[0][base] or '').strip()
            for j in range(base, min(base + 3, max_len3)):
                sub_raw = (final_rows[1][j] or '').strip()
                final_rows[1][j] = f'{group_name_raw}_{sub_raw}' if group_name_raw else sub_raw
    final_rows.pop(0)
    if final_rows and final_rows[0]:
        final_rows[0][0] = 'time'
    elif final_rows and (not final_rows[0]):
        final_rows[0].append('time')
    for i in range(1, len(final_rows)):
        if not final_rows[i]:
            continue
        num = to_num(final_rows[i][0])
        if num is not None:
            final_rows[i][0] = f'{(num - 1) / 100:.10g}'
    header = final_rows[0]
    data = final_rows[1:]
    df = pd.DataFrame(data, columns=header)
    df = rename_by_substring(df, 'LElbowAngles_X', 'l_elbow_Flex')
    df = rename_by_substring(df, 'RElbowAngles_X', 'r_elbow_Flex')
    df = df.loc[:, ~df.columns.str.contains('ElbowAngles')]
    df['neck_Flex'] = mean_neck(df, 'NeckAngles_X')
    df['neck_Twist'] = mean_neck_signed(df, 'NeckAngles_Y')
    df['neck_Bend'] = mean_neck_signed(df, 'NeckAngles_Z')
    df = df.loc[:, ~df.columns.str.contains('NeckAngles')]
    df = rename_by_substring(df, 'LShoulderAngles_X', 'l_shoulder_Flex')
    df = rename_by_substring(df, 'LShoulderAngles_Y', 'l_shoulder_Twist')
    df = rename_by_substring(df, 'LShoulderAngles_Z', 'l_shoulder_Abd')
    df = rename_by_substring(df, 'RShoulderAngles_X', 'r_shoulder_Flex')
    df = rename_by_substring(df, 'RShoulderAngles_Y', 'r_shoulder_Twist')
    df = rename_by_substring(df, 'RShoulderAngles_Z', 'r_shoulder_Abd')
    df = df.loc[:, ~df.columns.str.contains('Thorax')]
    df = rename_by_substring(df, 'LWristAngles_X', 'l_wrist_Flex')
    df = rename_by_substring(df, 'LWristAngles_Y', 'l_wrist_Twist')
    df = rename_by_substring(df, 'LWristAngles_Z', 'l_wrist_Abd')
    df = rename_by_substring(df, 'RWristAngles_X', 'r_wrist_Flex')
    df = rename_by_substring(df, 'RWristAngles_Y', 'r_wrist_Twist')
    df = rename_by_substring(df, 'RWristAngles_Z', 'r_wrist_Abd')
    order = [
        'time',
        'neck_Flex',
        'neck_Twist',
        'neck_Bend',
        'r_shoulder_Flex',
        'r_shoulder_Abd',
        'r_shoulder_Twist',
        'l_shoulder_Flex',
        'l_shoulder_Abd',
        'l_shoulder_Twist',
        'r_elbow_Flex',
        'l_elbow_Flex',
        'r_wrist_Flex',
        'r_wrist_Abd',
        'r_wrist_Twist',
        'l_wrist_Flex',
        'l_wrist_Abd',
        'l_wrist_Twist'
    ]
    for col in order:
        if col not in df.columns:
            df[col] = np.nan
    df = df[order]
    num_cols = [c for c in df.columns if c != 'time']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'OK Saved: {out_path.resolve()} | Shape: {df.shape} | DelimDetected: {repr(delim)}')
