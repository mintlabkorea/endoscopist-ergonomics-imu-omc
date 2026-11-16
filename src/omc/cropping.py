"""Select the analyzed OMC trial interval from movement endpoints."""

from pathlib import Path
import pandas as pd
import numpy as np
import traceback
PROC_DIR = Path('../data/processed')
LOG_FILE = PROC_DIR / 'log.txt'

def find_tstart_tend(df: pd.DataFrame):
    if df.shape[1] < 2:
        raise ValueError('The second column is missing.')
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    vals = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    n = len(vals)
    if n == 0:
        raise ValueError('The input contains no data rows.')
    w = min(1001, n)
    t_start_idx = vals.iloc[:w].idxmax(skipna=True)
    tail = vals.iloc[n - w:]
    t_end_local_idx = tail.idxmax(skipna=True)
    t_end_idx = t_end_local_idx
    if not time.isna().all():
        t_start = time.iloc[t_start_idx]
        t_end = time.iloc[t_end_idx]
    else:
        t_start = t_start_idx
        t_end = t_end_idx
    if t_end_idx < t_start_idx:
        t_start_idx, t_end_idx = (t_end_idx, t_start_idx)
        t_start, t_end = (t_end, t_start)
    return (t_start, t_end, t_start_idx, t_end_idx)

def process_file(path: Path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    t_start, t_end, t_start_idx, t_end_idx = find_tstart_tend(df)
    diff = t_end - t_start
    df_crop = df.iloc[t_start_idx:t_end_idx + 1].copy()
    time_col = df_crop.columns[0]
    time_vals = pd.to_numeric(df_crop.iloc[:, 0], errors='coerce')
    if time_vals.isna().all():
        df_crop[time_col] = np.arange(len(df_crop))
    else:
        df_crop[time_col] = time_vals - (t_start if pd.notna(t_start) else 0)
    df_crop.to_csv(path, index=False, encoding='utf-8-sig')
    return diff
