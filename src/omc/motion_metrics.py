"""Read OMC hand-motion exports and summarize vector magnitudes."""

import csv
from pathlib import Path
import numpy as np
import pandas as pd
DELIMS = [',', '\t', ';', '|']
SAMPLE_N = 200

def read_with_delim(path: Path, d):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f, delimiter=d))

def choose_best_delim(path):
    best, best_score = (DELIMS[0], (-1, -1))
    for d in DELIMS:
        try:
            rows = read_with_delim(path, d)
        except:
            continue
        if not rows:
            continue
        cnts = [len(r) for r in rows[:SAMPLE_N] if r]
        if not cnts:
            continue
        uniq = len(set(cnts))
        med = sorted(cnts)[len(cnts) // 2]
        score = (-uniq, med)
        if score > best_score:
            best, best_score = (d, score)
    return best

def fix_rows(rows):
    cnts = [len(r) for r in rows[:SAMPLE_N] if r]
    if not cnts:
        return rows
    med = sorted(cnts)[len(cnts) // 2]
    if med != 1:
        return rows
    fixed = []
    for r in rows:
        if len(r) == 1 and ',' in r[0]:
            fixed.append(next(csv.reader([r[0]], delimiter=',')))
        else:
            fixed.append(r)
    return fixed

def to_float(x):
    try:
        return float(str(x).strip())
    except:
        return np.nan

def mag(arr):
    return np.sqrt(np.nansum(np.square(arr), axis=1))

def extract_metrics(df, hand_key):
    out = {}
    nrows, ncols = df.shape
    starts = []
    for c in range(ncols):
        col = df.iloc[:, c].astype(str)
        rows = col.index[col == hand_key].tolist()
        for r in rows:
            if r + 1 < nrows:
                hdr = df.iloc[r + 1, c:c + 9].astype(str).tolist()
                if hdr[:3] in (["RX'", "RY'", "RZ'"], ["RX''", "RY''", "RZ''"]):
                    starts.append((r, c))
    ang_all = []
    for r, c in starts:
        hdr = df.iloc[r + 1, c:c + 9].astype(str).tolist()
        if hdr[:3] == ["RX'", "RY'", "RZ'"]:
            arr = df.iloc[r + 2:, c:c + 3].map(to_float).to_numpy(float)
            ang_all.append(mag(arr))
    ang_mean = float(np.nanmean(np.concatenate(ang_all))) if ang_all else np.nan
    lin_all = []
    for r, c in starts:
        hdr = df.iloc[r + 1, c:c + 9].astype(str).tolist()
        if hdr[3:6] == ["TX''", "TY''", "TZ''"]:
            arr = df.iloc[r + 2:, c + 3:c + 6].map(to_float).to_numpy(float)
            lin_all.append(mag(arr))
    lin_mean = float(np.nanmean(np.concatenate(lin_all))) if lin_all else np.nan
    return (lin_mean, ang_mean)
