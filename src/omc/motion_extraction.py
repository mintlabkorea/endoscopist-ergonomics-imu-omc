"""Extract hand acceleration and angular velocity from OMC exports."""

from pathlib import Path
import re, csv
import pandas as pd
RAW_DIR = Path('../data/raw')
OUT_DIR = Path('../data/accel')
FNAME_PAT = re.compile('^(?:c|g)_([^_]+)_.*\\.csv$', re.IGNORECASE)
DELIMS = [',', '\t', ';', '|']
SCORE_SAMPLE_ROWS = 200

def read_with_delim(path: Path, delim: str):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f, delimiter=delim))

def choose_best_delim(path: Path) -> str:
    best, best_score = (DELIMS[0], (-1, -1.0))
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
        med = sorted(counts)[len(counts) // 2]
        score = (-uniq, med)
        if score > best_score:
            best, best_score = (d, score)
    return best

def reparsed_by_comma(cell: str):
    txt = cell or ''
    if txt.startswith('"') and txt.endswith('"') or (txt.startswith("'") and txt.endswith("'")):
        txt = txt[1:-1]
    return next(csv.reader([txt], delimiter=','))

def fix_rows_if_single_column(rows):
    if not rows:
        return rows
    counts = [len(r) for r in rows[:SCORE_SAMPLE_ROWS] if r]
    if not counts:
        return rows
    med = sorted(counts)[len(counts) // 2]
    if med != 1:
        return rows
    fixed = []
    for r in rows:
        if len(r) == 1 and ',' in r[0]:
            fixed.append(reparsed_by_comma(r[0]))
        else:
            fixed.append(r)
    return fixed

def inclusive_iloc(df, r0, r1, c0, c1):
    r0 = max(0, r0)
    r1 = min(df.shape[0] - 1, r1)
    c0 = max(0, c0)
    c1 = min(df.shape[1] - 1, c1)
    if r1 < r0 or c1 < c0:
        return None
    return df.iloc[r0:r1 + 1, c0:c1 + 1]

def process_one(csv_path: Path):
    m = FNAME_PAT.match(csv_path.name)
    if not m:
        print(f'[SKIP] name rule mismatch: {csv_path.name}')
        return
    initials = m.group(1)
    delim = choose_best_delim(csv_path)
    rows = read_with_delim(csv_path, delim)
    rows = fix_rows_if_single_column(rows)
    df = pd.DataFrame(rows, dtype=str)
    df = df.fillna('')
    search_keywords = [f'{initials}:RHN', f'{initials}:LHN']
    slices = []
    nrows, ncols = df.shape
    for sk in search_keywords:
        for cn in range(ncols):
            col_vals = df.iloc[:, cn]
            matches = col_vals[col_vals == sk].index.tolist()
            for rn in matches:
                nxt = rn + 1
                if nxt < nrows:
                    v = df.iat[nxt, cn]
                    if v in ("RX'", "RX''"):
                        r0 = rn
                        r1 = 2 * rn - 10
                        c0 = cn
                        c1 = cn + 8
                        sl = inclusive_iloc(df, r0, r1, c0, c1)
                        if sl is not None and (not sl.empty):
                            slices.append(sl.reset_index(drop=True))
    out_df = pd.concat(slices, axis=1) if slices else pd.DataFrame()
    out_path = OUT_DIR / csv_path.name
    out_df.to_csv(out_path, index=False, header=False, encoding='utf-8-sig')
    print(f'[OK] saved: {out_path} | slices={len(slices)} | delim={repr(delim)}')
