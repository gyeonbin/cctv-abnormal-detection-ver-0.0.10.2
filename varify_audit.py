#!/usr/bin/env python3
# verify_audit.py
# Decode ↔ YOLO ↔ Track 프레임 일치/연속성 검증

import os
import re
import csv
import glob
import sys
import argparse
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser(description="Verify decode/yolo/track audit logs")
    ap.add_argument("--dir", "-d", default="./audit", help="Audit directory (default: ./audit)")
    ap.add_argument("--stream", "-s", default=None, help="Check only this stream id (e.g., 122-1_cam01...)")
    ap.add_argument("--limit", "-n", type=int, default=20, help="Max items to print in missing/extras (default: 20)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if decode continuity has gaps (normally allowed)")
    return ap.parse_args()

def read_csv_rows(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    return rows

def col_index(header, name, fallback_idx=None):
    try:
        return header.index(name)
    except Exception:
        return fallback_idx

def to_int(s):
    try:
        return int(s)
    except Exception:
        return None

def load_decode(dirpath):
    """return dict: sid -> list of (fid, pts?)"""
    out = {}
    for p in glob.glob(os.path.join(dirpath, "decode_*.csv")):
        m = re.search(r"decode_(.+)\.csv$", os.path.basename(p))
        if not m:
            continue
        sid = m.group(1)
        rows = read_csv_rows(p)
        if not rows:
            out[sid] = []
            continue
        header = rows[0]
        data = rows[1:] if any(h.lower() == "fid" for h in header) else rows  # 헤더 중복 방지
        i_fid = col_index(header, "fid", 1)     # 보통 2번째 컬럼
        i_pts = col_index(header, "pts", 2)     # 보통 3번째 컬럼
        fids = []
        for row in data:
            if not row:
                continue
            fid = to_int(row[i_fid]) if i_fid is not None and i_fid < len(row) else None
            if fid is None:
                continue
            fids.append((fid, row[i_pts] if i_pts is not None and i_pts < len(row) else ""))
        out[sid] = fids
    return out

def load_yolo(dirpath):
    """return dict: sid -> list of (fid, pts?)"""
    out = {}
    for p in glob.glob(os.path.join(dirpath, "yolo_*.csv")):
        m = re.search(r"yolo_(.+)\.csv$", os.path.basename(p))
        if not m:
            continue
        sid = m.group(1)
        rows = read_csv_rows(p)
        if not rows:
            out[sid] = []
            continue
        header = rows[0]
        data = rows[1:] if any(h.lower() == "fid" for h in header) else rows
        i_fid = col_index(header, "fid", 1)
        i_pts = col_index(header, "pts", 2)
        fids = []
        for row in data:
            if not row:
                continue
            fid = to_int(row[i_fid]) if i_fid is not None and i_fid < len(row) else None
            if fid is None:
                continue
            fids.append((fid, row[i_pts] if i_pts is not None and i_pts < len(row) else ""))
        out[sid] = fids
    return out

def load_track(dirpath):
    """
    track.csv (단일 파일, sid 컬럼 포함) 또는 track_*.csv (스트림별 파일) 모두 지원
    return dict: sid -> list of (fid, pts?)
    """
    out = defaultdict(list)

    # case A: track.csv
    p = os.path.join(dirpath, "track.csv")
    if os.path.exists(p):
        rows = read_csv_rows(p)
        if rows:
            header = rows[0]
            data = rows[1:] if any(h.lower() == "fid" for h in header) else rows
            i_sid = col_index(header, "sid", None)
            i_fid = col_index(header, "fid", 2 if i_sid is not None else 1)
            i_pts = col_index(header, "pts", 3 if i_sid is not None else 2)
            for row in data:
                if not row:
                    continue
                sid = (row[i_sid] if i_sid is not None and i_sid < len(row) else None)
                fid = to_int(row[i_fid]) if i_fid is not None and i_fid < len(row) else None
                if sid is None or fid is None:
                    continue
                out[sid].append((fid, row[i_pts] if i_pts is not None and i_pts < len(row) else ""))
    # case B: track_*.csv (optional)
    for p in glob.glob(os.path.join(dirpath, "track_*.csv")):
        m = re.search(r"track_(.+)\.csv$", os.path.basename(p))
        if not m:
            continue
        sid = m.group(1)
        rows = read_csv_rows(p)
        if not rows:
            continue
        header = rows[0]
        data = rows[1:] if any(h.lower() == "fid" for h in header) else rows
        i_fid = col_index(header, "fid", 1)
        i_pts = col_index(header, "pts", 2)
        for row in data:
            if not row:
                continue
            fid = to_int(row[i_fid]) if i_fid is not None and i_fid < len(row) else None
            if fid is None:
                continue
            out[sid].append((fid, row[i_pts] if i_pts is not None and i_pts < len(row) else ""))

    return dict(out)

def continuity_stats(fids_sorted):
    """
    fids_sorted: sorted list of fids
    return: (lost_count, dup_count)
    """
    if not fids_sorted:
        return 0, 0
    lost = 0
    dup = 0
    prev = None
    seen = set()
    for fid in fids_sorted:
        if prev is None:
            prev = fid
            seen.add(fid)
            continue
        if fid in seen:
            dup += 1
        else:
            if fid != prev + 1:
                if fid > prev + 1:
                    lost += (fid - (prev + 1))
                # fid < prev+1는 out-of-order지만 여기서는 dup로만 집계
            seen.add(fid)
            prev = fid if fid > prev else prev
            if fid == prev + 1:
                prev = fid
    return lost, dup

def brief(name, fids):
    if not fids:
        return f"{name}: count=0"
    only = [fid for fid, _ in fids]
    return f"{name}: count={len(only)} min={min(only)} max={max(only)} uniq={len(set(only))}"

def main():
    args = parse_args()
    dirpath = args.dir

    dec = load_decode(dirpath)
    yol = load_yolo(dirpath)
    trk = load_track(dirpath)

    all_sids = set(dec.keys()) | set(yol.keys()) | set(trk.keys())
    if args.stream:
        all_sids = {args.stream}

    if not all_sids:
        print("No streams found. Make sure your audit files exist in:", dirpath)
        return 1

    overall_fail = False

    for sid in sorted(all_sids):
        dec_f = dec.get(sid, [])
        yol_f = yol.get(sid, [])
        trk_f = trk.get(sid, [])

        dec_ids = [fid for fid, _ in dec_f]
        yol_ids = [fid for fid, _ in yol_f]
        trk_ids = [fid for fid, _ in trk_f]

        Sdec, Syol, Strk = set(dec_ids), set(yol_ids), set(trk_ids)

        miss_yol = sorted(Sdec - Syol)
        miss_trk = sorted(Sdec - Strk)
        extra_yol = sorted(Syol - Sdec)
        extra_trk = sorted(Strk - Sdec)

        # continuity (개별 스트림 파일 내부 연속성)
        d_lost, d_dup = continuity_stats(sorted(dec_ids))
        y_lost, y_dup = continuity_stats(sorted(yol_ids))
        t_lost, t_dup = continuity_stats(sorted(trk_ids))

        print("\n======================================")
        print(f"STREAM: {sid}")
        print(brief("DECODE", dec_f))
        print(brief("YOLO  ", yol_f))
        print(brief("TRACK ", trk_f))

        # 핵심 판정: YOLO/Track이 Decode 기준으로 누락 0인지
        pass_core = (len(miss_yol) == 0) and (len(miss_trk) == 0)
        # 엄격 모드면 디코드도 자체 연속성 검사
        pass_strict = True
        if args.strict and (d_lost > 0 or d_dup > 0):
            pass_strict = False

        if pass_core and pass_strict:
            print("RESULT: ✅ PASS")
        else:
            print("RESULT: ❌ FAIL")
            overall_fail = True

        # 상세 리포트
        if miss_yol:
            print(f"Missing from YOLO (vs DECODE) [showing up to {args.limit}]:", miss_yol[:args.limit])
        if miss_trk:
            print(f"Missing from TRACK (vs DECODE) [showing up to {args.limit}]:", miss_trk[:args.limit])
        if extra_yol:
            print(f"Extras in YOLO (not in DECODE) [showing up to {args.limit}]:", extra_yol[:args.limit])
        if extra_trk:
            print(f"Extras in TRACK (not in DECODE) [showing up to {args.limit}]:", extra_trk[:args.limit])

        print(f"Continuity — DECODE lost={d_lost} dup={d_dup} | YOLO lost={y_lost} dup={y_dup} | TRACK lost={t_lost} dup={t_dup}")

    return 1 if overall_fail else 0


if __name__ == "__main__":
    sys.exit(main())
