# -*- coding: utf-8 -*-
# utils/inmem_logger.py
import os, csv, numpy as np

class InMemoryCsvLoggerNP:
    """
    매우 가벼운 메모리 로거: 런타임에는 메모리에만 누적하고,
    종료 시 dump(csv_path)로 한 번에 저장.
    """
    def __init__(self, header, initial_capacity=12000):
        self.header = list(header)
        self.cap = int(initial_capacity)
        self.n = 0
        # 열 기반 배열 (정밀도는 필요에 맞게 조정)
        self.t_wall = np.empty(self.cap, dtype=np.float64)
        self.fid    = np.empty(self.cap, dtype=np.int32)
        self.pts    = np.empty(self.cap, dtype=np.float64)
        self.seq    = np.empty(self.cap, dtype=np.int64)

    def _grow(self):
        new_cap = max(self.cap * 2, 1)
        self.t_wall = np.resize(self.t_wall, new_cap)
        self.fid    = np.resize(self.fid,    new_cap)
        self.pts    = np.resize(self.pts,    new_cap)
        self.seq    = np.resize(self.seq,    new_cap)
        self.cap = new_cap

    def row(self, values):
        if self.n >= self.cap:
            self._grow()
        t_wall, fid, pts, seq = values
        self.t_wall[self.n] = float(t_wall)
        self.fid[self.n]    = int(fid)
        self.pts[self.n]    = float(pts)
        self.seq[self.n]    = int(seq)
        self.n += 1

    def dump(self, csv_path):
        if not csv_path:
            return
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if self.header:
                w.writerow(self.header)
            for i in range(self.n):
                w.writerow((
                    float(self.t_wall[i]),
                    int(self.fid[i]),
                    float(self.pts[i]),
                    int(self.seq[i]),
                ))
