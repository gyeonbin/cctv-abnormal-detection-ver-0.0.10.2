# -*- coding: utf-8 -*-
# core/logic_detector.py
# 기능: YOLO 추론 결과로 트래킹 수행 + ROI 기반 배회(Loitering)/침입(Intrusion) 판단
# 호환: InferOut 객체/튜플 입력 모두 처리 (process / process_any)
# 로깅: 프레임 단위(track.csv) + 이벤트 단위(events.csv)

import os
import time
import json
from typing import List, Tuple, Optional, Dict, Iterator

import numpy as np

from core.types import InferOut, TrackOut
from core.person_traker import PersonTracker  # 기존 경로 유지
from utils.inmem_logger import InMemoryCsvLoggerNP


# -------------------------------
# 유틸: 다각형 내부 판정 (Ray casting)
# -------------------------------
def point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if cond:
            inside = not inside
    return inside


class ROIRegion:
    def __init__(self, roi_id: str, name: str, polygon: List[Tuple[float, float]], kind: str, dwell_sec: float = 30.0):
        self.id = str(roi_id)
        self.name = str(name)
        self.poly = [(float(x), float(y)) for x, y in polygon]
        self.kind = kind  # 'intrusion' or 'loiter'
        self.dwell_sec = float(dwell_sec)

    def contains(self, x: float, y: float) -> bool:
        return point_in_poly(x, y, self.poly)


# -------------------------------
# LogicDetector 본체
# -------------------------------
class LogicDetector:
    """
    ROI 기반 배회/침입 판단
    - intrusion_rois: 진입 즉시 이벤트
    - loiter_rois: ROI 내부 연속 체류시간이 dwell_sec 이상이면 이벤트
    - 동일 track id/t 동일 ROI에 대해 중복 발생하지 않도록 1회만 발생(재-이탈 후 재-진입 시 재발생)
    """

    def __init__(
        self,
        stream_id: Optional[str] = None,
        intrusion_rois: Optional[List[ROIRegion]] = None,
        loiter_rois: Optional[List[ROIRegion]] = None,
        default_loiter_sec: float = 50.0,
        movement_threshold: int = 30,
        roi_json_path: Optional[str] = None,
    ):
        self.stream_id = stream_id
        self.tracker = PersonTracker(feature_extractor=None)
        self.movement_threshold = int(movement_threshold)

        # ---- ROI 로드 ----
        self.intrusion_rois: List[ROIRegion] = []
        self.loiter_rois: List[ROIRegion] = []
        roi_json_path = roi_json_path or os.getenv("ROI_JSON", "./cfg/rois.json")
        if (intrusion_rois is None and loiter_rois is None) and os.path.isfile(roi_json_path):
            try:
                with open(roi_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for r in data.get("intrusion", []):
                    self.intrusion_rois.append(
                        ROIRegion(
                            roi_id=r.get("id", r.get("name", "intrusion")),
                            name=r.get("name", "intrusion"),
                            polygon=r["polygon"],
                            kind="intrusion",
                        )
                    )
                for r in data.get("loiter", []):
                    self.loiter_rois.append(
                        ROIRegion(
                            roi_id=r.get("id", r.get("name", "loiter")),
                            name=r.get("name", "loiter"),
                            polygon=r["polygon"],
                            kind="loiter",
                            dwell_sec=float(r.get("dwell_sec", default_loiter_sec)),
                        )
                    )
            except Exception as e:
                print(f"[ROI] JSON 로드 실패({roi_json_path}): {e}")
        else:
            self.intrusion_rois = intrusion_rois or []
            self.loiter_rois = loiter_rois or []

        # ---- ReID 임베더(가능 시) ----
        try:
            from reid_embedder_onnx import build_embedder_onnx
            self.tracker.feature_extractor = build_embedder_onnx(auto_download=True, use_gpu=True)
            print("[ReID] ONNX(OSNet x0.25) 임베더 활성화")
        except Exception as e:
            print(f"[ReID] ONNX 임베더 초기화 실패: {e}")

        # ---- 인메모리 로거 ----
        audit_dir = os.getenv("AUDIT_DIR", "./audit")
        os.makedirs(audit_dir, exist_ok=True)
        self._trk_csv = os.path.join(audit_dir, f"{(self.stream_id or 'stream')}_track.csv")
        self._evt_csv = os.path.join(audit_dir, f"{(self.stream_id or 'stream')}_events.csv")
        self._log_trk = InMemoryCsvLoggerNP(header=["t_wall", "fid", "pts", "seq"], initial_capacity=240000)
        self._log_evt = InMemoryCsvLoggerNP(header=["t_wall", "fid", "pts", "tid", "event", "roi_id"], initial_capacity=60000)

        # ---- per-ID 상태 ----
        self._last_pts: Dict[int, float] = {}
        self._loiter_enter: Dict[int, Dict[str, float]] = {}
        self._loiter_fired: Dict[int, set] = {}
        self._intrusion_fired: Dict[int, set] = {}

    # -------------------------------
    # 외부 API
    # -------------------------------
    def set_roi(self, polygon_points: List[Tuple[int, int]], dwell_sec: Optional[float] = None):
        """
        단일 loiter ROI를 지정(에디터/GUI용). 기존 loiter 목록을 대체.
        """
        dwell = float(dwell_sec) if dwell_sec is not None else 50.0
        self.loiter_rois = [ROIRegion("roi0", "ROI-0", polygon_points, "loiter", dwell_sec=dwell)]

    # 메인 입력: InferOut 객체
    def process(self, infer_out: InferOut) -> TrackOut:
        det_xyxy = infer_out.det_xyxy.astype(np.float32)
        det_conf = infer_out.det_conf.astype(np.float32)
        det_cls  = infer_out.det_cls.astype(np.float32)
        tracks = self.tracker.update(det_xyxy, det_conf, det_cls)

        tags = []
        curr_pts = float(infer_out.pts if infer_out.pts is not None else 0.0)

        # 트래킹 로그(프레임 단위)
        self._log_trk.row((time.time(), infer_out.fid, curr_pts, infer_out.fid))

        for t in tracks:
            tid = int(t.tid)
            x1, y1, x2, y2 = float(t.x1), float(t.y1), float(t.x2), float(t.y2)
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # intrusion: 진입 순간 1회
            fired_set = self._intrusion_fired.setdefault(tid, set())
            for roi in self.intrusion_rois:
                if roi.contains(cx, cy) and roi.id not in fired_set:
                    self._emit_event(infer_out.fid, curr_pts, tid, "intrusion", roi.id)
                    fired_set.add(roi.id)

            # loiter: dwell_sec 경과 1회
            enter_map = self._loiter_enter.setdefault(tid, {})
            fired_loiter = self._loiter_fired.setdefault(tid, set())
            for roi in self.loiter_rois:
                if roi.contains(cx, cy):
                    if roi.id not in enter_map:
                        enter_map[roi.id] = curr_pts
                    else:
                        dwell = curr_pts - enter_map[roi.id]
                        if (dwell >= roi.dwell_sec) and (roi.id not in fired_loiter):
                            self._emit_event(infer_out.fid, curr_pts, tid, "loiter", roi.id)
                            fired_loiter.add(roi.id)
                else:
                    if roi.id in enter_map:
                        del enter_map[roi.id]
                    if roi.id in fired_loiter:
                        fired_loiter.remove(roi.id)

            # 라벨 구성
            label_bits = [f"id:{tid}"]
            if any(r.contains(cx, cy) for r in self.intrusion_rois):
                label_bits.append("INTRUSION")
            if any((roi.contains(cx, cy) and roi.id in self._loiter_fired.get(tid, set())) for roi in self.loiter_rois):
                label_bits.append("LOITER")
            label = "|".join(label_bits)
            tags.append((int(x1), int(y1), int(x2), int(y2), label))

            self._last_pts[tid] = curr_pts

        return TrackOut(
            stream_id=infer_out.stream_id,
            fid=infer_out.fid,
            pts=curr_pts,
            tags=tags,
        )

    # 호환 입력: InferOut 또는 (sid, fid, pts, det_xyxy, det_conf, det_cls)
    def process_any(self, obj) -> TrackOut:
        if isinstance(obj, InferOut):
            return self.process(obj)
        # 튜플/리스트 언패킹 시도
        if isinstance(obj, (tuple, list)) and len(obj) >= 6:
            sid, fid, pts, det_xyxy, det_conf, det_cls = obj[:6]
            # stream_id가 비어 있으면 보완
            sid = sid if sid is not None else (self.stream_id or "stream")
            pts = 0.0 if pts is None else float(pts)
            infer = InferOut(
                stream_id=sid,
                fid=int(fid),
                pts=pts,
                det_xyxy=np.asarray(det_xyxy, dtype=np.float32),
                det_conf=np.asarray(det_conf, dtype=np.float32),
                det_cls=np.asarray(det_cls, dtype=np.float32),
            )
            return self.process(infer)
        raise TypeError("Unsupported input for process_any: expected InferOut or (sid,fid,pts,det_xyxy,det_conf,det_cls) tuple")

    # -------------------------------
    # 내부
    # -------------------------------
    def _emit_event(self, fid: int, pts: float, tid: int, event: str, roi_id: str):
        self._log_evt.row((time.time(), fid, pts, tid, event, roi_id))

    def close(self):
        try:
            self._log_trk.dump(self._trk_csv)
            self._log_evt.dump(self._evt_csv)
            print(f"[TRACK] audit saved → {self._trk_csv}")
            print(f"[EVENT] audit saved → {self._evt_csv}")
        except Exception as e:
            print(f"[TRACK/EVENT] audit dump failed: {e}")
