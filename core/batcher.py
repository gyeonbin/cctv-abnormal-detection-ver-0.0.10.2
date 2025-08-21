# -*- coding: utf-8 -*-
# core/batcher.py
#
# - meta_q에서 FrameMeta를 읽어 SHM 링버퍼에서 프레임을 취득
# - YOLOv8 추론 실행 (predict)
# - 결과를 InferOut 객체로 패키징하여 infer_out_q로 전송
#
# 리팩터링 핵심: 튜플 대신 InferOut 객체를 사용해 파이프라인 일관성 확보

import os
import time
import queue
from threading import Thread

import numpy as np
import torch
from ultralytics import YOLO

from core.types import FrameMeta, InferOut
from core.routing import get_routing
from utils.inmem_logger import InMemoryCsvLoggerNP


class InferenceBatcher(Thread):
    def __init__(self, in_q, out_q, model_path: str, timer=None, stop_event=None):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.model_path = model_path
        self.timer = timer
        self._stop_flag = False

        # --- YOLO 모델 초기화 ---
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"Ultralytics YOLOv8.1.25 🚀 Python-{torch.__version__} {self.device}")
            # 첫 추론은 시간이 걸리므로 워밍업
            self.model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        except Exception as e:
            print(f"[YOLO] 모델 로딩 실패: {e}")
            self.model = None

        # --- 인메모리 로거 ---
        self._stream_id = None  # 첫 프레임에서 확인
        self._log = None

        self._processed = 0
        self._t0 = time.time()

    def _get_logger(self, sid: str) -> InMemoryCsvLoggerNP:
        if self._log is None:
            self._stream_id = sid
            audit_dir = os.getenv("AUDIT_DIR", "./audit")
            os.makedirs(audit_dir, exist_ok=True)
            csv_path = os.path.join(audit_dir, f"{sid}_yolo.csv")
            self._log = InMemoryCsvLoggerNP(
                header=["t_wall", "fid", "pts", "n_dets"],
                initial_capacity=200_000,
                dump_path=csv_path
            )
        return self._log

    def stop(self):
        self._stop_flag = True

    def run(self):
        if self.model is None:
            return

        try:
            while not (self._stop_flag or (self.stop_event and self.stop_event.is_set())):
                try:
                    meta: FrameMeta = self.in_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                sid, slot, fid, pts = meta

                # --- 공유 메모리에서 이미지 읽기 ---
                img_bgr = None
                try:
                    ring = get_routing().get_ring(sid)
                    if ring:
                        img_bgr = ring.read_copy(slot)
                except Exception as e:
                    print(f"[BATCHER] 링버퍼 읽기 실패 (fid={fid}): {e}")
                    continue

                if img_bgr is None:
                    continue

                # --- YOLO 추론 ---
                try:
                    # verbose=False로 설정하여 콘솔 로그 최소화
                    results = self.model.predict(img_bgr, classes=[0], verbose=False)
                    res = results[0]  # 첫 번째 결과 사용

                    # numpy로 변환
                    det_xyxy = res.boxes.xyxy.cpu().numpy()
                    det_conf = res.boxes.conf.cpu().numpy()
                    det_cls = res.boxes.cls.cpu().numpy()
                    n_dets = len(det_xyxy)
                except Exception as e:
                    print(f"[YOLO] 추론 실패 (fid={fid}): {e}")
                    det_xyxy = np.empty((0, 4), dtype=np.float32)
                    det_conf = np.empty((0,), dtype=np.float32)
                    det_cls = np.empty((0,), dtype=np.float32)
                    n_dets = 0

                # --- InferOut 객체 생성 및 전송 ---
                infer_out = InferOut(
                    stream_id=sid,
                    fid=fid,
                    pts=pts,
                    det_xyxy=det_xyxy,
                    det_conf=det_conf,
                    det_cls=det_cls,
                )
                self.out_q.put(infer_out)

                # --- 로깅 및 통계 ---
                if self.timer:
                    self.timer.tick()

                logger = self._get_logger(sid)
                logger.row((time.time(), fid, pts, n_dets))

                self._processed += 1
                if self._processed % 120 == 0:
                    dt = time.time() - self._t0
                    fps = self._processed / max(dt, 1e-6)
                    # print(f"[YOLO] processed={self._processed}, fps≈{fps:.1f}")

        finally:
            # --- 종료 시 로그 저장 ---
            if self._log:
                try:
                    self._log.dump()
                    print(f"[YOLO] audit saved → {self._log.dump_path}")
                except Exception as e:
                    print(f"[YOLO] audit dump failed: {e}")