# -*- coding: utf-8 -*-
# core/batcher.py
# YOLO 배치 추론 스레드: InMemoryCsvLoggerNP 로깅 + 지연 한계 기반 배치 + 백프레셔 안전

import os
import time
import queue
from threading import Thread
from typing import List, Tuple

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from core.types import FrameMeta, InferOut
from cfg.defaults import BATCH_SIZE, MAX_BATCH_DELAY_MS, YOLO_MAX_SIDE, PRINT_EVERY
from utils.inmem_logger import InMemoryCsvLoggerNP


class InferenceBatcher(Thread):
    """
    meta_q(FrameMeta) → SharedFrameRing 읽어 배치 구성 → YOLO 추론 → InferOut을 out_q로 전달.
    각 결과 프레임에 대해 (t_wall, fid, pts, seq=fid) 로깅을 수행하고 종료 시 CSV로 저장.
    """

    def __init__(
        self,
        stream_id: str,
        ring,
        meta_q,
        out_q,
        weights_path: str = "yolov8n.pt",
        classes: Tuple[int, ...] = (0,),
        conf: float = 0.4,
        iou: float = 0.5,
        device: str = None,
        use_half: bool = True,
        stop_event=None,
    ):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.ring = ring
        self.meta_q = meta_q
        self.out_q = out_q
        self.stop_event = stop_event

        self.classes = tuple(classes)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = bool(use_half and ("cuda" in str(self.device)))

        # --- YOLO 모델 로드 ---
        self.model = YOLO(weights_path)
        try:
            self.model.to(self.device)
            if self.use_half:
                # 일부 백엔드에서만 유효
                try:
                    self.model.model.half()  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

        # --- 로거 ---
        audit_dir = os.getenv("AUDIT_DIR", "./audit")
        os.makedirs(audit_dir, exist_ok=True)
        self._csv_path = os.path.join(audit_dir, f"{stream_id}_yolo.csv")
        self._log = InMemoryCsvLoggerNP(header=["t_wall", "fid", "pts", "seq"], initial_capacity=200_000)

        # --- 상태 ---
        self._processed = 0
        self._t0 = time.time()

    # ------------------------
    # 전처리: 한 변 상한 유지 리사이즈 (aspect 유지)
    # ------------------------
    @staticmethod
    def _resize_keep_max_side(img, max_side: int):
        h, w = img.shape[:2]
        scale = min(max_side / float(max(h, w)), 1.0)
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img

    def _pop_until_batch(self, batch_imgs: List[np.ndarray], batch_meta: List[FrameMeta]):
        """MAX_BATCH_DELAY_MS 지연 한계 내에서 배치를 채움."""
        start = time.time()
        while (len(batch_imgs) < BATCH_SIZE) and ((time.time() - start) * 1000.0 < MAX_BATCH_DELAY_MS):
            if self.stop_event is not None and self.stop_event.is_set():
                break
            try:
                m = self.meta_q.get(timeout=0.003)
                if not isinstance(m, FrameMeta):
                    continue
                img = self.ring.read_copy(m.slot)
                img = self._resize_keep_max_side(img, YOLO_MAX_SIDE)
                batch_imgs.append(img)
                batch_meta.append(m)
            except queue.Empty:
                continue

    def _do_infer(self, imgs: List[np.ndarray]):
        # 울트라리틱스는 list[np.ndarray] 입력을 지원.
        # classes/conf/iou 적용. verbose=False로 오버헤드 최소화.
        results = self.model(
            imgs,
            device=self.device,
            classes=list(self.classes) if self.classes else None,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        # 필요 시 동기화(디버그/정밀 측정용)
        if "cuda" in str(self.device) and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        return results

    def run(self):
        batch_imgs: List[np.ndarray] = []
        batch_meta: List[FrameMeta] = []

        try:
            # 간단한 워밍업(선택)
            try:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                _ = self._do_infer([dummy])
            except Exception:
                pass

            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                # 먼저 1개는 블록킹으로 확보해 배치 시동
                try:
                    m0 = self.meta_q.get(timeout=0.05)
                    if not isinstance(m0, FrameMeta):
                        continue
                    img0 = self.ring.read_copy(m0.slot)
                    img0 = self._resize_keep_max_side(img0, YOLO_MAX_SIDE)
                    batch_imgs = [img0]
                    batch_meta = [m0]
                except queue.Empty:
                    # 입력이 잠시 없는 경우
                    continue

                # 지연 한계 내 추가 수집
                self._pop_until_batch(batch_imgs, batch_meta)

                # 추론
                try:
                    results = self._do_infer(batch_imgs)
                except Exception as e:
                    print(f"[ERROR][YOLO:{self.stream_id}] infer failed: {e}")
                    continue

                # 결과 파싱 및 전송/로깅
                for m, r in zip(batch_meta, results):
                    det = getattr(r, "boxes", None)
                    if det is None:
                        det_xyxy = np.zeros((0, 4), dtype=np.float32)
                        det_conf = np.zeros((0,), dtype=np.float32)
                        det_cls = np.zeros((0,), dtype=np.float32)
                    else:
                        # tensor → numpy (안전 변환)
                        def _to_np(x, shape=None, dtype=np.float32):
                            try:
                                arr = x.detach().cpu().numpy()
                            except Exception:
                                arr = np.asarray(x)
                            if dtype is not None:
                                arr = arr.astype(dtype, copy=False)
                            if shape is not None and arr.size == 0:
                                return np.zeros(shape, dtype=dtype)
                            return arr

                        det_xyxy = _to_np(getattr(det, "xyxy", None), shape=(0, 4)) if hasattr(det, "xyxy") else np.zeros((0, 4), dtype=np.float32)
                        det_conf = _to_np(getattr(det, "conf", None), shape=(0,)) if hasattr(det, "conf") else np.zeros((0,), dtype=np.float32)
                        det_cls  = _to_np(getattr(det, "cls", None),  shape=(0,)) if hasattr(det, "cls")  else np.zeros((0,), dtype=np.float32)

                    # 결과 전달 (백프레셔: 기본 block=True)
                    self.out_q.put(InferOut(
                        stream_id=m.stream_id,
                        fid=m.fid,
                        pts=m.pts,
                        det_xyxy=det_xyxy,
                        det_conf=det_conf,
                        det_cls=det_cls,
                    ), block=True)

                    # 로깅
                    self._log.row((time.time(), m.fid, m.pts, m.fid))
                    self._processed += 1

                # 진행 로그
                if self._processed and (self._processed % PRINT_EVERY == 0):
                    dt = max(1e-6, time.time() - self._t0)
                    fps = self._processed / dt
                    print(f"[YOLO] batch={BATCH_SIZE} processed={self._processed} fps≈{fps:.1f}")

        except Exception as e:
            print(f"[ERROR][BATCH:{self.stream_id}] {e}")
        finally:
            try:
                self._log.dump(self._csv_path)
                print(f"[YOLO] audit saved → {self._csv_path}")
            except Exception as e:
                print(f"[YOLO] audit dump failed: {e}")
