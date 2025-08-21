# -*- coding: utf-8 -*-
# core/tracker_manager.py — dual-compatible (InferOut & legacy tuple)
#
# InferenceBatcher → (1) InferOut 객체 또는 (2) 레거시 튜플 입력을 모두 처리.
# LogicDetector 최신( process(InferOut)->TrackOut ) 및 과거( detect_and_track(...) ) 인터페이스를 모두 지원.
# 렌더 큐에는 (sid, fid, pts, img_rgb, tags) 형태로 푸시.

import os
import time
import queue
from threading import Thread

import numpy as np

from core.shm_csv_logger import ShmCsvProducer
from core.types import InferOut
from core.routing import get_routing


class TrackerWorker(Thread):
    def __init__(self, detection_logic, infer_in_q, render_out_q,
                 print_every: int = 120, timer=None, stop_event=None):
        super().__init__(daemon=True)
        self.logic = detection_logic
        self.in_q = infer_in_q
        self.out_q = render_out_q
        self.print_every = int(print_every)
        self.timer = timer
        self.stop_event = stop_event
        self._stop_flag = False

        # 스트림별 SHM 로거 캐시 (sid -> logger)
        self._logs = {}
        self._audit_dir = os.environ.get("AUDIT_DIR", "./audit")
        os.makedirs(self._audit_dir, exist_ok=True)

        self._processed = 0
        self._t0 = time.time()

    # -------------------------
    # SHM CSV Logger (per stream)
    # -------------------------
    def _get_logger(self, sid: str) -> ShmCsvProducer:
        lg = self._logs.get(sid)
        if lg is None:
            ring_name = f"shmlog_track_{sid}"
            lg = ShmCsvProducer(
                name=ring_name,
                csv_path=None,  # 런타임 디스크 I/O OFF (별도 서버에서 flush 가능)
                columns=["t_wall", "fid", "pts", "seq"],
                meta_dir=os.path.join(self._audit_dir, ".shm_meta"),
                nslots=1024,
                batch=128,
                max_delay_sec=0.05,
                overwrite_when_full=False,
                drop_on_full=False,
                capacity=262144,
            )
            self._logs[sid] = lg
        return lg

    def stop(self):
        self._stop_flag = True

    # -------------------------
    # 내부 유틸
    # -------------------------
    @staticmethod
    def _bgr_to_rgb(img):
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 3:
            return img[:, :, ::-1].copy()
        return img

    def _push_render(self, sid: str, fid: int, pts: float, img_rgb, tags):
        """렌더 큐가 가득 차도 절대 드롭하지 않고, 자리가 날 때까지 대기한다.
        stop_event 또는 stop_flag가 걸리면 즉시 중단."""
        while True:
            if self._stop_flag:
                return
            if self.stop_event is not None and self.stop_event.is_set():
                return
            try:
                # block-put with finite timeout to allow stop checks
                self.out_q.put((sid, fid, pts, img_rgb, tags), block=True, timeout=0.5)
                return
            except queue.Full:
                # 대기 후 재시도 (절대 드롭 금지)
                time.sleep(0.005)

    def _log_row(self, sid: str, fid: int, pts: float):
        t_wall = time.perf_counter()
        try:
            self._get_logger(sid).write((t_wall, int(fid), float(pts if pts is not None else 0.0), int(fid)))
        except Exception:
            pass

    # -------------------------
    # 메인 루프
    # -------------------------
    def run(self):
        try:
            while True:
                if self._stop_flag:
                    break
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                try:
                    item = self.in_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                # === 경로 1) 최신 파이프라인: InferOut 객체 ===
                if isinstance(item, InferOut):
                    infer_out: InferOut = item
                    sid = infer_out.stream_id
                    fid = int(infer_out.fid)
                    pts = float(infer_out.pts if infer_out.pts is not None else 0.0)

                    # LogicDetector 최신 인터페이스
                    try:
                        track_out = self.logic.process(infer_out)
                        tags = track_out.tags if track_out is not None else []
                    except Exception as e:
                        print(f"[TRACK] process() 예외: {e}")
                        tags = []

                    # 렌더용 원본 프레임: routing에서 ring 찾아 읽기 (BGR→RGB)
                    img_rgb = None
                    try:
                        ring = get_routing().get_ring(sid)
                        if ring is not None:
                            img_bgr = ring.read_copy(fid % ring.n_slots)
                            img_rgb = self._bgr_to_rgb(img_bgr)
                    except Exception:
                        pass

                    self._push_render(sid, fid, pts, img_rgb, tags)
                    self._log_row(sid, fid, pts)

                # === 경로 2) 레거시 파이프라인: 튜플 언패킹 ===
                else:
                    # 예상 포맷: (sid, fid, pts, img_src[BGR], img_yolo[BGR], (sx,sy), detections, confs)
                    # 일부 환경에서 (sid,fid,pts,img,det) 등 변형이 올 수 있으니 길이로 분기
                    try:
                        if isinstance(item, (list, tuple)):
                            if len(item) == 8:
                                sid, fid, pts, img_src, img_yolo, scale_xy, detections, confs = item
                                sx, sy = scale_xy
                                try:
                                    # 과거 인터페이스: detect_and_track
                                    id_bboxes = self.logic.detect_and_track(detections, frame=img_yolo, confidences=confs)
                                except Exception as e:
                                    print(f"[TRACK] detect_and_track 예외: {e}")
                                    id_bboxes = []

                                # 원본 해상도로 태그 좌표 환산
                                tags = []
                                for (x1, y1, x2, y2, text) in id_bboxes:
                                    tags.append((
                                        int(round(x1 * sx)), int(round(y1 * sy)),
                                        int(round(x2 * sx)), int(round(y2 * sy)),
                                        text,
                                    ))

                                img_rgb = self._bgr_to_rgb(img_src)
                                self._push_render(sid, int(fid), float(pts or 0.0), img_rgb, tags)
                                self._log_row(sid, int(fid), float(pts or 0.0))

                            elif len(item) == 5:
                                # 이미 렌더 포맷이면 그대로 패스스루
                                sid, fid, pts, img_rgb, tags = item
                                self._push_render(sid, int(fid), float(pts or 0.0), img_rgb, tags)
                                self._log_row(sid, int(fid), float(pts or 0.0))
                            else:
                                # 알 수 없는 포맷은 스킵
                                continue
                        else:
                            continue
                    except Exception as e:
                        print(f"[ERROR][TRACK] {e}")
                        time.sleep(0.01)

                # 진행 통계
                self._processed += 1
                if self.timer is not None:
                    try:
                        self.timer.tick()
                    except Exception:
                        pass
                if self._processed % self.print_every == 0:
                    dt = time.time() - self._t0
                    fps = self._processed / max(dt, 1e-6)
                    print(f"[TRACK] processed={self._processed}, fps≈{fps:.1f}")

        finally:
            # 로거 정리
            for sid, lg in list(self._logs.items()):
                try:
                    lg.close()
                except Exception:
                    pass
            self._logs.clear()
            # LogicDetector 종료 훅
            try:
                if hasattr(self.logic, "close"):
                    self.logic.close()
            except Exception:
                pass
