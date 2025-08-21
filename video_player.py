# -*- coding: utf-8 -*-
# gui/video_player.py
#
# 프로세스 분리 구조 (Decoder Proc ↔ Main GUI Proc)
# - 공유메모리 링버퍼 전달
# - 디코딩: 무제한(옵션 NVDEC) + (옵션) PTS 페이싱
# - 메인: YOLO(배치) + LogicDetector 추적
# - 렌더: 동기화 + drift 출력
# - 각 스테이지 EMA FPS 출력

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 중복 OpenMP 허용
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ctypes
import sys
import time as _time

# --- Ultralytics 내부 프로파일러 시간원천을 perf_counter로 교체 (로그 과다시 시간왜곡 방지)
try:
    from ultralytics.utils import ops as _yops
    def _safe_time(self):
        return _time.perf_counter()
    _yops.Profile.time = _safe_time
except Exception:
    pass

import time
import queue
import threading
import numpy as np
import multiprocessing as mp

import av
import cv2
import torch

from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QToolBar, QPushButton, QMessageBox, QOpenGLWidget, QApplication
)
from PyQt5.QtGui import QPainter, QFont, QColor, QPen
from PyQt5.QtCore import QTimer, Qt, QPoint

from OpenGL.GL import *
from OpenGL.GLUT import glutInit

# --- 선택적(존재할 때만) 중앙 로그 서버 임포트
try:
    from core.shm_csv_logger import run_shm_csv_server  # 프로젝트에 있을 때만 사용
except Exception:
    run_shm_csv_server = None  # 없으면 None으로 가드

# --- 커스텀 클래스/유틸 ---
from core.logic_detector import LogicDetector
from core.decode_proc import decoder_process
from core.types import FrameMeta
from core.metrics import LoopTimer
from core.ringbuf import SharedFrameRing
from cfg.defaults import (
    BATCH_SIZE, MAX_BATCH_DELAY_MS, YOLO_MAX_SIDE,
    TEXT_MODE, WINDOW_WIDTH, WINDOW_HEIGHT,
    SLOTS, LOCAL_Q_MAX, PRINT_EVERY, USE_HWACCEL,
    LIMIT_DECODE_SPEED, WAIT_WHEN_QUEUES_FULL, BACKPRESSURE_HWM
)
from core.batcher import InferenceBatcher
from core.tracker_manager import TrackerWorker
from core.bus import get_bus
from core.routing import get_routing
from core.async_csv_logger import close_all


# ===================== 큐 유틸 =====================
def put_latest(q, item):
    """
    WAIT_WHEN_QUEUES_FULL=True: 자리가 날 때까지 non-blocking put 재시도
    False: 가득 차면 오래된 항목 버리고 put
    """
    if WAIT_WHEN_QUEUES_FULL:
        while True:
            try:
                q.put(item, block=False)
                return
            except Exception:
                time.sleep(0.001)
    else:
        try:
            if q.full():
                q.get_nowait()
        except Exception:
            pass
        q.put(item)


# ===================== 렌더링 보조 =====================
def draw_labels_on_frame_cpu(frame_rgb: np.ndarray, detections, scale, offset_x, offset_y):
    if not detections:
        return frame_rgb
    img = frame_rgb
    for (x1, y1, x2, y2, text) in detections:
        px = offset_x + int(round(x1 * scale))
        py = offset_y + int(round(y1 * scale)) - 6
        if py < 0:
            py = 0
        cv2.putText(img, str(text), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, str(text), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ===================== GL 위젯 =====================
class GLVideoWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.detections = []
        self.w, self.h = 1280, 720
        self.tex = None
        self.pbos = None
        self.pbo_index = 0
        self.tex_w = 0
        self.tex_h = 0
        self.roi_mode = False
        self.roi_points = []
        self.draw_labels = True
        self._last_scale = 1.0
        self._last_offset_x = 0
        self._last_offset_y = 0

    def initializeGL(self):
        glClearColor(0, 0, 0, 1)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)
        self.pbos = glGenBuffers(2)

    def _recreate_gl_resources(self, w, h):
        self.tex_w, self.tex_h = int(w), int(h)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.tex_w, self.tex_h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        bytes_needed = self.tex_w * self.tex_h * 3
        for p in self.pbos:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, p)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes_needed, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self.pbo_index = 0

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()

    def update_frame(self, frame, detections):
        if frame is None:
            return
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be HxWx3 RGB uint8")
        self.frame = frame if frame.flags['C_CONTIGUOUS'] else np.ascontiguousarray(frame)
        self.h, self.w, _ = self.frame.shape
        self.detections = detections or []
        self.update()

    def _compute_layout(self):
        widget_w, widget_h = max(self.width(), 1), max(self.height(), 1)
        if self.frame is not None and (self.tex_w != self.w or self.tex_h != self.h):
            self._recreate_gl_resources(self.w, self.h)
        ratio_w = widget_w / max(self.tex_w, 1)
        ratio_h = widget_h / max(self.tex_h, 1)
        scale = min(ratio_w, ratio_h)
        video_w = int(round(self.tex_w * scale))
        video_h = int(round(self.tex_h * scale))
        offset_x = (widget_w - video_w) // 2
        offset_y = (widget_h - video_h) // 2
        draw_w = self.tex_w * scale / widget_w if self.tex_w else 0.0
        draw_h = self.tex_h * scale / widget_h if self.tex_h else 0.0
        self._last_scale = scale
        self._last_offset_x = offset_x
        self._last_offset_y = offset_y
        return draw_w, draw_h, scale, offset_x, offset_y

    def paintGL(self):
        painter = QPainter(self)
        if TEXT_MODE == "qt":
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setFont(QFont("Malgun Gothic", 14))

        painter.beginNativePainting()
        glClear(GL_COLOR_BUFFER_BIT)

        if self.frame is not None:
            draw_w, draw_h, scale, offset_x, offset_y = self._compute_layout()
            cur = self.pbo_index
            prev = 1 - cur
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbos[prev])
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glEnable(GL_TEXTURE_2D)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.tex_w, self.tex_h,
                            GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbos[cur])
            glBufferData(GL_PIXEL_UNPACK_BUFFER, self.frame.nbytes, None, GL_STREAM_DRAW)

            frame_to_upload = self.frame
            if TEXT_MODE == "cpu" and self.draw_labels and self.detections:
                bgr = cv2.cvtColor(frame_to_upload, cv2.COLOR_RGB2BGR)
                draw_labels_on_frame_cpu(bgr, self.detections, scale, offset_x, offset_y)
                frame_to_upload = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, frame_to_upload)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self.pbo_index = prev

            glColor3f(1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(-draw_w, -draw_h)
            glTexCoord2f(1, 1); glVertex2f(draw_w, -draw_h)
            glTexCoord2f(1, 0); glVertex2f(draw_w, draw_h)
            glTexCoord2f(0, 0); glVertex2f(-draw_w, draw_h)
            glEnd()

            glDisable(GL_TEXTURE_2D)
            glLineWidth(2)
            glColor3f(0.0, 1.0, 0.0)
            for (x1, y1, x2, y2, text) in self.detections:
                x1_gl = (x1 / max(self.tex_w, 1)) * 2 * draw_w - draw_w
                x2_gl = (x2 / max(self.tex_w, 1)) * 2 * draw_w - draw_w
                y1_gl = draw_h - (y1 / max(self.tex_h, 1)) * 2 * draw_h
                y2_gl = draw_h - (y2 / max(self.tex_h, 1)) * 2 * draw_h
                glBegin(GL_LINE_LOOP)
                glVertex2f(x1_gl, y1_gl); glVertex2f(x2_gl, y1_gl)
                glVertex2f(x2_gl, y2_gl); glVertex2f(x1_gl, y2_gl)
                glEnd()

            if len(self.roi_points) >= 2:
                glColor3f(1.0, 0.0, 0.0)
                glBegin(GL_LINE_LOOP)
                for pt in self.roi_points:
                    x_gl = (pt.x() / max(self.tex_w, 1)) * 2 * draw_w - draw_w
                    y_gl = draw_h - (pt.y() / max(self.tex_h, 1)) * 2 * draw_h
                    glVertex2f(x_gl, y_gl)
                glEnd()

        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glDisable(GL_BLEND)
        glFlush()
        painter.endNativePainting()

        if TEXT_MODE == "qt" and self.draw_labels and self.detections and self.frame is not None:
            scale = self._last_scale
            offset_x = self._last_offset_x
            offset_y = self._last_offset_y
            outline_pen = QPen(QColor(0, 0, 0)); outline_pen.setWidth(3)
            text_pen = QPen(QColor(255, 255, 0))
            for (x1, y1, x2, y2, text) in self.detections:
                px = offset_x + int(round(x1 * scale))
                py = offset_y + int(round(y1 * scale)) - 6
                if py < 0:
                    py = 0
                painter.setPen(outline_pen); painter.drawText(px, py, str(text))
                painter.setPen(text_pen);    painter.drawText(px, py, str(text))
        painter.end()

    def mousePressEvent(self, event):
        if not self.roi_mode:
            return
        widget_w = max(self.width(), 1)
        widget_h = max(self.height(), 1)
        scale_x = self.tex_w / widget_w
        scale_y = self.tex_h / widget_h
        if event.button() == Qt.LeftButton:
            scaled_point = QPoint(int(event.pos().x() * scale_x), int(event.pos().y() * scale_y))
            self.roi_points.append(scaled_point); self.update()
        elif event.button() == Qt.RightButton and self.roi_points:
            self.roi_points.pop(); self.update()


# ===================== 메인 플레이어 =====================
class VideoPlayerWindow(QMainWindow):
    def __init__(self, video_path):
        super().__init__()
        self.stream_id = os.path.splitext(os.path.basename(video_path))[0] or "stream0"
        self.setWindowTitle(f"YOLO PBO Detection (SharedMemory) - {video_path}")
        self.setGeometry(200, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        self.video_path = video_path

        ctx = mp.get_context("spawn")
        self.init_q: mp.Queue = ctx.Queue(maxsize=1)
        self.meta_q: mp.Queue = ctx.Queue(maxsize=SLOTS * 2)
        self.stop_event: mp.Event = ctx.Event()

        # --- 디코더 프로세스 기동 ---
        self.ring_name = f"cctv_ring_{os.getpid()}"
        self.decoder_proc = ctx.Process(
            target=decoder_process,
            args=(self.stream_id, self.video_path, self.init_q, self.meta_q,
                  self.stop_event, USE_HWACCEL, self.ring_name),
            daemon=True,
        )
        self.decoder_proc.start()

        # --- init 메타 수신 (구버전/신버전 호환)
        init = self.init_q.get()
        if isinstance(init, dict):
            # 신버전(딕셔너리):
            self.nominal_fps = float(init.get("nominal_fps", 30.0))
            self.time_base = float(init.get("time_base", 0.0)) if init.get("time_base", None) is not None else None
            self.h = int(init["h"]) ; self.w = int(init["w"])
            n_slots = int(init.get("n_slots", SLOTS))
            c = int(init.get("c", 3))
            shm_name = init.get("shm_name", self.ring_name)
        else:
            # 구버전(tuple): (W, H, SLOTS)
            try:
                W, H, n_slots = int(init[0]), int(init[1]), int(init[2])
            except Exception:
                # 방어적 처리
                W, H, n_slots = 1280, 720, SLOTS
            self.w, self.h = W, H
            self.nominal_fps = 30.0
            self.time_base = None
            c = 3
            shm_name = self.ring_name

        # --- 링버퍼 접속 ---
        self.ring = SharedFrameRing(name=shm_name, n_slots=n_slots, h=self.h, w=self.w, c=c, create=False)

        # --- 중앙 버스/라우팅 ---
        bus = get_bus()
        self.bus = bus
        self.infer_queue = bus.infer_out_q
        self.render_queue = bus.track_out_q

        # 1) 먼저 위젯을 만들고
        self.gl_widget = GLVideoWidget()
        layout = QVBoxLayout(); layout.addWidget(self.gl_widget)
        _container = QWidget(); _container.setLayout(layout)
        self.setCentralWidget(_container)

        # 2) 그런 다음에 라우팅 등록
        get_routing().register(self.stream_id, self.ring, panel=self.gl_widget)

        self.toolbar = QToolBar("Controls"); self.addToolBar(self.toolbar)
        self.roi_button = QPushButton("ROI 지정"); self.toolbar.addWidget(self.roi_button)
        self.roi_button.clicked.connect(self.enter_roi_mode)
        self.complete_button = QPushButton("ROI 완료"); self.toolbar.addWidget(self.complete_button)
        self.complete_button.clicked.connect(self.exit_roi_mode)

        # LogicDetector 시그니처 정합(필수: stream_id)
        self.detection_logic = LogicDetector(stream_id=self.stream_id)

        # 타이머들
        self.decode_timer_shadow = LoopTimer(alpha=0.8)
        self.yolo_timer = LoopTimer(alpha=0.8)
        self.track_timer = LoopTimer(alpha=0.8)
        self.render_timer = LoopTimer(alpha=0.8)

        self.decoded_count = 0
        self.render_count = 0
        self.render_times = []
        self.print_every = PRINT_EVERY

        self.render_interval_ms = max(1, int(round(1000.0 / max(self.nominal_fps, 1e-6))))
        self.timer = QTimer(); self.timer.timeout.connect(self.render_frame)

        self.play_wall_start = None
        self.play_pts_start = None
        self.last_drift = 0.0

        self.roi_mode = False

    def start_playback(self):
        # YOLO 배처 & 트래커 시작 — 시그니처 정합 (core/batcher.py 고정본 기준)
        self.batcher = InferenceBatcher(
            stream_id=self.stream_id,
            ring=self.ring,
            meta_q=self.meta_q,
            out_q=self.infer_queue,              # 파라미터명: out_q
            weights_path="yolov8n.pt",          # 필요시 교체
            classes=(0,), conf=0.4, iou=0.5,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_half=True,
            stop_event=self.stop_event,
        )
        self.tracker = TrackerWorker(
            detection_logic=self.detection_logic,
            infer_in_q=self.infer_queue,
            render_out_q=self.render_queue,
            print_every=self.print_every,
            timer=self.track_timer
        )
        self.batcher.start()
        self.tracker.start()
        self.timer.start(self.render_interval_ms)

    def enter_roi_mode(self):
        self.timer.stop()
        self.gl_widget.roi_mode = True

    def exit_roi_mode(self):
        self.gl_widget.roi_mode = False
        roi_orig = [(pt.x(), pt.y()) for pt in self.gl_widget.roi_points]

        in_w, in_h = self.w, self.h
        if in_w >= in_h:
            yolo_w = min(YOLO_MAX_SIDE, in_w)
            yolo_h = int(round(in_h * (yolo_w / in_w)))
        else:
            yolo_h = min(YOLO_MAX_SIDE, in_h)
            yolo_w = int(round(in_w * (yolo_h / in_h)))

        sx = yolo_w / max(in_w, 1)
        sy = yolo_h / max(in_h, 1)
        roi_yolo = [(int(round(x * sx)), int(round(y * sy))) for (x, y) in roi_orig]
        # LogicDetector에 ROI 지정 메서드가 있다면 사용, 없으면 JSON 로딩 방식 사용
        if hasattr(self.detection_logic, "set_roi"):
            try:
                self.detection_logic.set_roi(roi_yolo)
            except Exception:
                pass

        QMessageBox.information(self, "ROI 완료",
                                f"총 {len(self.gl_widget.roi_points)}개의 포인트가 지정되었습니다.")
        self.timer.start(self.render_interval_ms)

    def render_frame(self):
        latest = None
        # 큐에 여러 스트림이 섞일 수 있으니 내 것만 뽑자
        for _ in range(8):  # 한번에 너무 오래 잡지 않도록 소규모 폴링
            if self.render_queue.empty():
                break
            try:
                item = self.render_queue.get_nowait()
                # 예상 포맷: (sid, fid, pts, img, detections)
                if len(item) == 5:
                    sid, fid, pts, img, detections = item
                else:
                    # 구버전 호환: (fid, pts, img, detections)
                    sid, fid, pts, img, detections = self.stream_id, *item
                if sid != self.stream_id:
                    continue
                latest = (fid, pts, img, detections)
            except queue.Empty:
                break

        if latest is None:
            return

        fid, pts, img, detections = latest

        now = time.perf_counter()
        if self.play_wall_start is None:
            self.play_wall_start = now
        if self.play_pts_start is None:
            self.play_pts_start = pts if pts is not None else 0.0
        expected_wall = self.play_wall_start + ((pts - self.play_pts_start) if pts is not None else 0.0)
        self.last_drift = now - expected_wall

        self.gl_widget.update_frame(img, detections)

        _, render_ema = self.render_timer.tick()
        self.render_count += 1
        self.render_times.append(now)

        if self.render_count % self.print_every == 0 and render_ema is not None:
            y = self.yolo_timer.ema or 0.0
            t = self.track_timer.ema or 0.0
            r = render_ema
            drift_ms = self.last_drift * 1000.0
            print(f"[EMA FPS] YOLO: {y:.2f} | Track: {t:.2f} | Render: {r:.2f} | drift: {drift_ms:+.1f} ms")

    def closeEvent(self, event):
        try:
            self.timer.stop()
            self.stop_event.set()
            if self.decoder_proc.is_alive():
                self.decoder_proc.join(timeout=2.0)
        finally:
            if len(self.render_times) > 1:
                total_time = self.render_times[-1] - self.render_times[0]
                total_time = max(total_time, 1e-9)
                avg_render_fps = self.render_count / total_time
                diffs = np.diff(self.render_times)
                jitter = float(np.std(diffs)) if diffs.size > 0 else 0.0
            else:
                avg_render_fps = 0.0; jitter = 0.0
            drop_rate = (1 - (self.render_count / max(1, self.decoded_count, 1))) * 100

            print("\n===== 🎯 재생 품질 평가 (Proc + Threads) =====")
            print(f"총 디코딩 프레임(수신 기준): {self.decoded_count}")
            print(f"총 렌더링 프레임: {self.render_count}")
            print(f"평균 렌더링 FPS: {avg_render_fps:.2f}")
            print(f"프레임 드롭률: {drop_rate:.2f}%")
            print(f"프레임 타이밍 변동(Jitter): {jitter:.4f}s")
            print(f"마지막 측정 drift: {self.last_drift:+.4f}s")
            print("🛑 재생 종료\n")

            try:
                get_routing().unregister(self.stream_id)
            except Exception:
                pass
            event.accept()


# ===================== 실행 =====================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    if len(sys.argv) < 2:
        print("Usage: python video_player.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]

    # --- 중앙 UDP/SHM CSV 서버 시작 (선택적) ---
    import os as _os

    AUDIT_DIR = _os.getenv("AUDIT_DIR", "./audit")
    META_DIR = _os.path.join(AUDIT_DIR, ".shm_meta")
    _os.makedirs(AUDIT_DIR, exist_ok=True)
    _os.makedirs(META_DIR, exist_ok=True)

    log_server = None
    if run_shm_csv_server is not None:
        log_server = mp.Process(
            target=run_shm_csv_server,
            kwargs={"meta_dir": META_DIR, "scan_interval": 0.25, "flush_interval": 0.5, "batch_threshold": 512},
            daemon=False,  # ★ 반드시 False
        )
        log_server.start()
    else:
        print("[LOG] run_shm_csv_server 미탑재: 중앙 로그 서버 실행은 생략합니다.")

    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        glutInit(sys.argv)
        torch.backends.cudnn.benchmark = True

        app = QApplication(sys.argv)
        win = VideoPlayerWindow(video_path)
        win.show()
        win.start_playback()
        sys.exit(app.exec_())
    finally:
        if log_server is not None and log_server.is_alive():
            log_server.terminate()
            log_server.join(2)

        close_all()
