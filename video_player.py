# video_player.py

import os
import sys
import time
import argparse
from multiprocessing import Process, Event, set_start_method
from threading import Thread
from pathlib import Path
import queue
from typing import Optional, List, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLineEdit, QFileDialog, QPlainTextEdit, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import QTimer, Qt, QPoint

# -----------------
# 프로젝트 경로 설정
# -----------------
try:
    BASE_DIR = Path(sys._MEIPASS)
except Exception:
    BASE_DIR = Path(__file__).resolve().parent

# video_player.py가 루트에 있으므로, 현재 폴더를 경로에 추가
sys.path.append(str(BASE_DIR))

# -----------------
# 코어 모듈 임포트
# -----------------
from core.bus import get_bus
from core.decode_proc import decoder_process
from core.batcher import InferenceBatcher
from core.logic_detector import LogicDetector
from core.tracker_manager import TrackerWorker
from core.metrics import LoopTimer
from core.routing import get_routing, register_ring

# -----------------
# 상수 및 기본값
# -----------------
DEFAULT_MODEL = "yolov8n.pt"
ROI_DWELL_SEC = 30.0


# -------------------------------------------------
# ROI 편집 기능이 포함된 QGraphicsView
# -------------------------------------------------
class ROIEditorView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.roi_points: List[QPoint] = []
        self.is_drawing = False
        self._last_img = None
        self.tags_to_draw: List[Tuple] = []

    def set_image(self, q_img: QImage):
        if q_img is None or q_img.isNull():
            if self._last_img is not None:
                pixmap = QPixmap.fromImage(self._last_img)
                self.pixmap_item.setPixmap(pixmap)
            return

        pixmap = QPixmap.fromImage(q_img)
        self.pixmap_item.setPixmap(pixmap)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._last_img = q_img

    def mousePressEvent(self, event):
        if not self.is_drawing:
            return
        pos_in_view = event.pos()
        pos_in_scene = self.mapToScene(pos_in_view)
        self.roi_points.append(pos_in_scene.toPoint())
        self.update()

    def get_roi_points_in_image_coords(self) -> List[Tuple[int, int]]:
        if not self.pixmap_item.pixmap(): return []
        img_w = self.pixmap_item.pixmap().width()
        img_h = self.pixmap_item.pixmap().height()
        if img_w == 0 or img_h == 0: return []

        scene_rect = self.scene.sceneRect()
        scene_w, scene_h = scene_rect.width(), scene_rect.height()
        if scene_w == 0 or scene_h == 0: return []

        scaled_points = []
        for p in self.roi_points:
            x_ratio = p.x() / scene_w
            y_ratio = p.y() / scene_h
            scaled_points.append((int(x_ratio * img_w), int(y_ratio * img_h)))
        return scaled_points

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.Antialiasing)

        if self.is_drawing and self.roi_points:
            painter.setPen(QPen(QColor(255, 0, 0, 200), 2, Qt.SolidLine))
            poly_points = [self.mapFromScene(p) for p in self.roi_points]
            for i in range(len(poly_points)):
                p1, p2 = poly_points[i], poly_points[(i + 1) % len(poly_points)]
                painter.drawLine(p1, p2)
                painter.drawEllipse(p1, 4, 4)

        if not self.pixmap_item.pixmap() or self.pixmap_item.pixmap().isNull(): return

        img_w, img_h = self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()
        view_w, view_h = self.viewport().width(), self.viewport().height()
        if img_w == 0 or img_h == 0: return

        scale = min(view_w / img_w, view_h / img_h)
        offset_x, offset_y = (view_w - img_w * scale) / 2, (view_h - img_h * scale) / 2

        painter.setPen(QPen(Qt.green, 1))
        painter.setBrush(Qt.NoBrush)
        for x1, y1, x2, y2, label in self.tags_to_draw:
            vx1, vy1 = x1 * scale + offset_x, y1 * scale + offset_y
            vx2, vy2 = x2 * scale + offset_x, y2 * scale + offset_y
            painter.drawRect(int(vx1), int(vy1), int(vx2 - vx1), int(vy2 - vy1))
            painter.drawText(int(vx1), int(vy1) - 5, label)


# -------------------------------------------------
# 메인 위젯
# -------------------------------------------------
class VideoPlayerWindow(QWidget):
    def __init__(self, video_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.stream_id = "stream0"

        self.decoder_proc: Optional[Process] = None
        self.batcher: Optional[InferenceBatcher] = None
        self.tracker: Optional[TrackerWorker] = None
        self.logic: Optional[LogicDetector] = None
        self.stop_event = Event()

        self.timers = {
            "yolo": LoopTimer(alpha=0.05), "track": LoopTimer(alpha=0.05),
            "render": LoopTimer(alpha=0.05)
        }
        self.last_print_time = 0

        self.init_ui()

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self.update_frame)
        self.render_timer.start(33)

        if self.video_path and os.path.isfile(self.video_path):
            self.video_path_input.setText(self.video_path)
            # QTimer를 사용하여 UI가 완전히 로드된 후 재생을 시작
            QTimer.singleShot(100, self.start_playback)

    def init_ui(self):
        layout = QGridLayout(self)
        self.video_view = ROIEditorView()
        layout.addWidget(self.video_view, 0, 0, 1, 4)

        self.video_path_input = QLineEdit()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_video)
        layout.addWidget(QLabel("Video Path:"), 1, 0)
        layout.addWidget(self.video_path_input, 1, 1, 1, 2)
        layout.addWidget(self.browse_btn, 1, 3)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_playback)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_all)
        layout.addWidget(self.start_btn, 2, 0)
        layout.addWidget(self.stop_btn, 2, 1)

        self.roi_draw_btn = QPushButton("Draw ROI")
        self.roi_draw_btn.clicked.connect(self.toggle_roi_drawing)
        self.roi_set_btn = QPushButton("Set ROI")
        self.roi_set_btn.clicked.connect(self.set_roi)
        self.roi_clear_btn = QPushButton("Clear ROI")
        self.roi_clear_btn.clicked.connect(self.clear_roi)
        layout.addWidget(self.roi_draw_btn, 3, 0)
        layout.addWidget(self.roi_set_btn, 3, 1)
        layout.addWidget(self.roi_clear_btn, 3, 2)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 4, 0, 1, 4)

    def browse_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Video files (*.mp4 *.avi *.mkv)")
        if fname: self.video_path_input.setText(fname)

    def toggle_roi_drawing(self):
        self.video_view.is_drawing = not self.video_view.is_drawing
        self.roi_draw_btn.setText("Finish Drawing" if self.video_view.is_drawing else "Draw ROI")
        if self.video_view.is_drawing: self.video_view.roi_points.clear()

    def set_roi(self):
        if not self.logic:
            QMessageBox.warning(self, "Warning", "Pipeline is not running.")
            return
        points = self.video_view.get_roi_points_in_image_coords()
        if len(points) < 3:
            QMessageBox.warning(self, "Warning", "Please draw a polygon with at least 3 points.")
            return
        self.logic.set_roi(points, dwell_sec=ROI_DWELL_SEC)
        self.log_message(f"Loitering ROI set with {len(points)} points.")
        self.video_view.is_drawing = False
        self.roi_draw_btn.setText("Draw ROI")

    def clear_roi(self):
        self.video_view.roi_points.clear()
        self.video_view.update()
        if self.logic: self.logic.set_roi([], dwell_sec=0)
        self.log_message("ROI cleared.")

    def log_message(self, msg: str):
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def start_playback(self):
        video_path = self.video_path_input.text()
        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "Error", "Video file not found.")
            return

        self.stop_all()
        self.log_message(f"Starting pipeline for {video_path}")
        self.video_path, self.stream_id = video_path, Path(video_path).stem
        self.stop_event.clear()

        bus, ring_name = get_bus(), f"shm_ring_{self.stream_id}"
        register_ring(self.stream_id, ring_name)

        self.decoder_proc = Process(
            target=decoder_process,
            args=(self.stream_id, self.video_path, None, bus.meta_q, self.stop_event, True, ring_name),
            daemon=True
        )
        self.decoder_proc.start()

        model_path = os.path.join(BASE_DIR, DEFAULT_MODEL)
        if not os.path.isfile(model_path): model_path = os.path.join(BASE_DIR, 'gui', DEFAULT_MODEL)
        if not os.path.isfile(model_path):
            QMessageBox.critical(self, "Error", f"YOLO model not found at {model_path}")
            self.stop_all()
            return

        self.batcher = InferenceBatcher(
            in_q=bus.meta_q, out_q=bus.infer_out_q, model_path=model_path,
            timer=self.timers["yolo"], stop_event=self.stop_event
        )
        self.batcher.start()

        self.logic = LogicDetector(stream_id=self.stream_id, roi_json_path=None)
        self.tracker = TrackerWorker(
            detection_logic=self.logic, infer_in_q=bus.infer_out_q,
            render_out_q=bus.track_out_q, timer=self.timers["track"], stop_event=self.stop_event
        )
        self.tracker.start()
        self.log_message("All pipeline components started.")

    def update_frame(self):
        try:
            sid, fid, pts, img_rgb, tags = get_bus().track_out_q.get_nowait()
        except queue.Empty:
            return

        if img_rgb is not None:
            h, w, ch = img_rgb.shape
            q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_view.tags_to_draw = tags
            self.video_view.set_image(q_img)

        self.timers["render"].tick()
        self.print_stats()

    def print_stats(self):
        now = time.time()
        if now - self.last_print_time < 2.0: return

        yolo_fps = self.timers["yolo"].ema or 0.0
        track_fps = self.timers["track"].ema or 0.0
        render_fps = self.timers["render"].ema or 0.0

        bus = get_bus()
        # multiprocessing.Queue는 qsize()가 불안정할 수 있으므로 예외처리
        q_meta = -1
        try:
            q_meta = bus.meta_q.qsize()
        except (NotImplementedError, FileNotFoundError):
            pass

        q_infer = bus.infer_out_q.qsize()

        drift_ms = 0
        if yolo_fps > 0 and q_meta != -1:
            drift_ms = (q_meta + q_infer) / yolo_fps * 1000

        stats_str = f"[EMA FPS] YOLO: {yolo_fps:.2f} | Track: {track_fps:.2f} | Render: {render_fps:.2f} | drift: {drift_ms:+.1f} ms"
        print(stats_str)
        self.last_print_time = now

    def stop_all(self):
        self.log_message("Stopping pipeline...")
        self.stop_event.set()
        time.sleep(0.5)

        if self.batcher and self.batcher.is_alive(): self.batcher.join(timeout=1)
        if self.tracker and self.tracker.is_alive(): self.tracker.join(timeout=1)
        if self.decoder_proc and self.decoder_proc.is_alive():
            self.decoder_proc.join(timeout=2)
            if self.decoder_proc.is_alive(): self.decoder_proc.terminate()

        bus = get_bus()
        for q in [bus.meta_q, bus.infer_out_q, bus.track_out_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                except (queue.Empty, FileNotFoundError):
                    break
        self.log_message("Pipeline stopped.")

    def closeEvent(self, event):
        self.stop_all()
        event.accept()


# -------------------------------------------------
# 단독 실행을 위한 엔트리포인트
# -------------------------------------------------
if __name__ == '__main__':
    # Windows/macOS에서 spawn 방식이 안정적이므로 명시적으로 설정
    # 이 코드는 모든 자식 프로세스가 생성되기 전에 호출되어야 함
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="CCTV Abnormal Detection Video Player")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to the video file to play automatically."
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    main_win = QMainWindow()
    main_win.setWindowTitle("CCTV Abnormal Detection (Standalone Test)")
    main_win.setGeometry(100, 100, 1280, 800)

    player_widget = VideoPlayerWindow(video_path=args.video_path, parent=main_win)
    main_win.setCentralWidget(player_widget)

    main_win.show()
    sys.exit(app.exec_())
