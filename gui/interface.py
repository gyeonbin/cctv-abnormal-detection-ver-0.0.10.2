#gui.interface.py

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
from video_player import VideoPlayerWindow  # ✅ PyQt5 QOpenGLWidget 플레이어 임포트



class Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCTV YOLO Detection (PyQt5)")
        self.setGeometry(400, 200, 800, 600)
        self.video_windows = []   # ✅ 여러 개 창 참조 유지

        layout = QVBoxLayout()
        btn_open = QPushButton("동영상 열기")
        btn_open.clicked.connect(self.open_video)
        layout.addWidget(btn_open)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_video(self):
        """동영상 선택 후 독립된 VideoPlayerWindow 실행"""
        file_path, _ = QFileDialog.getOpenFileName(self, "비디오 선택", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            player_window = VideoPlayerWindow(file_path)
            player_window.show()
            player_window.start_playback()
            self.video_windows.append(player_window)  # ✅ 참조 유지 (GC 방지)

def main_interface():
    app = QApplication([])
    win = Interface()
    win.show()
    app.exec_()

if __name__ == "__main__":
    main_interface()
