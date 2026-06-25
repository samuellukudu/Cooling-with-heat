# diffheat/viz/controls.py
"""Play/pause, frame navigation, and parameter display."""
from PyQt6 import QtCore, QtWidgets


class ControlPanel(QtWidgets.QWidget):
    """Play/pause, frame navigation, and parameter display."""

    frame_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_frames = 0
        self.current_frame = 0
        self._playing = False
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout()

        # Play/Pause
        self.btn_play = QtWidgets.QPushButton("▶ Play")
        self.btn_play.clicked.connect(self._toggle_play)
        layout.addWidget(self.btn_play)

        # Step back
        self.btn_back = QtWidgets.QPushButton("◀ Step")
        self.btn_back.clicked.connect(self._step_back)
        layout.addWidget(self.btn_back)

        # Step forward
        self.btn_fwd = QtWidgets.QPushButton("Step ▶")
        self.btn_fwd.clicked.connect(self._step_forward)
        layout.addWidget(self.btn_fwd)

        # Reset
        self.btn_reset = QtWidgets.QPushButton("⏮ Reset")
        self.btn_reset.clicked.connect(self._reset)
        layout.addWidget(self.btn_reset)

        # Frame slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        # Frame counter
        self.label_frame = QtWidgets.QLabel("0 / 0")
        layout.addWidget(self.label_frame)

        self.setLayout(layout)

    def set_n_frames(self, n: int):
        self.n_frames = n
        self.slider.setRange(0, n - 1)
        self.set_frame(0)

    def set_frame(self, frame_idx: int):
        self.current_frame = max(0, min(frame_idx, self.n_frames - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame)
        self.slider.blockSignals(False)
        self.label_frame.setText(f"{self.current_frame} / {self.n_frames - 1}")
        self.frame_changed.emit(self.current_frame)

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self.btn_play.setText("⏸ Pause")
            self._timer.start(50)  # ~20 fps
        else:
            self.btn_play.setText("▶ Play")
            self._timer.stop()

    def _step_forward(self):
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._timer.stop()
        self.set_frame(self.current_frame + 1)

    def _step_back(self):
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._timer.stop()
        self.set_frame(self.current_frame - 1)

    def _reset(self):
        self._playing = False
        self.btn_play.setText("▶ Play")
        self._timer.stop()
        self.set_frame(0)

    def _tick(self):
        if self.current_frame < self.n_frames - 1:
            self.set_frame(self.current_frame + 1)
        else:
            self._playing = False
            self.btn_play.setText("▶ Play")
            self._timer.stop()

    def _on_slider(self, value: int):
        self.current_frame = value
        self.label_frame.setText(f"{self.current_frame} / {self.n_frames - 1}")
        self.frame_changed.emit(self.current_frame)
