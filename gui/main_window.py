# gui/main_window.py
import os
from pathlib import Path
import random

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QGroupBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

from .styles import COLORS, GLOBAL_STYLESHEET

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

# Real predictor
try:
    from src.predictor import EmotionPredictor as RealEmotionPredictor  # type: ignore
except Exception:
    RealEmotionPredictor = None


class DummyEmotionPredictor:
    def predict(self, audio_path: str) -> dict:
        probs = [random.random() for _ in EMOTION_LABELS]
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]
        max_idx = max(range(len(probs)), key=lambda i: probs[i])

        return {
            "emotion": EMOTION_LABELS[max_idx],
            "confidence": probs[max_idx],
            "probabilities": {label: p for label, p in zip(EMOTION_LABELS, probs)},
        }


class PredictionThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, predictor, audio_path: str):
        super().__init__()
        self.predictor = predictor
        self.audio_path = audio_path

    def run(self):
        try:
            result = self.predictor.predict(self.audio_path)
            if not result:
                raise ValueError("Empty prediction result.")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Prediction error: {e}")


class EmotionRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.current_audio_path: str | None = None
        self.prediction_thread: PredictionThread | None = None
        self.timer: QTimer | None = None

        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle("Speech Emotion Recognition")
        self.setGeometry(100, 100, 1050, 720)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(20, 20, 20, 20)

        main_layout.addWidget(self.create_left_panel(), 38)
        main_layout.addWidget(self.create_right_panel(), 62)

    def create_left_panel(self):
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        layout.setSpacing(14)

        title = QLabel("Emotion Predictor")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Upload a voice clip and let the model detect the emotion.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(18)

        model_label = QLabel("Model Status")
        model_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(model_label)

        self.model_status = QLabel("Loading model...")
        self.model_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.model_status)

        layout.addSpacing(18)

        file_label = QLabel("Audio File")
        file_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(file_label)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 8px;")
        self.file_path_label.setWordWrap(True)
        layout.addWidget(self.file_path_label)

        browse_btn = QPushButton("Browse Audio File")
        browse_btn.clicked.connect(self.browse_audio)
        layout.addWidget(browse_btn)

        layout.addSpacing(16)

        self.predict_btn = QPushButton("Predict Emotion")
        self.predict_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.predict_btn.setMinimumHeight(46)
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addSpacing(28)
        recent_label = QLabel("Recent Predictions")
        recent_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(170)
        layout.addWidget(self.recent_list)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_right_panel(self):
        group = QGroupBox("Results")
        layout = QVBoxLayout()
        layout.setSpacing(18)

        result_container = QGroupBox("Primary Emotion")
        result_layout = QVBoxLayout()

        self.emotion_label = QLabel("—")
        self.emotion_label.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        self.emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emotion_label.setStyleSheet(f"color: {COLORS['primary']};")
        result_layout.addWidget(self.emotion_label)

        self.confidence_label = QLabel("Confidence: —")
        self.confidence_label.setFont(QFont("Segoe UI", 14))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        result_layout.addWidget(self.confidence_label)

        result_container.setLayout(result_layout)
        layout.addWidget(result_container)

        prob_label = QLabel("All Emotion Probabilities")
        prob_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(prob_label)

        self.prob_list = QListWidget()
        layout.addWidget(self.prob_list)

        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    def load_model(self):
        """
        Loads the real predictor if available and if a trained run exists.
        Otherwise uses dummy mode.
        """
        if RealEmotionPredictor is None:
            self.predictor = DummyEmotionPredictor()
            self.model_status.setText("Demo mode: predictor not available.")
            self.predict_btn.setEnabled(False)
            return

        try:
            self.predictor = RealEmotionPredictor(models_root="notebooks/models")
            self.model_status.setText("Model loaded successfully.")
            self.model_status.setStyleSheet(f"color: {COLORS['success']};")
        except Exception as e:
            self.predictor = DummyEmotionPredictor()
            self.model_status.setText(f"Demo mode: failed to load model.\n{e}")
            self.model_status.setStyleSheet(f"color: {COLORS['warning']};")

        self.predict_btn.setEnabled(False)

    def browse_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*.*)",
        )

        if file_path:
            self.current_audio_path = file_path
            self.file_path_label.setText(str(Path(file_path).name))
            if self.predictor is not None:
                self.predict_btn.setEnabled(True)

    def run_prediction(self):
        if not self.current_audio_path or not self.predictor:
            return

        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        def animate():
            self.progress_bar.setValue((self.progress_bar.value() + 8) % 100)

        self.timer = QTimer()
        self.timer.timeout.connect(animate)
        self.timer.start(100)

        self.prediction_thread = PredictionThread(self.predictor, self.current_audio_path)
        self.prediction_thread.finished.connect(self.display_results)
        self.prediction_thread.error.connect(self.show_error)
        self.prediction_thread.finished.connect(lambda: self.timer.stop())
        self.prediction_thread.start()

    def display_results(self, result: dict):
        emotion = result.get("emotion", "—")
        confidence = float(result.get("confidence", 0.0))
        probabilities = result.get("probabilities", {})

        self.emotion_label.setText(emotion)
        self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")

        self.prob_list.clear()

        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_items:
            if name not in EMOTION_LABELS:
                continue
            self.prob_list.addItem(f"{name}: {prob * 100:.1f}%")

        if self.current_audio_path:
            filename = Path(self.current_audio_path).name
            self.recent_list.insertItem(0, f"{filename}: {emotion} ({confidence * 100:.1f}%)")
            if self.recent_list.count() > 10:
                self.recent_list.takeItem(self.recent_list.count() - 1)

        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def show_error(self, msg: str):
        self.emotion_label.setText("Error")
        self.confidence_label.setText(msg)
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def clear_results(self):
        self.emotion_label.setText("—")
        self.confidence_label.setText("Confidence: —")
        self.prob_list.clear()
        self.current_audio_path = None
        self.file_path_label.setText("No file selected")
        self.predict_btn.setEnabled(False)