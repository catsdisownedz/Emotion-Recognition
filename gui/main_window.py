"""
Main GUI window for Speech Emotion Recognition.
Modern dark purple theme, professional appearance.
"""

import os
from pathlib import Path
import random

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar,
    QGroupBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from .styles import COLORS, GLOBAL_STYLESHEET

# -----------------------------
# Expected emotion labels
# -----------------------------
EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fear", "disgust", "surprise"
]

# -----------------------------
# Optional real predictor import
# -----------------------------
try:
    from src.predict import EmotionPredictor as RealEmotionPredictor  # type: ignore
except ImportError:
    RealEmotionPredictor = None


class DummyEmotionPredictor:
    """
    Temporary fake predictor so the GUI can work & demo
    even before the real model is implemented.
    """

    def __init__(self, *_args, **_kwargs):
        pass

    def predict(self, audio_path: str) -> dict:
        # Generate a random but sensible probability distribution
        probs = [random.random() for _ in EMOTION_LABELS]
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]

        # Pick the top emotion
        max_idx = max(range(len(probs)), key=lambda i: probs[i])
        emotion = EMOTION_LABELS[max_idx]
        confidence = probs[max_idx]

        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                label: p for label, p in zip(EMOTION_LABELS, probs)
            },
        }


class PredictionThread(QThread):
    """
    Run prediction in a separate thread to avoid freezing the GUI.
    """

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
            self.error.emit(f"Prediction error: {str(e)}")


class EmotionRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.current_audio_path: str | None = None
        self.prediction_thread: PredictionThread | None = None
        self.timer: QTimer | None = None

        self.init_ui()
        self.load_model()

    # -----------------------------
    # UI Setup
    # -----------------------------
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("🎵 Speech Emotion Recognition")
        self.setGeometry(100, 100, 1050, 720)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Controls
        left_panel = self.create_left_panel()

        # Right panel - Results
        right_panel = self.create_right_panel()

        main_layout.addWidget(left_panel, 38)
        main_layout.addWidget(right_panel, 62)

    def create_left_panel(self):
        """Create left control panel."""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        layout.setSpacing(14)

        # Title
        title = QLabel("🎤 Emotion Predictor")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Upload a voice clip and let the model detect the emotion.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(18)

        # Model status
        model_label = QLabel("Model Status")
        model_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(model_label)

        self.model_status = QLabel("Loading model...")
        self.model_status.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.model_status)

        layout.addSpacing(18)

        # File selection
        file_label = QLabel("Audio File")
        file_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(file_label)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; padding: 8px;"
        )
        self.file_path_label.setWordWrap(True)
        layout.addWidget(self.file_path_label)

        # Browse button
        browse_btn = QPushButton("📁 Browse Audio File")
        browse_btn.clicked.connect(self.browse_audio)
        layout.addWidget(browse_btn)

        layout.addSpacing(16)

        # Predict button
        self.predict_btn = QPushButton("🎯 Predict Emotion")
        self.predict_btn.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.predict_btn.setMinimumHeight(46)
        self.predict_btn.clicked.connect(self.run_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
            }}
        """
        )
        layout.addWidget(self.progress_bar)

        # Recent files
        layout.addSpacing(28)
        recent_label = QLabel("Recent Predictions")
        recent_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(170)
        self.recent_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
        """
        )
        layout.addWidget(self.recent_list)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_right_panel(self):
        """Create right results panel."""
        group = QGroupBox("Results")
        layout = QVBoxLayout()
        layout.setSpacing(18)

        # Main emotion result
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
        self.confidence_label.setStyleSheet(
            f"color: {COLORS['text_secondary']};"
        )
        result_layout.addWidget(self.confidence_label)

        result_container.setLayout(result_layout)
        layout.addWidget(result_container)

        # Probability bars
        prob_label = QLabel("All Emotion Probabilities")
        prob_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(prob_label)

        self.prob_list = QListWidget()
        self.prob_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {COLORS['bg_secondary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """
        )
        layout.addWidget(self.prob_list)

        # Clear button
        clear_btn = QPushButton("Clear Results")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    # -----------------------------
    # Model loading + prediction
    # -----------------------------
    def load_model(self):
        """
        Load the real model if available, otherwise fall back to dummy.
        This keeps the GUI usable even before the ML part is finished.
        """
        model_path = "models/trained/emotion_cnn.h5"

        if RealEmotionPredictor is not None and os.path.exists(model_path):
            try:
                self.predictor = RealEmotionPredictor(model_path)
                self.model_status.setText("✅ Model loaded successfully")
                self.model_status.setStyleSheet(
                    f"color: {COLORS['success']};"
                )
            except Exception as e:
                self.model_status.setText(
                    f"⚠️ Failed to load real model, using demo mode.\n{e}"
                )
                self.model_status.setStyleSheet(
                    f"color: {COLORS['warning']};"
                )
                self.predictor = DummyEmotionPredictor()
        else:
            # No real model yet
            self.model_status.setText(
                "🧪 Demo mode: waiting for real model (using dummy predictions)."
            )
            self.model_status.setStyleSheet(
                f"color: {COLORS['text_secondary']};"
            )
            self.predictor = DummyEmotionPredictor()

        # We only allow prediction once a file is selected.
        self.predict_btn.setEnabled(False)

    def browse_audio(self):
        """Open file dialog to select audio."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*.*)",
        )

        if file_path:
            self.current_audio_path = file_path
            filename = Path(file_path).name
            self.file_path_label.setText(f"📁 {filename}")
            # Enable prediction once we have a file and a predictor
            if self.predictor is not None:
                self.predict_btn.setEnabled(True)

    def run_prediction(self):
        """Run emotion prediction in a separate thread."""
        if not self.current_audio_path or not self.predictor:
            return

        self.predict_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Small loading animation
        def animate():
            self.progress_bar.setValue((self.progress_bar.value() + 8) % 100)

        self.timer = QTimer()
        self.timer.timeout.connect(animate)
        self.timer.start(100)

        # Run prediction in a background thread
        self.prediction_thread = PredictionThread(
            self.predictor, self.current_audio_path
        )
        self.prediction_thread.finished.connect(self.display_results)
        self.prediction_thread.error.connect(self.show_error)
        self.prediction_thread.finished.connect(lambda: self.timer.stop())
        self.prediction_thread.start()

    def display_results(self, result: dict):
        """Display prediction results from model/dummy."""
        emotion = result.get("emotion", "—")
        confidence = float(result.get("confidence", 0.0))
        probabilities = result.get("probabilities", {})

        # Update main emotion
        self.emotion_label.setText(emotion)
        self.confidence_label.setText(f"Confidence: {confidence * 100:.1f}%")
        self.confidence_label.setStyleSheet(
            f"color: {COLORS['text_secondary']};"
        )

        # Update probability list
        self.prob_list.clear()

        # Sort by probability descending; make sure we only display known labels
        sorted_items = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        for emotion_name, prob in sorted_items:
            if emotion_name not in EMOTION_LABELS:
                continue

            prob_pct = prob * 100

            # Create progress bar item
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)

            emotion_label = QLabel(f"{emotion_name.capitalize()}: {prob_pct:.1f}%")
            emotion_label.setMinimumWidth(140)
            emotion_label.setFont(QFont("Segoe UI", 10))
            item_layout.addWidget(emotion_label)

            # Progress bar
            bar = QProgressBar()
            bar.setValue(int(prob_pct))
            bar.setMaximumHeight(18)
            bar.setStyleSheet(
                f"""
                QProgressBar {{
                    background-color: {COLORS['bg_tertiary']};
                    border: none;
                    border-radius: 3px;
                }}
                QProgressBar::chunk {{
                    background-color: {COLORS['primary']};
                    border-radius: 3px;
                }}
            """
            )
            item_layout.addWidget(bar)

            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.prob_list.addItem(list_item)
            self.prob_list.setItemWidget(list_item, item_widget)

        # Add to recent
        if self.current_audio_path:
            filename = Path(self.current_audio_path).name
            recent_text = f"{filename}: {emotion} ({confidence * 100:.1f}%)"
            self.recent_list.insertItem(0, recent_text)
            if self.recent_list.count() > 10:
                self.recent_list.takeItem(self.recent_list.count() - 1)

        # Re-enable button and hide progress
        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def show_error(self, error_msg: str):
        """Show error message."""
        self.emotion_label.setText("❌ Error")
        self.confidence_label.setText(error_msg)
        self.confidence_label.setStyleSheet(f"color: {COLORS['error']};")

        self.predict_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def clear_results(self):
        """Clear all results."""
        self.emotion_label.setText("—")
        self.confidence_label.setText("Confidence: —")
        self.confidence_label.setStyleSheet(
            f"color: {COLORS['text_secondary']};"
        )
        self.prob_list.clear()
        self.current_audio_path = None
        self.file_path_label.setText("No file selected")
        self.predict_btn.setEnabled(False)
