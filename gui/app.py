"""
Application entry point.
Uses the custom dark purple theme from styles.py.
"""

import sys
from PyQt6.QtWidgets import QApplication

from .main_window import EmotionRecognitionGUI


def run():
    """Run the application."""
    app = QApplication(sys.argv)

    # Create and show main window
    window = EmotionRecognitionGUI()
    window.show()

    sys.exit(app.exec())
