"""
Application entry point with dark theme.
"""

import sys
from PyQt6.QtWidgets import QApplication
import pyqtdarktheme

from .main_window import EmotionRecognitionGUI


def run():
    """Run the application."""
    app = QApplication(sys.argv)
    
    # Apply modern dark theme
    app.setStyleSheet(pyqtdarktheme.load_stylesheet("dark"))
    
    # Create and show main window
    window = EmotionRecognitionGUI()
    window.show()
    
    sys.exit(app.exec())
