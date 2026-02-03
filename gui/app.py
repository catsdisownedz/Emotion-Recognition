# gui/app.py
import sys
from PyQt5.QtWidgets import QApplication

from gui.main_window import EmotionRecognitionGUI


def run():
    app = QApplication(sys.argv)

    # Apply dark theme (correct module name)
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
    except Exception as e:
        print("Dark theme not applied:", e)

    window = EmotionRecognitionGUI()
    window.show()
    sys.exit(app.exec_())