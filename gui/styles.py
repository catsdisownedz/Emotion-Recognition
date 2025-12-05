"""
Custom dark purple theme for the emotion recognition GUI.
Modern, sleek, professional appearance.
"""

# Purple color palette
COLORS = {
    "primary": "#7C3AED",        # Vibrant purple
    "primary_dark": "#6D28D9",   # Darker purple
    "primary_light": "#A78BFA",  # Lighter purple
    
    "bg_primary": "#0F172A",     # Almost black (dark blue-gray)
    "bg_secondary": "#1E293B",   # Slightly lighter
    "bg_tertiary": "#334155",    # Even lighter for hover
    
    "text_primary": "#F1F5F9",   # Off-white
    "text_secondary": "#CBD5E1", # Lighter gray
    "text_tertiary": "#94A3B8",  # Medium gray
    
    "success": "#10B981",        # Green
    "warning": "#F59E0B",        # Orange
    "error": "#EF4444",          # Red
    "info": "#3B82F6",           # Blue
    
    "border": "#475569",         # Dark gray
    "shadow": "rgba(0, 0, 0, 0.5)"
}

# Global stylesheet
GLOBAL_STYLESHEET = f"""
    QMainWindow {{
        background-color: {COLORS['bg_primary']};
    }}
    
    QWidget {{
        background-color: {COLORS['bg_primary']};
        color: {COLORS['text_primary']};
    }}
    
    QLabel {{
        color: {COLORS['text_primary']};
        font-size: 12px;
    }}
    
    QLabel#title {{
        font-size: 20px;
        font-weight: bold;
        color: {COLORS['primary']};
    }}
    
    QLabel#subtitle {{
        font-size: 14px;
        color: {COLORS['text_secondary']};
    }}
    
    QPushButton {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
        font-size: 12px;
    }}
    
    QPushButton:hover {{
        background-color: {COLORS['primary_dark']};
    }}
    
    QPushButton:pressed {{
        background-color: {COLORS['primary_light']};
    }}
    
    QPushButton:disabled {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_tertiary']};
    }}
    
    QPushButton#secondary {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_secondary']};
    }}
    
    QPushButton#secondary:hover {{
        background-color: {COLORS['primary']};
        color: white;
    }}
    
    QLineEdit, QTextEdit {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 6px;
        font-size: 11px;
    }}
    
    QLineEdit:focus, QTextEdit:focus {{
        border: 2px solid {COLORS['primary']};
        background-color: {COLORS['bg_primary']};
    }}
    
    QComboBox {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        padding: 6px;
        font-size: 11px;
    }}
    
    QComboBox::drop-down {{
        border: none;
    }}
    
    QComboBox::down-arrow {{
        image: none;
        border: none;
        width: 12px;
    }}
    
    QComboBox QAbstractItemView {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        selection-background-color: {COLORS['primary']};
        border: 1px solid {COLORS['border']};
    }}
    
    QGroupBox {{
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        margin-top: 6px;
        padding-top: 6px;
        font-weight: bold;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }}
    
    QProgressBar {{
        background-color: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 4px;
        text-align: center;
        color: {COLORS['text_primary']};
    }}
    
    QProgressBar::chunk {{
        background-color: {COLORS['primary']};
        border-radius: 3px;
    }}
    
    QScrollBar:vertical {{
        background-color: {COLORS['bg_secondary']};
        width: 12px;
        border: none;
    }}
    
    QScrollBar::handle:vertical {{
        background-color: {COLORS['primary']};
        border-radius: 6px;
        min-height: 20px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['primary_dark']};
    }}
    
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
    }}
    
    QTabBar::tab {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_secondary']};
        padding: 6px 20px;
        border: none;
    }}
    
    QTabBar::tab:selected {{
        background-color: {COLORS['primary']};
        color: white;
    }}
"""
