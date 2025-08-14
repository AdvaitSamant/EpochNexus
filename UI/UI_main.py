import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QLabel, QTextEdit, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self, switch_callback=None): 
        super().__init__()
        self.switch_callback = switch_callback
        self.setWindowTitle("EpochNexus")
        self.setMinimumSize(800, 600) 
        self.setStyleSheet(""" QWidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0f7fa, stop:1 #80deea); } """)
        
        # Create central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 20)
        layout.setSpacing(20)

        title = QLabel("Welcome to EpochNexus")
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: #006064;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        central_widget.setLayout(layout) 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()  
    window.show()
    sys.exit(app.exec_())