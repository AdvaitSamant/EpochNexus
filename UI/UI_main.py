import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QLabel, QTextEdit, QSizePolicy,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from ai_model_manager import AIModelManager

class MainWindow(QMainWindow):
    def __init__(self, switch_callback=None): 
        super().__init__()
        self.switch_callback = switch_callback
        self.setWindowTitle("EpochNexus - AI Model Manager")
        self.setMinimumSize(1000, 700) 
        self.setStyleSheet(""" 
            QWidget { 
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e0f7fa, stop:1 #80deea); 
            }
            QPushButton {
                background-color: #006064;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00838f;
            }
            QPushButton:pressed {
                background-color: #004d40;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #006064;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        self.ai_manager = AIModelManager()
        
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 20)
        layout.setSpacing(20)

        
        title = QLabel("Welcome to EpochNexus")
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: #006064;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        
        file_group = QGroupBox("Dataset Selection")
        file_layout = QVBoxLayout()
        
        
        file_button_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("Select CSV/Excel File")
        self.select_file_btn.clicked.connect(self.select_file)
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        file_button_layout.addWidget(self.select_file_btn)
        file_button_layout.addWidget(self.file_path_label)
        file_layout.addLayout(file_button_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        
        
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_table)
        
        
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        preview_layout.addWidget(self.dataset_info_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to load dataset")
        self.status_label.setStyleSheet("color: #006064; font-weight: bold; padding: 5px;")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        central_widget.setLayout(layout)

    def select_file(self):
        """Open file dialog to select CSV or Excel file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.load_dataset(file_path)

    def load_dataset(self, file_path):
        """Load the selected dataset and display preview."""
        try:
         
            self.status_label.setText("Loading dataset...")
            self.file_path_label.setText(f"File: {os.path.basename(file_path)}")
            
            self.ai_manager.load_data(file_path)
            
            preview_data = self.ai_manager.get_data_preview()
            self.display_preview(preview_data)
            
            dataset_info = self.ai_manager.get_dataset_info()
            self.display_dataset_info(dataset_info)
            
            self.status_label.setText("Dataset loaded successfully!")
            self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 5px;")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
            self.status_label.setText("Error loading dataset")
            self.status_label.setStyleSheet("color: #c62828; font-weight: bold; padding: 5px;")

    def display_preview(self, preview_data):
        """Display the data preview in the table."""
        if preview_data is None:
            return
        
        self.preview_table.setRowCount(len(preview_data))
        self.preview_table.setColumnCount(len(preview_data.columns))
        
        self.preview_table.setHorizontalHeaderLabels(preview_data.columns)
        self.preview_table.setVerticalHeaderLabels([f"Row {i+1}" for i in range(len(preview_data))])
        
        for row_idx, (_, row_data) in enumerate(preview_data.iterrows()):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(row_idx, col_idx, item)
        
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def display_dataset_info(self, dataset_info):
        """Display dataset information."""
        if dataset_info is None:
            return
        
        info_text = f"Shape: {dataset_info['shape'][0]} rows × {dataset_info['shape'][1]} columns\n"
        info_text += f"Columns: {', '.join(dataset_info['columns'][:5])}"
        if len(dataset_info['columns']) > 5:
            info_text += f" ... and {len(dataset_info['columns']) - 5} more"
        
        self.dataset_info_label.setText(info_text)
        self.dataset_info_label.setStyleSheet("color: #006064; font-weight: bold; padding: 5px;")

    def show_error(self, message):
        """Show error message in a dialog."""
        QMessageBox.critical(self, "Error", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()  
    window.show()
    sys.exit(app.exec_())