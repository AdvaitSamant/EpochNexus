# UI_main.py
import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QStackedWidget, QLabel, QTextEdit, QSizePolicy,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QFrame, QComboBox, QSpinBox, QCheckBox,
    QRadioButton, QButtonGroup, QTabWidget, QProgressBar, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

from core import AIModelManager


class SimpleModeWidget(QWidget):
    """
    Simple mode: MCQ-style quick configuration.
    """
    def __init__(self, start_callback=None, back_callback=None):
        super().__init__()
        self.start_callback = start_callback
        self.back_callback = back_callback
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Simple Mode — Quick Setup")
        title.setStyleSheet("font-size:20px; font-weight:bold; color: #004d40;")
        layout.addWidget(title, alignment=Qt.AlignLeft)

        # Step 1: Task type (radio buttons)
        task_group = QGroupBox("1) What do you want to do?")
        task_layout = QVBoxLayout()
        self.task_radios = []
        self.task_btn_group = QButtonGroup(self)
        for i, txt in enumerate(["Classification", "Regression", "Clustering"]):
            r = QRadioButton(txt)
            if i == 0:
                r.setChecked(True)
            self.task_btn_group.addButton(r, i)
            task_layout.addWidget(r)
            self.task_radios.append(r)
        task_group.setLayout(task_layout)
        layout.addWidget(task_group)

        # Step 2: Model size
        size_group = QGroupBox("2) Model size")
        size_layout = QVBoxLayout()
        self.size_btn_group = QButtonGroup(self)
        for i, txt in enumerate(["Tiny (fast)", "Balanced", "Accurate (slower)"]):
            r = QRadioButton(txt)
            if i == 1:
                r.setChecked(True)
            self.size_btn_group.addButton(r, i)
            size_layout.addWidget(r)
        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Step 3: Training length
        train_group = QGroupBox("3) Training length")
        train_layout = QVBoxLayout()
        self.train_btn_group = QButtonGroup(self)
        for i, txt in enumerate(["Quick (5 epochs)", "Standard (20 epochs)", "Thorough (50 epochs)"]):
            r = QRadioButton(txt)
            if i == 1:
                r.setChecked(True)
            self.train_btn_group.addButton(r, i)
            train_layout.addWidget(r)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # Start button + back
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.on_start)
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self._on_back)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(back_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.setLayout(layout)

    def _on_back(self):
        if callable(self.back_callback):
            self.back_callback()

    def on_start(self):
        # Map MCQ answers to a config dictionary
        task = self.task_btn_group.checkedId()
        size = self.size_btn_group.checkedId()
        length = self.train_btn_group.checkedId()

        config = {}
        # Task mapping
        config['task'] = ['classification', 'regression', 'clustering'][task]
        # Size presets -> architecture mapping
        if size == 0:  # Tiny
            config.update({'layers': 2, 'hidden_units': 32, 'batch_size': 32})
        elif size == 1:  # Balanced
            config.update({'layers': 3, 'hidden_units': 128, 'batch_size': 32})
        else:  # Accurate
            config.update({'layers': 6, 'hidden_units': 512, 'batch_size': 16})

        # Epochs mapping
        config['epochs'] = [5, 20, 50][length]

        # Default training params
        config.update({'optimizer': 'Adam', 'learning_rate': 1e-3, 'dropout': 0.2, 'augmentation': False})

        # Send config to parent to handle training
        if callable(self.start_callback):
            self.start_callback(config)
        else:
            QMessageBox.information(self, "Config", json.dumps(config, indent=2))


class AdvancedModeWidget(QWidget):
    """
    Advanced mode: tabbed, full customization and save/load presets.
    """
    def __init__(self, start_callback=None, back_callback=None):
        super().__init__()
        self.start_callback = start_callback
        self.back_callback = back_callback
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Advanced Mode — Full Customization")
        title.setStyleSheet("font-size:20px; font-weight:bold; color: #004d40;")
        layout.addWidget(title, alignment=Qt.AlignLeft)

        # Presets + load/save
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Presets:"))
        self.presets_combo = QComboBox()
        self.presets_combo.addItems(["Custom", "Beginner", "Balanced", "Expert"])
        self.presets_combo.currentIndexChanged.connect(self._on_preset_change)
        presets_layout.addWidget(self.presets_combo)

        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.clicked.connect(self.save_preset)
        presets_layout.addWidget(self.save_preset_btn)

        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self.load_preset)
        presets_layout.addWidget(self.load_preset_btn)

        layout.addLayout(presets_layout)

        # Tab widget
        tabs = QTabWidget()

        # Model Architecture tab
        arch_tab = QWidget()
        arch_layout = QVBoxLayout()
        self.layers_spin = QSpinBox(); self.layers_spin.setRange(1, 200); self.layers_spin.setValue(3)
        self.hidden_units_spin = QSpinBox(); self.hidden_units_spin.setRange(1, 4096); self.hidden_units_spin.setValue(128)
        arch_layout.addWidget(QLabel("Layers:"))
        arch_layout.addWidget(self.layers_spin)
        arch_layout.addWidget(QLabel("Hidden units per layer:"))
        arch_layout.addWidget(self.hidden_units_spin)
        arch_tab.setLayout(arch_layout)
        tabs.addTab(arch_tab, "Model")

        # Training tab
        train_tab = QWidget()
        train_layout = QVBoxLayout()
        self.optimizer_combo = QComboBox(); self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "Adagrad"])
        self.lr_edit = QLineEdit("0.001")
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 1024); self.batch_spin.setValue(32)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(20)
        train_layout.addWidget(QLabel("Optimizer:"))
        train_layout.addWidget(self.optimizer_combo)
        train_layout.addWidget(QLabel("Learning rate:"))
        self.lr_edit.setToolTip("Enter as floating point, e.g. 0.001")
        train_layout.addWidget(self.lr_edit)
        train_layout.addWidget(QLabel("Batch size:"))
        train_layout.addWidget(self.batch_spin)
        train_layout.addWidget(QLabel("Epochs:"))
        train_layout.addWidget(self.epochs_spin)
        train_tab.setLayout(train_layout)
        tabs.addTab(train_tab, "Training")

        # Regularization tab
        reg_tab = QWidget()
        reg_layout = QVBoxLayout()
        self.augmentation_chk = QCheckBox("Use data augmentation")
        self.dropout_spin = QSpinBox(); self.dropout_spin.setRange(0, 100); self.dropout_spin.setValue(20)
        reg_layout.addWidget(self.augmentation_chk)
        reg_layout.addWidget(QLabel("Dropout (%)"))
        reg_layout.addWidget(self.dropout_spin)
        reg_tab.setLayout(reg_layout)
        tabs.addTab(reg_tab, "Regularization")

        layout.addWidget(tabs)

        # Buttons layout
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply & Start")
        self.apply_btn.clicked.connect(self.on_apply)
        back_btn = QPushButton("← Back")
        back_btn.clicked.connect(self._on_back)
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(back_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()
        self.setLayout(layout)

    def _on_preset_change(self, idx):
        name = self.presets_combo.currentText()
        if name == 'Beginner':
            self.layers_spin.setValue(2); self.hidden_units_spin.setValue(64); self.batch_spin.setValue(32); self.epochs_spin.setValue(10); self.lr_edit.setText('0.001')
        elif name == 'Balanced':
            self.layers_spin.setValue(3); self.hidden_units_spin.setValue(128); self.batch_spin.setValue(32); self.epochs_spin.setValue(20); self.lr_edit.setText('0.001')
        elif name == 'Expert':
            self.layers_spin.setValue(8); self.hidden_units_spin.setValue(512); self.batch_spin.setValue(16); self.epochs_spin.setValue(100); self.lr_edit.setText('0.0005')
        # Custom keeps values as-is

    def save_preset(self):
        cfg = self._gather_config()
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'w') as f:
                    json.dump(cfg, f, indent=2)
                QMessageBox.information(self, "Saved", "Preset saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r') as f:
                    cfg = json.load(f)
                self._apply_config(cfg)
                QMessageBox.information(self, "Loaded", "Preset loaded.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _gather_config(self):
        try:
            lr = float(self.lr_edit.text())
        except Exception:
            lr = 0.001
        return {
            'mode': 'advanced',
            'layers': self.layers_spin.value(),
            'hidden_units': self.hidden_units_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'learning_rate': lr,
            'batch_size': self.batch_spin.value(),
            'epochs': self.epochs_spin.value(),
            'augmentation': self.augmentation_chk.isChecked(),
            'dropout': self.dropout_spin.value() / 100.0
        }

    def _apply_config(self, cfg):
        if 'layers' in cfg:
            self.layers_spin.setValue(int(cfg.get('layers', 3)))
        if 'hidden_units' in cfg:
            self.hidden_units_spin.setValue(int(cfg.get('hidden_units', 128)))
        if 'optimizer' in cfg:
            idx = self.optimizer_combo.findText(cfg.get('optimizer', 'Adam'))
            if idx >= 0:
                self.optimizer_combo.setCurrentIndex(idx)
        if 'learning_rate' in cfg:
            self.lr_edit.setText(str(cfg.get('learning_rate', 0.001)))
        if 'batch_size' in cfg:
            self.batch_spin.setValue(int(cfg.get('batch_size', 32)))
        if 'epochs' in cfg:
            self.epochs_spin.setValue(int(cfg.get('epochs', 20)))
        if 'augmentation' in cfg:
            self.augmentation_chk.setChecked(bool(cfg.get('augmentation', False)))
        if 'dropout' in cfg:
            self.dropout_spin.setValue(int(float(cfg.get('dropout', 0.2)) * 100))

    def _on_back(self):
        if callable(self.back_callback):
            self.back_callback()

    def on_apply(self):
        cfg = self._gather_config()
        if callable(self.start_callback):
            self.start_callback(cfg)
        else:
            QMessageBox.information(self, "Config", json.dumps(cfg, indent=2))


class LandingWidget(QWidget):
    """
    Landing page with two large cards/buttons.
    """
    def __init__(self, simple_callback=None, advanced_callback=None):
        super().__init__()
        self.simple_callback = simple_callback
        self.advanced_callback = advanced_callback
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 60, 20, 20)
        title = QLabel("Welcome to EpochNexus")
        title.setStyleSheet("font-size:28px; font-weight:bold; color: #006064;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        subtitle = QLabel("Quickly build/train models using Simple or Advanced mode")
        subtitle.setStyleSheet("color:#004d40; font-size:14px;")
        layout.addWidget(subtitle, alignment=Qt.AlignCenter)

        btn_layout = QHBoxLayout()
        self.simple_btn = QPushButton("Simple Mode")
        self.advanced_btn = QPushButton("Advanced Mode")

        for btn in (self.simple_btn, self.advanced_btn):
            btn.setMinimumSize(220, 120)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    font-weight: bold;
                    border-radius: 10px;
                }
            """)

        self.simple_btn.clicked.connect(self._on_simple)
        self.advanced_btn.clicked.connect(self._on_advanced)
        btn_layout.addWidget(self.simple_btn)
        btn_layout.addWidget(self.advanced_btn)
        layout.addLayout(btn_layout)

        footer = QLabel("Choose a mode to continue — non-CS friendly forms available.")
        footer.setStyleSheet("color:#666; font-style:italic; padding-top:20px;")
        layout.addWidget(footer, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def _on_simple(self):
        if callable(self.simple_callback):
            self.simple_callback()

    def _on_advanced(self):
        if callable(self.advanced_callback):
            self.advanced_callback()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EpochNexus - AI Model Manager")
        self.setMinimumSize(1100, 700)
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
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Left: pages stack
        self.stack = QStackedWidget()
        self.landing_page = LandingWidget(simple_callback=self.show_simple_mode,
                                          advanced_callback=self.show_advanced_mode)
        self.simple_page = SimpleModeWidget(start_callback=self.start_training, back_callback=self.show_landing)
        self.advanced_page = AdvancedModeWidget(start_callback=self.start_training, back_callback=self.show_landing)

        self.stack.addWidget(self.landing_page)
        self.stack.addWidget(self.simple_page)
        self.stack.addWidget(self.advanced_page)

        left_frame = QFrame()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.stack)
        left_frame.setLayout(left_layout)
        left_frame.setMinimumWidth(650)
        main_layout.addWidget(left_frame)

        # Right: shared dataset + preview + status + progress + logs
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # Dataset selection
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
        right_layout.addWidget(file_group)

        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        self.preview_table = QTableWidget(); self.preview_table.setMaximumHeight(240)
        preview_layout.addWidget(self.preview_table)
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        preview_layout.addWidget(self.dataset_info_label)
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group)

        # Status + progress + logs
        status_group = QGroupBox("Status & Training")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #006064; font-weight: bold; padding: 5px;")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(160)
        status_layout.addWidget(self.log_box)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        right_panel.setMaximumWidth(380)
        main_layout.addWidget(right_panel)

        central_widget.setLayout(main_layout)

        # Training timer (simulated progress) state
        self._train_timer = QTimer()
        self._train_timer.setInterval(500)
        self._train_timer.timeout.connect(self._on_train_tick)
        self._train_state = None

        self.show_landing()

    # Navigation
    def show_landing(self):
        self.stack.setCurrentIndex(0)

    def show_simple_mode(self):
        self.stack.setCurrentIndex(1)

    def show_advanced_mode(self):
        self.stack.setCurrentIndex(2)

    # Dataset functions
    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)")
        if file_path:
            self.load_dataset(file_path)

    def load_dataset(self, file_path):
        try:
            self.status_label.setText("Loading dataset...")
            self.file_path_label.setText(f"File: {os.path.basename(file_path)}")
            QApplication.processEvents()

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
        if preview_data is None:
            return
        self.preview_table.clear()
        rows = len(preview_data)
        cols = len(preview_data.columns)
        self.preview_table.setRowCount(rows)
        self.preview_table.setColumnCount(cols)
        self.preview_table.setHorizontalHeaderLabels(list(preview_data.columns))
        try:
            self.preview_table.setVerticalHeaderLabels([f"Row {i+1}" for i in range(rows)])
        except Exception:
            pass
        for row_idx, (_, row_data) in enumerate(preview_data.iterrows()):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(row_idx, col_idx, item)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def display_dataset_info(self, dataset_info):
        if dataset_info is None:
            return
        info_text = f"Shape: {dataset_info['shape'][0]} rows × {dataset_info['shape'][1]} columns\n"
        info_text += f"Columns: {', '.join(dataset_info['columns'][:5])}"
        if len(dataset_info['columns']) > 5:
            info_text += f" ... and {len(dataset_info['columns']) - 5} more"
        self.dataset_info_label.setText(info_text)
        self.dataset_info_label.setStyleSheet("color: #006064; font-weight: bold; padding: 5px;")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)

    # Training orchestration (simple simulation + hook to AIModelManager)
    def start_training(self, config):
        # Called by simple/advanced mode widgets
        # Validate dataset
        try:
            ds_info = self.ai_manager.get_dataset_info()
            if not ds_info:
                raise RuntimeError('No dataset loaded. Please upload dataset before training.')
        except Exception:
            self.show_error('No dataset loaded. Please upload dataset before training.')
            return

        # Display config summary in log
        self.log_box.append("Starting training with configuration:")
        self.log_box.append(json.dumps(config, indent=2))
        self.progress_bar.setValue(0)
        self.status_label.setText("Training...")
        self.status_label.setStyleSheet("color: #ef6c00; font-weight: bold; padding: 5px;")

        # Save training state: we'll simulate epochs and ticks
        epochs = int(config.get('epochs', 10))
        self._train_state = {'config': config, 'epochs': epochs, 'current_epoch': 0}

        # In a real app you'd call AIModelManager.train(config) and connect callbacks; here we simulate
        self._train_timer.start()

    def _on_train_tick(self):
        if not self._train_state:
            return
        state = self._train_state
        state['current_epoch'] += 1
        epoch = state['current_epoch']
        epochs = state['epochs']

        # Simulated metrics
        acc = min(0.99, 0.5 + 0.5 * (epoch / max(1, epochs)))
        loss = max(0.01, 1.0 * (1 - (epoch / max(1, epochs))))

        self.log_box.append(f"Epoch {epoch}/{epochs} — acc: {acc:.3f} loss: {loss:.3f}")
        progress = int((epoch / epochs) * 100)
        self.progress_bar.setValue(progress)

        if epoch >= epochs:
            self._train_timer.stop()
            self.progress_bar.setValue(100)
            self.log_box.append("Training complete. Model saved to: models/last_model.pth (simulated)")
            self.status_label.setText("Training complete")
            self.status_label.setStyleSheet("color: #2e7d32; font-weight: bold; padding: 5px;")
            self._train_state = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
