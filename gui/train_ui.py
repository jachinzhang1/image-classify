import sys
import yaml
import os
import torch
from train import main, Configs
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QProgressBar,
    QComboBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class TrainingController(QObject):
    should_stop = False


class TrainingThread(QThread):
    finished = pyqtSignal(bool)
    terminated = pyqtSignal()
    progress_update = pyqtSignal(int, int, int)  # current, total, epoch

    def __init__(self, config, controller):
        super().__init__()
        self.config = config
        self.controller = controller

    def run(self):
        try:
            main(self.config, self.controller, self.progress_update)
            if not self.controller.should_stop:
                self.finished.emit(True)
        except Exception as e:
            print(f"Training error: {e}")
            self.finished.emit(False)
        finally:
            self.terminated.emit()

    def stop(self):
        self.controller.should_stop = True
        self.terminated.emit()
        self.quit()


class TrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Training Interface")
        self.setGeometry(100, 100, 1500, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.normal_style = """
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 10px;
            }
        """
        self.cancelled_style = """
            QProgressBar {
                border: 1px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF0000;
                width: 10px;
            }
        """
        self.setStyleSheet(
            """
            QWidget {
                font-family: 'Microsoft YaHei';
                font-size: 12pt;
            }
            QLabel {
                font-size: 12pt;
            }
            QPushButton {
                min-width: 100px;
                min-height: 30px;
                font-size: 12pt;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                min-height: 30px;
                font-size: 12pt;
            }
        """
        )

        self.config_path = "options.yaml"
        self.default_config = self.load_config()
        self.init_ui()
        self.load_default_values()
        self.training_controller = None

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found, using hardcoded defaults")
            return {
                "device": "cuda",
                "data_root": "./data",
                "num_classes": 10,
                "train": {
                    "n_epochs": 5,
                    "batch_size": 64,
                    "lr": 0.001,
                    "output_dir": "./ckpts",
                },
            }

    def init_ui(self):
        layout = QVBoxLayout()

        # Data settings
        data_group = QWidget()
        data_group.setMinimumWidth(600)
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)
        data_layout.addWidget(QLabel("<b>Data Settings</b>"))

        self.data_root_edit = QLineEdit()
        data_layout.addWidget(QLabel("Data Root:"))
        data_layout.addWidget(self.data_root_edit)
        browse_data_btn = QPushButton("Browse...")
        browse_data_btn.clicked.connect(self.browse_data_dir)
        data_layout.addWidget(browse_data_btn)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Training settings
        train_group = QWidget()
        train_layout = QVBoxLayout()
        train_layout.addWidget(QLabel("<b>Training Settings</b>"))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        train_layout.addWidget(QLabel("Epochs:"))
        train_layout.addWidget(self.epochs_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        train_layout.addWidget(QLabel("Batch Size:"))
        train_layout.addWidget(self.batch_size_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)  # Step size of 1e-4
        train_layout.addWidget(QLabel("Learning Rate:"))
        train_layout.addWidget(self.lr_spin)

        self.output_dir_edit = QLineEdit()
        train_layout.addWidget(QLabel("Output Directory:"))
        train_layout.addWidget(self.output_dir_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        train_layout.addWidget(browse_output_btn)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # Model settings
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("<b>Model Settings</b>"))

        self.model_combo = QComboBox()
        model_layout.addWidget(QLabel("Model Type:"))
        model_layout.addWidget(self.model_combo)

        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(1, 1000)
        model_layout.addWidget(QLabel("Number of Classes:"))
        model_layout.addWidget(self.num_classes_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Device selection - modified to use QRadioButton
        device_group = QWidget()
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_group = QButtonGroup(self)
        self.device_cpu = QRadioButton("CPU")
        self.device_gpu = QRadioButton("GPU")
        self.device_group.addButton(self.device_cpu)
        self.device_group.addButton(self.device_gpu)
        device_layout.addWidget(self.device_cpu)
        device_layout.addWidget(self.device_gpu)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Control buttons
        control_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_training)  # Add click handler
        control_layout.addWidget(self.cancel_btn)
        layout.addLayout(control_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_label = QLabel("Ready")
        self.progress_bar.setStyleSheet(self.normal_style)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        self.central_widget.setLayout(layout)

    def load_default_values(self):
        # Set values from config file
        cfg = self.default_config
        self.data_root_edit.setText(cfg.get("data_root", "./data"))
        self.epochs_spin.setValue(cfg["train"].get("n_epochs", 5))
        self.batch_size_spin.setValue(cfg["train"].get("batch_size", 64))
        self.lr_spin.setValue(float(cfg["train"].get("lr", 0.001)))
        self.output_dir_edit.setText(cfg["train"].get("output_dir", "./ckpts"))
        self.num_classes_spin.setValue(cfg.get("num_classes", 10))

        # Set device radio button
        device = cfg.get("device", "cuda").lower()
        if device == torch.device("cpu") or not torch.cuda.is_available():
            self.device_cpu.setChecked(True)
        else:
            self.device_gpu.setChecked(True)

        # Load model types from config
        self.model_combo.clear()
        self.model_combo.addItems(
            self.default_config.get("model_types", ["attention_cnn"])
        )

        # Set selected model
        selected_model = self.default_config.get("selected_model", "attention_cnn")
        index = self.model_combo.findText(selected_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_root_edit.setText(dir_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def cancel_training(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Cancel",
                "Are you sure you want to cancel training?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.thread.stop()
                self.progress_bar.setStyleSheet(self.cancelled_style)
                self.progress_label.setText("Training cancelled.")
                self.set_ui_enabled(True)

    def start_training(self):
        # Create config dictionary
        config_dict = {
            "device": "cuda" if self.device_gpu.isChecked() else "cpu",
            "data_root": os.path.abspath(self.data_root_edit.text()),
            "model_types": self.default_config.get("model_types", ["attention_cnn"]),
            "selected_model": self.model_combo.currentText(),
            "num_classes": self.num_classes_spin.value(),
            "train": {
                "n_epochs": self.epochs_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "lr": float(self.lr_spin.value()),
                "output_dir": os.path.abspath(self.output_dir_edit.text()),
            },
        }

        # Save config to temporary file
        config_path = "temp_train_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        # Create Configs object
        config = Configs(config_path=config_path)

        # Disable UI during training
        self.set_ui_enabled(False)
        self.progress_bar.setStyleSheet(self.normal_style)
        self.progress_label.setText("Training started...")

        # Start training thread
        self.training_controller = TrainingController()
        self.thread = TrainingThread(config, self.training_controller)
        self.thread.finished.connect(self.training_finished)
        self.thread.terminated.connect(lambda: self.set_ui_enabled(True))
        self.thread.progress_update.connect(self.update_progress)
        self.thread.start()

    def update_progress(self, current, total, epoch):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Epoch {epoch} - {current}/{total} iterations")

    def set_ui_enabled(self, enabled):
        self.train_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        # Disable all input widgets
        for widget in self.findChildren(
            (QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton)
        ):
            if widget not in [self.train_btn, self.cancel_btn]:
                widget.setEnabled(enabled)

    def training_finished(self, success):
        self.set_ui_enabled(True)
        if success:
            QMessageBox.information(self, "Success", "Training completed successfully!")
        else:
            QMessageBox.warning(
                self, "Error", "Training failed. Check console for details."
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingApp()
    window.show()
    sys.exit(app.exec_())
