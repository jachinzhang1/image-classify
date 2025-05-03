import sys
import os
import yaml
import torch
from configs import Configs
from test import main as test_main
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
    QFileDialog,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QProgressBar,
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class TestController(QObject):
    should_stop = False


class TestThread(QThread):
    finished = pyqtSignal(bool)
    terminated = pyqtSignal()
    progress_update = pyqtSignal(int, int)  # current, total
    accuracy_update = pyqtSignal(float)  # Add accuracy signal

    def __init__(self, config, controller):
        super().__init__()
        self.config = config
        self.controller = controller

    def run(self):
        try:
            # Modify test_main to emit progress and accuracy
            test_main(
                self.config, self.controller, self.progress_update, self.accuracy_update
            )
            if not self.controller.should_stop:
                self.finished.emit(True)
        except Exception as e:
            print(f"Testing error: {e}")
            self.finished.emit(False)

    def stop(self):
        self.controller.should_stop = True
        self.terminated.emit()
        self.quit()


class TestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Testing Interface")
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
            QLineEdit, QSpinBox {
                min-height: 30px;
                font-size: 12pt;
            }
        """
        )

        self.config_path = "options.yaml"
        self.default_config = self.load_config()
        self.init_ui()
        self.load_default_values()
        self.testing_controller = None

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found")
            return {
                "device": "cuda",
                "data_root": "./data",
                "num_classes": 10,
                "test": {"batch_size": 64, "ckpt_path": "./ckpts/model.pth"},
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

        # Model settings
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("<b>Model Settings</b>"))

        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(1, 1000)
        model_layout.addWidget(QLabel("Number of Classes:"))
        model_layout.addWidget(self.num_classes_spin)

        self.ckpt_path_edit = QLineEdit()
        model_layout.addWidget(QLabel("Checkpoint Path:"))
        model_layout.addWidget(self.ckpt_path_edit)
        browse_ckpt_btn = QPushButton("Browse...")
        browse_ckpt_btn.clicked.connect(self.browse_ckpt_file)
        model_layout.addWidget(browse_ckpt_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Test settings
        test_group = QWidget()
        test_layout = QVBoxLayout()
        test_layout.addWidget(QLabel("<b>Test Settings</b>"))

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1024)
        test_layout.addWidget(QLabel("Batch Size:"))
        test_layout.addWidget(self.batch_size_spin)

        test_group.setLayout(test_layout)
        layout.addWidget(test_group)

        # Device selection
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
        self.test_btn = QPushButton("Start Testing")
        self.test_btn.clicked.connect(self.start_testing)
        control_layout.addWidget(self.test_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_testing)
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
        cfg = self.default_config
        self.data_root_edit.setText(cfg.get("data_root", "./data"))
        self.num_classes_spin.setValue(cfg.get("num_classes", 10))
        self.batch_size_spin.setValue(cfg["test"].get("batch_size", 64))
        self.ckpt_path_edit.setText(cfg["test"].get("ckpt_path", "./ckpts/model.pth"))

        device = cfg.get("device", "cuda").lower()
        if device == torch.device("cpu") or not torch.cuda.is_available():
            self.device_cpu.setChecked(True)
        else:
            self.device_gpu.setChecked(True)

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_root_edit.setText(dir_path)

    def browse_ckpt_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", "./ckpts", "Model Files (*.pth *.pt)"
        )
        if file_path:
            self.ckpt_path_edit.setText(file_path)

    def cancel_testing(self):
        if hasattr(self, "thread") and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Cancel",
                "Are you sure you want to cancel testing?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.thread.stop()
                self.progress_bar.setStyleSheet(self.cancelled_style)
                self.progress_label.setText("Testing cancelled.")
                self.set_ui_enabled(True)

    def start_testing(self):
        ckpt_path = os.path.abspath(self.ckpt_path_edit.text())
        model_types = self.default_config.get("model_types", ["attention_cnn"])

        # Find model type that matches checkpoint path
        selected_model = None
        for model in model_types:
            if model.lower() in ckpt_path.lower():
                selected_model = model
                break

        if not selected_model:
            QMessageBox.warning(
                self,
                "Error",
                "Could not determine model type from checkpoint path.\n"
                f"Path should contain one of: {', '.join(model_types)}",
            )
            return

        config_dict = {
            "device": "cuda" if self.device_gpu.isChecked() else "cpu",
            "data_root": os.path.abspath(self.data_root_edit.text()),
            "num_classes": self.num_classes_spin.value(),
            "model_types": model_types,
            "selected_model": selected_model,
            "test": {
                "batch_size": self.batch_size_spin.value(),
                "ckpt_path": ckpt_path,
            },
        }

        config_path = "temp_test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = Configs(config_path=config_path)

        self.set_ui_enabled(False)
        self.progress_bar.setStyleSheet(self.normal_style)
        self.progress_label.setText("Testing started...")

        self.testing_controller = TestController()
        self.thread = TestThread(config, self.testing_controller)
        self.thread.finished.connect(self.testing_finished)
        self.thread.progress_update.connect(self.update_progress)
        self.thread.accuracy_update.connect(self.update_accuracy)  # Add this line
        self.thread.start()

    def update_progress(self, current, total):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Processing {current}/{total} batches")

    def update_accuracy(self, acc):
        self.progress_label.setText(f"Accuracy: {acc:.2%}")

    def set_ui_enabled(self, enabled):
        self.test_btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        for widget in self.findChildren((QLineEdit, QSpinBox, QPushButton)):
            if widget not in [self.test_btn, self.cancel_btn]:
                widget.setEnabled(enabled)

    def testing_finished(self, success):
        self.set_ui_enabled(True)
        if success:
            QMessageBox.information(self, "Success", "Testing completed successfully!")
        else:
            QMessageBox.warning(
                self, "Error", "Testing failed. Check console for details."
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestApp()
    window.show()
    sys.exit(app.exec_())
