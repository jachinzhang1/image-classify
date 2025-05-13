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
    QComboBox,
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
        data_layout.addWidget(QLabel("<b>Dataset Settings</b>"))

        # Add dataset selection dropdown
        self.dataset_combo = QComboBox()
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)

        # Add dataset description label
        self.dataset_desc_label = QLabel("")
        data_layout.addWidget(self.dataset_desc_label)

        self.data_root_edit = QLineEdit()
        data_layout.addWidget(QLabel("Data Directory:"))
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

        # Add model type selection
        self.model_combo = QComboBox()
        model_layout.addWidget(QLabel("Model Type:"))
        model_layout.addWidget(self.model_combo)

        # 添加ResNet变体选择下拉框
        self.resnet_variant_group = QWidget()
        resnet_variant_layout = QVBoxLayout()
        resnet_variant_layout.addWidget(QLabel("ResNet Variant:"))
        self.resnet_variant_combo = QComboBox()
        resnet_variant_layout.addWidget(self.resnet_variant_combo)
        self.resnet_variant_group.setLayout(resnet_variant_layout)
        self.resnet_variant_group.setVisible(False)  # 默认隐藏
        model_layout.addWidget(self.resnet_variant_group)

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
        # Set values from config file
        cfg = self.default_config

        # Load dataset list
        self.dataset_combo.clear()
        datasets = self.default_config.get("datasets", {})
        if datasets:
            self.dataset_combo.addItems(list(datasets.keys()))
            # Set current selected dataset
            selected_dataset = self.default_config.get("selected_dataset", "cifar10")
            index = self.dataset_combo.findText(selected_dataset)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
                # Show dataset description
                if selected_dataset in datasets:
                    self.dataset_desc_label.setText(
                        datasets[selected_dataset].get("description", "")
                    )

            # Connect dataset change signal
            self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)

            # Set data directory
            if selected_dataset in datasets:
                self.data_root_edit.setText(
                    datasets[selected_dataset].get("path", "./data")
                )
                # Set number of classes
                self.num_classes_spin.setValue(
                    datasets[selected_dataset].get("num_classes", 10)
                )
        else:
            self.data_root_edit.setText(cfg.get("data_root", "./data"))
            self.num_classes_spin.setValue(cfg.get("num_classes", 10))

        # Load model types
        self.model_combo.clear()
        self.model_combo.addItems(
            self.default_config.get("model_types", ["attention_cnn"])
        )

        # Set selected model
        selected_model = self.default_config.get("selected_model", "attention_cnn")
        index = self.model_combo.findText(selected_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        # 加载ResNet变体
        self.resnet_variant_combo.clear()
        self.resnet_variant_combo.addItems(
            self.default_config.get("resnet_variants", ["resnet18"])
        )

        # 设置选择的ResNet变体
        selected_variant = self.default_config.get(
            "selected_resnet_variant", "resnet18"
        )
        index = self.resnet_variant_combo.findText(selected_variant)
        if index >= 0:
            self.resnet_variant_combo.setCurrentIndex(index)

        # Connect model change signal
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        # Connect ResNet variant change signal
        self.resnet_variant_combo.currentTextChanged.connect(
            self.on_resnet_variant_changed
        )

        # 根据当前选择的模型显示或隐藏ResNet变体选择
        self.update_resnet_variant_visibility(selected_model)

        self.batch_size_spin.setValue(cfg["test"].get("batch_size", 64))

        # Set checkpoint path
        self.update_checkpoint_path()

        # Set device radio button
        device = cfg.get("device", "cuda").lower()
        if device == torch.device("cpu") or not torch.cuda.is_available():
            self.device_cpu.setChecked(True)
        else:
            self.device_gpu.setChecked(True)

    def on_dataset_changed(self, dataset_name):
        """Handle dataset selection change"""
        datasets = self.default_config.get("datasets", {})
        if dataset_name in datasets:
            dataset_config = datasets[dataset_name]
            # Update data directory
            self.data_root_edit.setText(dataset_config.get("path", "./data"))
            # Update number of classes
            self.num_classes_spin.setValue(dataset_config.get("num_classes", 10))
            # Update description
            self.dataset_desc_label.setText(dataset_config.get("description", ""))
            # Update checkpoint path
            self.update_checkpoint_path()

    def on_model_changed(self, model_name):
        """Handle model type selection change"""
        # 更新ResNet变体选择框的可见性
        self.update_resnet_variant_visibility(model_name)
        # Update checkpoint path when model changes
        self.update_checkpoint_path()

    def on_resnet_variant_changed(self, variant_name):
        """Handle ResNet variant selection change"""
        selected_model = self.model_combo.currentText()
        if selected_model == "resnet":
            # Update checkpoint path based on the selected variant
            self.update_checkpoint_path()

    def update_resnet_variant_visibility(self, model_name):
        """根据选择的模型类型显示或隐藏ResNet变体选择"""
        self.resnet_variant_group.setVisible(model_name == "resnet")

    def update_checkpoint_path(self):
        """Update checkpoint path based on selected dataset and model"""
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()

        # 获取实际要使用的模型名称
        actual_model = model
        if model == "resnet":
            actual_model = self.resnet_variant_combo.currentText()

        # For autoencoder, we need to handle the checkpoint path differently
        if model == "Autoencoder":
            actual_model = "autoencoder"

        ckpt_path = f"./ckpts/{dataset}/{actual_model}/latest.pth"
        self.ckpt_path_edit.setText(ckpt_path)

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.data_root_edit.setText(dir_path)

    def browse_ckpt_file(self):
        # Start in the directory for the currently selected dataset/model
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()
        start_dir = (
            os.path.join("./ckpts", dataset, model) if dataset and model else "./ckpts"
        )
        # 将反斜杠替换为正斜杠以保持一致性
        start_dir = start_dir.replace("\\", "/")

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", start_dir, "Model Files (*.pth *.pt)"
        )
        if file_path:
            # 将反斜杠替换为正斜杠以保持一致性
            file_path = file_path.replace("\\", "/")
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
        # Create config dictionary
        try:
            selected_dataset = self.dataset_combo.currentText()
            selected_model = self.model_combo.currentText()

            # Get the actual model name to use
            actual_model = selected_model
            if selected_model == "resnet":
                actual_model = self.resnet_variant_combo.currentText()

            # 获取检查点路径并统一斜杠
            ckpt_path = os.path.abspath(self.ckpt_path_edit.text())
            ckpt_path = ckpt_path.replace("\\", "/")

            # Create config dictionary
            config_dict = {
                "device": "cuda" if self.device_gpu.isChecked() else "cpu",
                "datasets": self.default_config.get("datasets", {}),
                "selected_dataset": selected_dataset,
                "selected_model": actual_model,  # Use the actual model name (resnet18 instead of resnet)
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
            self.thread.accuracy_update.connect(self.update_accuracy)
            self.thread.start()
        except Exception as e:
            print(f"Testing error: {e}")
            self.set_ui_enabled(True)
            QMessageBox.warning(
                self, "Error", "Testing failed. Check console for details."
            )

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
