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
    QCheckBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject


class TrainingController(QObject):
    should_stop = False


class TrainingThread(QThread):
    finished = pyqtSignal(bool)
    terminated = pyqtSignal()
    progress_update = pyqtSignal(int, int, int)  # current, total, epoch
    best_model_update = pyqtSignal(float, int)   # accuracy, epoch

    def __init__(self, config, controller):
        super().__init__()
        self.config = config
        self.controller = controller
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def run(self):
        try:
            model, best_acc, best_epoch = main(
                self.config, self.controller, self.progress_update)
            self.best_accuracy = best_acc
            self.best_epoch = best_epoch

            if not self.controller.should_stop:
                self.best_model_update.emit(best_acc, best_epoch)
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
            print(
                f"Warning: {self.config_path} not found, using hardcoded defaults")
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

        # Add early stopping patience parameter
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        train_layout.addWidget(QLabel("Early Stopping Patience:"))
        train_layout.addWidget(self.patience_spin)

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

        # 添加ResNet变体选择下拉框
        self.resnet_variant_group = QWidget()
        resnet_variant_layout = QVBoxLayout()
        resnet_variant_layout.addWidget(QLabel("ResNet Variant:"))
        self.resnet_variant_combo = QComboBox()
        resnet_variant_layout.addWidget(self.resnet_variant_combo)
        self.resnet_variant_group.setLayout(resnet_variant_layout)
        self.resnet_variant_group.setVisible(False)  # Default hidden
        model_layout.addWidget(self.resnet_variant_group)

        self.num_classes_spin = QSpinBox()
        self.num_classes_spin.setRange(1, 1000)
        model_layout.addWidget(QLabel("Number of Classes:"))
        model_layout.addWidget(self.num_classes_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 在model_group布局之后添加
        # 自编码器预训练设置（仅在选择Autoencoder模型时显示）
        self.pretrain_group = QWidget()
        pretrain_layout = QVBoxLayout()
        pretrain_layout.addWidget(QLabel("<b>Autoencoder Pretraining</b>"))

        # 添加是否启用预训练的复选框
        self.enable_pretrain_check = QCheckBox(
            "Enable autoencoder pretraining")
        self.enable_pretrain_check.setChecked(True)
        pretrain_layout.addWidget(self.enable_pretrain_check)

        # 预训练轮数
        self.pretrain_epochs_spin = QSpinBox()
        self.pretrain_epochs_spin.setRange(1, 100)
        self.pretrain_epochs_spin.setValue(5)
        pretrain_layout.addWidget(QLabel("Pretraining Epochs:"))
        pretrain_layout.addWidget(self.pretrain_epochs_spin)

        # 预训练学习率
        self.pretrain_lr_spin = QDoubleSpinBox()
        self.pretrain_lr_spin.setRange(0.00001, 1.0)
        self.pretrain_lr_spin.setDecimals(5)
        self.pretrain_lr_spin.setSingleStep(0.0001)
        self.pretrain_lr_spin.setValue(0.001)
        pretrain_layout.addWidget(QLabel("Pretraining Learning Rate:"))
        pretrain_layout.addWidget(self.pretrain_lr_spin)

        self.pretrain_group.setLayout(pretrain_layout)
        self.pretrain_group.setVisible(False)  # Default hidden
        layout.addWidget(self.pretrain_group)

        # 连接模型选择变化事件
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        # 连接ResNet变体更改信号
        self.resnet_variant_combo.currentTextChanged.connect(
            self.on_resnet_variant_changed)

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
        self.cancel_btn.clicked.connect(
            self.cancel_training)  # Add click handler
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
            selected_dataset = self.default_config.get(
                "selected_dataset", "cifar10")
            index = self.dataset_combo.findText(selected_dataset)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
                # Show dataset description
                if selected_dataset in datasets:
                    self.dataset_desc_label.setText(
                        datasets[selected_dataset].get("description", ""))

            # Connect dataset change signal
            self.dataset_combo.currentTextChanged.connect(
                self.on_dataset_changed)

            # Set data directory
            if selected_dataset in datasets:
                self.data_root_edit.setText(
                    datasets[selected_dataset].get("path", "./data"))
                # Set number of classes
                self.num_classes_spin.setValue(
                    datasets[selected_dataset].get("num_classes", 10))
        else:
            self.data_root_edit.setText(cfg.get("data_root", "./data"))
            self.num_classes_spin.setValue(cfg.get("num_classes", 10))

        self.epochs_spin.setValue(cfg["train"].get("n_epochs", 5))
        self.batch_size_spin.setValue(cfg["train"].get("batch_size", 64))
        self.lr_spin.setValue(float(cfg["train"].get("lr", 0.001)))
        # Set patience value for early stopping
        self.patience_spin.setValue(
            cfg["train"].get("early_stopping_patience", 10))
        self.output_dir_edit.setText(cfg["train"].get("output_dir", "./ckpts"))

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
        selected_model = self.default_config.get(
            "selected_model", "attention_cnn")
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
            "selected_resnet_variant", "resnet18")
        index = self.resnet_variant_combo.findText(selected_variant)
        if index >= 0:
            self.resnet_variant_combo.setCurrentIndex(index)

        # 连接模型更改信号
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        # 连接ResNet变体更改信号
        self.resnet_variant_combo.currentTextChanged.connect(
            self.on_resnet_variant_changed)

        # 根据当前选择的模型显示或隐藏ResNet变体选择
        self.update_resnet_variant_visibility(selected_model)

    def browse_data_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Data Directory")
        if dir_path:
            self.data_root_edit.setText(dir_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
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

    def on_dataset_changed(self, dataset_name):
        """Handle dataset selection change"""
        datasets = self.default_config.get("datasets", {})
        if dataset_name in datasets:
            dataset_config = datasets[dataset_name]
            # Update data directory
            self.data_root_edit.setText(dataset_config.get("path", "./data"))
            # Update number of classes
            self.num_classes_spin.setValue(
                dataset_config.get("num_classes", 10))
            # Update description
            self.dataset_desc_label.setText(
                dataset_config.get("description", ""))
            # Update output directory path
            self.update_output_dir_path()

    def on_model_changed(self, model_name):
        """Handle model selection change"""
        # Update ResNet variant selection visibility
        self.update_resnet_variant_visibility(model_name)

        # Show pretraining settings only if Autoencoder is selected
        self.pretrain_group.setVisible(model_name == "Autoencoder")

        # Update output directory path
        self.update_output_dir_path()

    def on_resnet_variant_changed(self, variant_name):
        """Handle ResNet variant selection change"""
        # No need to update visibility, just update configs for checkpoint path
        selected_model = self.model_combo.currentText()
        if selected_model == "resnet":
            # Update output directory path according to the selected variant
            self.update_output_dir_path()

    def update_resnet_variant_visibility(self, model_name):
        """Show or hide ResNet variant selection based on model type"""
        self.resnet_variant_group.setVisible(model_name == "resnet")

    def update_output_dir_path(self):
        """Update output directory based on current selections"""
        # This is primarily for showing the expected output path, the actual path
        # is determined during training start based on current selections
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()

        # Get the actual model name for the output path
        if model == "resnet":
            model = self.resnet_variant_combo.currentText()

        # Get the base directory from config
        base_dir = self.default_config["train"].get("output_dir", "./ckpts")

        # Set the output path to include dataset and model
        output_path = f"{base_dir}/{dataset}/{model}"

        # Update UI
        self.output_dir_edit.setText(output_path)

    def start_training(self):
        # Create config dictionary
        try:
            selected_dataset = self.dataset_combo.currentText()
            selected_model = self.model_combo.currentText()

            # Get the actual model name to use
            actual_model = selected_model
            if selected_model == "resnet":
                actual_model = self.resnet_variant_combo.currentText()

            # Set device
            if self.device_cpu.isChecked():
                device = torch.device("cpu")
            else:
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")

            # Create training configuration dictionary
            config_dict = {
                "device": "cuda" if self.device_gpu.isChecked() else "cpu",
                "data_root": self.data_root_edit.text(),
                "num_classes": self.num_classes_spin.value(),
                "selected_dataset": selected_dataset,
                "selected_model": actual_model,
                "train": {
                    "n_epochs": self.epochs_spin.value(),
                    "batch_size": self.batch_size_spin.value(),
                    "lr": float(self.lr_spin.value()),
                    "early_stopping_patience": self.patience_spin.value(),
                    "output_dir": os.path.abspath(self.output_dir_edit.text()),
                }
            }

            # Add pretraining configuration if Autoencoder is selected
            if selected_model == "Autoencoder":
                config_dict["train"].update({
                    "use_pretrain": self.enable_pretrain_check.isChecked() if hasattr(self, 'enable_pretrain_check') else True,
                    "pretrain_epochs": self.pretrain_epochs_spin.value() if hasattr(self, 'pretrain_epochs_spin') else 5,
                    "pretrain_lr": float(self.pretrain_lr_spin.value()) if hasattr(self, 'pretrain_lr_spin') else 0.001,
                })

            # Save configuration to temporary file
            config_path = "temp_train_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            # Create Configs object
            config = Configs(config_path=config_path)

            # Disable UI during training
            self.set_ui_enabled(False)

            # Create training controller and thread
            self.training_controller = TrainingController()
            self.thread = TrainingThread(config, self.training_controller)

            # Connect signals
            self.thread.finished.connect(self.training_finished)
            self.thread.terminated.connect(lambda: self.set_ui_enabled(True))
            self.thread.progress_update.connect(self.update_progress)
            self.thread.best_model_update.connect(self.update_best_model)
            self.thread.start()
        except Exception as e:
            print(f"Training error: {e}")
            self.set_ui_enabled(True)
            QMessageBox.warning(
                self, "Error", "Training failed. Check console for details."
            )

    def update_progress(self, current, total, epoch):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)

        # 根据epoch值判断是预训练还是分类训练
        if epoch < 0:  # 负值表示预训练阶段
            pretrain_epoch = -epoch
            self.progress_label.setText(
                f"Pretraining - Epoch {pretrain_epoch} - {current}/{total} iterations"
            )
        else:  # 正值表示分类训练阶段
            self.progress_label.setText(
                f"Classification - Epoch {epoch} - {current}/{total} iterations"
            )

    def update_best_model(self, accuracy, epoch):
        self.progress_bar.setStyleSheet(self.normal_style)
        self.progress_label.setText(
            f"Best model from epoch {epoch+1} (acc: {accuracy:.2%}) will be saved as latest.pth"
        )

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
            message = "Training completed successfully!\n"
            message += f"Best model (acc: {self.thread.best_accuracy:.2%} at epoch {self.thread.best_epoch+1}) "
            message += "has been saved as latest.pth and in timestamped directory."
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(
                self, "Error", "Training failed. Check console for details."
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingApp()
    window.show()
    sys.exit(app.exec_())
