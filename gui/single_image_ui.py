from single_image_test import classify_image, class_dict
from configs import Configs
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QComboBox,
)
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SingleImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single Image Classifier")
        self.setGeometry(100, 100, 800, 600)
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
            QLineEdit {
                min-height: 30px;
                font-size: 12pt;
            }
            .error {
                color: red;
            }
        """
        )

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.init_ui()
        self.config = Configs(config_path="options.yaml")

        # 初始化数据集选择器
        self.load_datasets()

    def init_ui(self):
        layout = QVBoxLayout()

        # Model settings
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("<b>Model Settings</b>"))

        # Add dataset selection dropdown
        self.dataset_combo = QComboBox()
        model_layout.addWidget(QLabel("Dataset:"))
        model_layout.addWidget(self.dataset_combo)

        # Add dataset description label
        self.dataset_desc_label = QLabel("")
        model_layout.addWidget(self.dataset_desc_label)

        # Add model type selection
        self.model_combo = QComboBox()
        model_layout.addWidget(QLabel("Model Type:"))
        model_layout.addWidget(self.model_combo)

        self.ckpt_path_edit = QLineEdit()
        model_layout.addWidget(QLabel("Checkpoint Path:"))
        model_layout.addWidget(self.ckpt_path_edit)
        browse_ckpt_btn = QPushButton("Browse...")
        browse_ckpt_btn.clicked.connect(self.browse_ckpt_file)
        model_layout.addWidget(browse_ckpt_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Image selection
        image_group = QWidget()
        image_layout = QVBoxLayout()
        image_layout.addWidget(QLabel("<b>Image Selection</b>"))

        self.image_path_edit = QLineEdit()
        image_layout.addWidget(QLabel("Image Path:"))
        image_layout.addWidget(self.image_path_edit)
        browse_image_btn = QPushButton("Browse...")
        browse_image_btn.clicked.connect(self.browse_image_file)
        image_layout.addWidget(browse_image_btn)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 700)
        self.image_label.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.image_label)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Status and result
        self.status_label = QLabel("Please select model and image")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_label)

        # Control buttons
        self.test_btn = QPushButton("Start Testing")
        self.test_btn.clicked.connect(self.start_testing)
        layout.addWidget(self.test_btn)

        self.central_widget.setLayout(layout)

    def load_datasets(self):
        """Load available datasets"""
        self.dataset_combo.clear()
        datasets = self.config.datasets
        if datasets:
            self.dataset_combo.addItems(list(datasets.keys()))
            # Set current selected dataset
            selected_dataset = self.config.selected_dataset
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

        # Load model types
        self.model_combo.clear()
        self.model_combo.addItems(self.config.model_types)

        # Set selected model
        selected_model = self.config.selected_model
        index = self.model_combo.findText(selected_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        # Connect model change signal
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        # Update the checkpoint path
        self.update_checkpoint_path()

    def on_dataset_changed(self, dataset_name):
        """Handle dataset selection change"""
        if dataset_name in self.config.datasets:
            dataset_config = self.config.datasets[dataset_name]
            # Update description
            self.dataset_desc_label.setText(
                dataset_config.get("description", ""))
            # Update configuration
            self.config.update_dataset(dataset_name)
            # Update checkpoint path
            self.update_checkpoint_path()

    def on_model_changed(self, model_name):
        """Handle model type selection change"""
        # Update selected model in config
        self.config.selected_model = model_name
        # Update checkpoint path
        self.update_checkpoint_path()

    def update_checkpoint_path(self):
        """Update checkpoint path based on dataset and model selections"""
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()

        if dataset and model:
            # Set path to latest model for the selected dataset and model
            ckpt_path = os.path.join(
                self.config.training_config["output_dir"],
                dataset,
                model,
                "latest.pth"
            )
            # 将反斜杠替换为正斜杠以保持一致性
            ckpt_path = ckpt_path.replace("\\", "/")
            self.ckpt_path_edit.setText(ckpt_path)
            self.update_status()

    def browse_ckpt_file(self):
        # Start in the directory for the currently selected dataset/model
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()
        start_dir = os.path.join(
            "./ckpts", dataset, model) if dataset and model else "./ckpts"
        # 将反斜杠替换为正斜杠以保持一致性
        start_dir = start_dir.replace("\\", "/")

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", start_dir, "Model Files (*.pth *.pt)"
        )
        if file_path:
            # 将反斜杠替换为正斜杠以保持一致性
            file_path = file_path.replace("\\", "/")
            self.ckpt_path_edit.setText(file_path)
            self.update_status()

    def browse_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "./images", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            # 将反斜杠替换为正斜杠以保持一致性
            file_path = file_path.replace("\\", "/")
            self.image_path_edit.setText(file_path)
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            )
            self.update_status()

    def update_status(self):
        ready = bool(self.ckpt_path_edit.text()) and bool(
            self.image_path_edit.text())
        if ready:
            self.status_label.setText("Ready to test")
            self.status_label.setStyleSheet("")
        else:
            self.status_label.setText("Please select model and image")
            self.status_label.setStyleSheet("color: red")

    def start_testing(self):
        if not (self.ckpt_path_edit.text() and self.image_path_edit.text()):
            self.status_label.setText(
                "Please complete model and image selection first!"
            )
            self.status_label.setStyleSheet("color: red")
            return

        # Update current selected dataset
        selected_dataset = self.dataset_combo.currentText()
        if selected_dataset in self.config.datasets:
            self.config.update_dataset(selected_dataset)

        # 获取检查点路径并统一斜杠
        ckpt_path = os.path.abspath(self.ckpt_path_edit.text())
        ckpt_path = ckpt_path.replace("\\", "/")

        # 获取图像路径并统一斜杠
        image_path = self.image_path_edit.text()
        image_path = image_path.replace("\\", "/")

        # Update config
        self.config.test_config["ckpt_path"] = ckpt_path

        try:
            class_idx = classify_image(
                self.config, image_path)
            self.result_label.setText(
                f"Predicted class: {class_dict[class_idx]}")
        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Classification failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SingleImageApp()
    window.show()
    sys.exit(app.exec_())
