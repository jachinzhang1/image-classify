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

        # 添加ResNet变体选择下拉框
        self.resnet_variant_group = QWidget()
        resnet_variant_layout = QVBoxLayout()
        resnet_variant_layout.addWidget(QLabel("ResNet Variant:"))
        self.resnet_variant_combo = QComboBox()
        resnet_variant_layout.addWidget(self.resnet_variant_combo)
        self.resnet_variant_group.setLayout(resnet_variant_layout)
        self.resnet_variant_group.setVisible(False)  # Default hidden
        model_layout.addWidget(self.resnet_variant_group)

        # VGG变体选择下拉框
        self.vgg_variant_group = QWidget()
        vgg_variant_layout = QVBoxLayout()
        vgg_variant_layout.addWidget(QLabel("VGG Variant:"))
        self.vgg_variant_combo = QComboBox()
        vgg_variant_layout.addWidget(self.vgg_variant_combo)
        self.vgg_variant_group.setLayout(vgg_variant_layout)
        self.vgg_variant_group.setVisible(False)  # Default hidden
        model_layout.addWidget(self.vgg_variant_group)
        
        # DenseNet变体选择下拉框
        self.densenet_variant_group = QWidget()
        densenet_variant_layout = QVBoxLayout()
        densenet_variant_layout.addWidget(QLabel("DenseNet Variant:"))
        self.densenet_variant_combo = QComboBox()
        densenet_variant_layout.addWidget(self.densenet_variant_combo)
        self.densenet_variant_group.setLayout(densenet_variant_layout)
        self.densenet_variant_group.setVisible(False)  # Default hidden
        model_layout.addWidget(self.densenet_variant_group)
        
        # EfficientNet变体选择下拉框
        self.efficientnet_variant_group = QWidget()
        efficientnet_variant_layout = QVBoxLayout()
        efficientnet_variant_layout.addWidget(QLabel("EfficientNet Variant:"))
        self.efficientnet_variant_combo = QComboBox()
        efficientnet_variant_layout.addWidget(self.efficientnet_variant_combo)
        self.efficientnet_variant_group.setLayout(efficientnet_variant_layout)
        self.efficientnet_variant_group.setVisible(False)  # Default hidden
        model_layout.addWidget(self.efficientnet_variant_group)

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
        self.image_label.setMinimumSize(300, 1000)
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
                        datasets[selected_dataset].get("description", "")
                    )

            # Connect dataset change signal
            self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)

        # Load model types
        self.model_combo.clear()
        self.model_combo.addItems(self.config.model_types)

        # Set selected model
        selected_model = self.config.selected_model
        index = self.model_combo.findText(selected_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

        # 加载ResNet变体
        self.resnet_variant_combo.clear()
        self.resnet_variant_combo.addItems(self.config.resnet_variants)

        # 设置选择的ResNet变体
        selected_variant = self.config.selected_resnet_variant
        index = self.resnet_variant_combo.findText(selected_variant)
        if index >= 0:
            self.resnet_variant_combo.setCurrentIndex(index)
            
        # 加载 VGG 变体
        self.vgg_variant_combo.clear()
        self.vgg_variant_combo.addItems(self.config.vgg_variants)
        
        # 设置选择的 VGG 变体
        selected_vgg_variant = self.config.selected_vgg_variant
        index = self.vgg_variant_combo.findText(selected_vgg_variant)
        if index >= 0:
            self.vgg_variant_combo.setCurrentIndex(index)
            
        # 加载 DenseNet 变体
        self.densenet_variant_combo.clear()
        self.densenet_variant_combo.addItems(self.config.densenet_variants)
        
        # 设置选择的 DenseNet 变体
        selected_densenet_variant = self.config.selected_densenet_variant
        index = self.densenet_variant_combo.findText(selected_densenet_variant)
        if index >= 0:
            self.densenet_variant_combo.setCurrentIndex(index)
            
        # 加载 EfficientNet 变体
        self.efficientnet_variant_combo.clear()
        self.efficientnet_variant_combo.addItems(self.config.efficientnet_variants)
        
        # 设置选择的 EfficientNet 变体
        selected_efficientnet_variant = self.config.selected_efficientnet_variant
        index = self.efficientnet_variant_combo.findText(selected_efficientnet_variant)
        if index >= 0:
            self.efficientnet_variant_combo.setCurrentIndex(index)

        # Connect model change signal
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        # Connect model variant change signals
        self.resnet_variant_combo.currentTextChanged.connect(
            self.on_model_variant_changed
        )
        self.vgg_variant_combo.currentTextChanged.connect(
            self.on_model_variant_changed
        )
        self.densenet_variant_combo.currentTextChanged.connect(
            self.on_model_variant_changed
        )
        self.efficientnet_variant_combo.currentTextChanged.connect(
            self.on_model_variant_changed
        )

        # 根据当前选择的模型显示或隐藏相应变体选择
        self.update_model_variant_visibility(selected_model)

        # Update the checkpoint path
        self.update_checkpoint_path()

    def on_dataset_changed(self, dataset_name):
        """Handle dataset selection change"""
        if dataset_name in self.config.datasets:
            dataset_config = self.config.datasets[dataset_name]
            # Update description
            self.dataset_desc_label.setText(dataset_config.get("description", ""))
            # Update configuration
            self.config.update_dataset(dataset_name)
            # Update checkpoint path
            self.update_checkpoint_path()

    def on_model_changed(self, model_name):
        """Handle model type selection change"""
        # Update ResNet variant selection visibility
        self.update_model_variant_visibility(model_name)
        # Update checkpoint path
        self.update_checkpoint_path()
        # Clear previous results
        self.clear_result()

    def on_model_variant_changed(self, variant_name):
        """Handle model variant selection change"""
        selected_model = self.model_combo.currentText()
        # Update checkpoint path based on the selected variant
        self.update_checkpoint_path()
        # Clear previous results
        self.clear_result()

    def update_model_variant_visibility(self, model_name):
        """Show or hide model variant selection based on model type"""
        self.resnet_variant_group.setVisible(model_name == "resnet")
        self.vgg_variant_group.setVisible(model_name == "vgg")
        self.densenet_variant_group.setVisible(model_name == "densenet")
        self.efficientnet_variant_group.setVisible(model_name == "efficientnet")

    def update_checkpoint_path(self):
        """Update checkpoint path based on selected dataset and model"""
        dataset = self.dataset_combo.currentText()
        model = self.model_combo.currentText()

        # Get the actual model name to use
        actual_model = model
        if model == "resnet":
            actual_model = self.resnet_variant_combo.currentText()
        elif model == "vgg":
            actual_model = self.vgg_variant_combo.currentText()
        elif model == "densenet":
            actual_model = self.densenet_variant_combo.currentText()
        elif model == "efficientnet":
            actual_model = self.efficientnet_variant_combo.currentText()
        # For autoencoder, we need to handle the checkpoint path differently
        elif model == "Autoencoder":
            actual_model = "autoencoder"

        ckpt_path = f"./ckpts/{dataset}/{actual_model}/latest.pth"
        self.ckpt_path_edit.setText(ckpt_path)
        self.update_status()

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
        ready = bool(self.ckpt_path_edit.text()) and bool(self.image_path_edit.text())
        if ready:
            self.status_label.setText("Ready to test")
            self.status_label.setStyleSheet("")
        else:
            self.status_label.setText("Please select model and image")
            self.status_label.setStyleSheet("color: red")

    def start_testing(self):
        """Start testing a single image"""
        try:
            # Check if image is selected
            image_path = self.image_path_edit.text()
            if not image_path or not os.path.isfile(image_path):
                self.status_label.setText("Please select a valid image file")
                return

            # Get selected model type
            selected_model = self.model_combo.currentText()

            # Get the actual model name to use
            actual_model = selected_model
            if selected_model == "resnet":
                actual_model = self.resnet_variant_combo.currentText()
            elif selected_model == "vgg":
                actual_model = self.vgg_variant_combo.currentText()
            elif selected_model == "densenet":
                actual_model = self.densenet_variant_combo.currentText()
            elif selected_model == "efficientnet":
                actual_model = self.efficientnet_variant_combo.currentText()

            # Update current selected dataset
            selected_dataset = self.dataset_combo.currentText()
            if selected_dataset in self.config.datasets:
                self.config.update_dataset(selected_dataset)

            # Update selected model in config
            self.config.selected_model = actual_model

            # Get checkpoint path and standardize slashes
            ckpt_path = os.path.abspath(self.ckpt_path_edit.text())
            ckpt_path = ckpt_path.replace("\\", "/")

            # Get image path and standardize slashes
            image_path = self.image_path_edit.text()
            image_path = image_path.replace("\\", "/")

            # Update config
            self.config.test_config["ckpt_path"] = ckpt_path

            class_idx = classify_image(self.config, image_path)
            self.result_label.setText(f"Predicted class: {class_dict[class_idx]}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Classification failed: {str(e)}")

    def clear_result(self):
        """Clear previous classification result"""
        self.result_label.setText("")
        self.status_label.setText("Please select model and image")
        self.status_label.setStyleSheet("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SingleImageApp()
    window.show()
    sys.exit(app.exec_())
