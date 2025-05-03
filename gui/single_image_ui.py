import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from configs import Configs
from single_image_test import classify_image, class_dict


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

    def init_ui(self):
        layout = QVBoxLayout()

        # Model settings
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("<b>Model Settings</b>"))

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

    def browse_ckpt_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint File", "./ckpts", "Model Files (*.pth *.pt)"
        )
        if file_path:
            self.ckpt_path_edit.setText(file_path)
            self.update_status()

    def browse_image_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "./images", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
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
        if not (self.ckpt_path_edit.text() and self.image_path_edit.text()):
            self.status_label.setText(
                "Please complete model and image selection first!"
            )
            self.status_label.setStyleSheet("color: red")
            return

        # Update config
        self.config.test_config["ckpt_path"] = os.path.abspath(
            self.ckpt_path_edit.text()
        )

        try:
            class_idx = classify_image(self.config, self.image_path_edit.text())
            self.result_label.setText(f"Predicted class: {class_dict[class_idx]}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Classification failed: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SingleImageApp()
    window.show()
    sys.exit(app.exec_())
