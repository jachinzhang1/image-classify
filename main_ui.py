import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget
from gui.train_ui import TrainingApp
from gui.test_ui import TestApp
from gui.single_image_ui import SingleImageApp


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Training & Testing Interface")
        self.setStyleSheet(
            """
            QWidget {
                font-family: 'Microsoft YaHei';
                font-size: 12pt;
            }
            QTabWidget::pane {
                border: 1px solid #D4D4D4;
                padding: 10px;
            }
            QTabBar::tab {
                padding: 8px 15px;
                font-size: 12pt;
            }
        """
        )
        self.setGeometry(400, 200, 1600, 1500)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()

        # Create instances of both UIs
        self.train_ui = TrainingApp()
        self.test_ui = TestApp()
        self.single_image_ui = SingleImageApp()

        # Remove their main window properties
        self.train_ui.setParent(None)
        self.test_ui.setParent(None)
        self.single_image_ui.setParent(None)

        # Add them as tabs
        self.tabs.addTab(self.train_ui.central_widget, "Training")
        self.tabs.addTab(self.test_ui.central_widget, "Dataset Test")
        self.tabs.addTab(self.single_image_ui.central_widget, "Single Image Test")

        layout.addWidget(self.tabs)
        self.central_widget.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
