import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
)
from PyQt5.QtCore import QEvent
from gui.train_ui import TrainingApp
from gui.test_ui import TestApp
from gui.single_image_ui import SingleImageApp


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification Model Training & Testing Interface")
        self.setGeometry(400, 200, 1600, 1500)

        # load styles from CSS file
        self.load_styles()

        # Create scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.setCentralWidget(self.scroll)

        # Create container widget
        self.container = QWidget()
        self.scroll.setWidget(self.container)
        self.installEventFilter(self)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self.container)

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

    def load_styles(self):
        try:
            with open("styles.css", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("Warning: styles.css not found, using default styles")

    def eventFilter(self, obj, event):
        if event.type() == event.Wheel and isinstance(
            obj, (QSpinBox, QDoubleSpinBox, QComboBox)
        ):
            return True  # Block wheel events for numeric inputs
        return super().eventFilter(obj, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
