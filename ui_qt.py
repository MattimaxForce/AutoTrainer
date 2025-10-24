from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QProgressBar, QComboBox, QTextEdit, QSpinBox, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QRect, QUrl
import sys
import threading
import json
from trainer_core import avvia_training

# import avvia_ui_web lazily to avoid circular imports at module import time
def _import_avvia_ui_web():
    try:
        from ui_web import avvia_ui_web
        return avvia_ui_web
    except Exception:
        return None

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    def __init__(self, model, dataset, epochs=1, batch_size=4):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        avvia_training(self.model, self.dataset, epochs=self.epochs, batch_size=self.batch_size, log_callback=self.log_signal.emit)

class AutotrainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autotrain by Mattia")
        self.setGeometry(100, 100, 420, 250)
        layout = QVBoxLayout()
        self.label = QLabel("Seleziona modello e dataset", alignment=Qt.AlignCenter)
        self.model_select = QComboBox()
        self.model_select.addItems(["distilbert-base-uncased", "bert-base-uncased"])
        self.dataset_select = QComboBox()
        self.dataset_select.addItems(["imdb", "ag_news"])
        self.progress = QProgressBar()
        # add simple controls for epochs and batch size
        controls_layout = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setMinimum(1)
        self.epochs_spin.setValue(1)
        self.epochs_spin.setPrefix("Epochs: ")
        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setValue(4)
        self.batch_spin.setPrefix("Batch: ")
        controls_layout.addWidget(self.epochs_spin)
        controls_layout.addWidget(self.batch_spin)

        self.start_button = QPushButton("Start Training")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.label)
        layout.addWidget(self.model_select)
        layout.addWidget(self.dataset_select)
        layout.addLayout(controls_layout)
        layout.addWidget(self.progress)
        layout.addWidget(self.start_button)
        layout.addWidget(self.log_output)
        self.setLayout(layout)
        self.start_button.clicked.connect(self.start_training)
        self.anim = QPropertyAnimation(self.progress, b"geometry")
        self.anim.setDuration(1000)
        self.anim.setStartValue(QRect(0, 0, 100, 30))
        self.anim.setEndValue(QRect(0, 0, 420, 30))
        self.anim.start()

    def start_training(self):
        model = self.model_select.currentText()
        dataset = self.dataset_select.currentText()
        epochs = int(self.epochs_spin.value())
        batch = int(self.batch_spin.value())
        self.thread = TrainingThread(model, dataset, epochs=epochs, batch_size=batch)
        self.thread.log_signal.connect(self.update_log)
        self.thread.start()

    def update_log(self, text):
        # parse PROGRESS:NN messages or generic logs
        if text.startswith("PROGRESS:"):
            try:
                val = int(text.split(":")[-1].replace("%", ""))
                self.progress.setValue(val)
            except Exception:
                pass
        else:
            # append colored log; strip color codes for display in QTextEdit
            clean = text
            self.log_output.append(clean)
        if "complet" in text.lower() or text == "TRAINING_COMPLETED":
            self.label.setText("Training terminato!")

def avvia_ui_qt():
    # Try to open the HTML UI inside a Qt WebEngine view if available.
    # Read host/port from config.json (fallback to defaults)
    host = "127.0.0.1"
    port = 7860
    try:
        with open("config.json", "r") as f:
            cfg = json.load(f)
            host = cfg.get("web_host", host)
            port = int(cfg.get("web_port", port))
    except Exception:
        pass

    avvia_ui_web = _import_avvia_ui_web()

    # Prefer to embed the HTML UI if Qt WebEngine is available
    try:
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        # Start the web server in background (do not open external browser)
        if avvia_ui_web:
            avvia_ui_web(host, port, open_browser=False, start_in_thread=True)

        app = QApplication(sys.argv)
        view = QWebEngineView()
        url = f"http://{host}:{port}/"
        view.setWindowTitle("Autotrain - Web UI")
        view.resize(1200, 800)
        view.load(QUrl(url))
        view.show()
        sys.exit(app.exec_())
    except Exception:
        # Fallback: show the built-in Qt widget UI
        app = QApplication(sys.argv)
        win = AutotrainUI()
        win.show()
        sys.exit(app.exec_())
