from PySide6.QtCore import *
import numpy as np


class UpdateThread(QThread):
    update_signal = Signal(np.ndarray)

    def __init__(self, runner, slot):
        super().__init__()
        self.runner = runner
        self.update_signal.connect(slot)

    def run(self):
        self.runner(self.update_signal)
