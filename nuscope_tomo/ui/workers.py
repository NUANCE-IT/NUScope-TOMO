"""Run a blocking processing call (import / align / threshold / mesh) on a
background QThread so the UI stays responsive, with progress reported back
through a Qt signal.
"""
from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            kwargs = dict(self.kwargs)
            params = inspect.signature(self.fn).parameters
            if "progress_callback" in params:
                kwargs["progress_callback"] = lambda done, total: self.progress.emit(done, total)
            result = self.fn(*self.args, **kwargs)
            self.finished_ok.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
