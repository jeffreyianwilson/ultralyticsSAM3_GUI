"""Worker helpers for background GUI tasks."""

from __future__ import annotations

import inspect
import threading
import traceback

from PySide6 import QtCore

from .exceptions import InferenceCancelledError


class WorkerSignals(QtCore.QObject):
    """Signals emitted by background tasks."""

    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    cancelled = QtCore.Signal(str)
    finished = QtCore.Signal()
    progress = QtCore.Signal(int, int, str)
    item_started = QtCore.Signal(int, int, str)
    item_result = QtCore.Signal(int, int, object, str)


class BackendTask(QtCore.QRunnable):
    """QRunnable wrapper that executes backend jobs off the UI thread."""

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.cancel_event = threading.Event()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            callback_kwargs = {
                "progress_callback": self._emit_progress,
                "cancel_callback": self.cancel_event.is_set,
            }
            if self._supports_callback_kwarg("item_start_callback"):
                callback_kwargs["item_start_callback"] = self._emit_item_started
            if self._supports_callback_kwarg("item_result_callback"):
                callback_kwargs["item_result_callback"] = self._emit_item_result
            result = self.fn(*self.args, **callback_kwargs, **self.kwargs)
        except InferenceCancelledError as exc:
            self.signals.cancelled.emit(str(exc) or "Operation cancelled.")
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

    def cancel(self) -> None:
        """Request task cancellation."""
        self.cancel_event.set()

    def _emit_progress(self, current: int, total: int | None, message: str) -> None:
        self.signals.progress.emit(current, total or 0, message)

    def _emit_item_started(self, index: int, total: int | None, label: str) -> None:
        self.signals.item_started.emit(index, total or 0, label)

    def _emit_item_result(self, index: int, total: int | None, result: object, label: str) -> None:
        self.signals.item_result.emit(index, total or 0, result, label)

    def _supports_callback_kwarg(self, name: str) -> bool:
        try:
            signature = inspect.signature(self.fn)
        except (TypeError, ValueError):
            return False
        if name in signature.parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
