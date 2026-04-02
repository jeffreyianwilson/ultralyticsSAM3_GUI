"""Custom GUI widgets."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class PreviewCanvas(QtWidgets.QWidget):
    """Compact preview canvas with point, box, and manual mask tools."""

    point_added = QtCore.Signal(float, float, int)
    box_added = QtCore.Signal(float, float, float, float, int)
    manual_mask_changed = QtCore.Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 420)
        self.setMouseTracking(True)
        self._image = None
        self._pixmap = None
        self._tool = "none"
        self._points: list[tuple[float, float, int]] = []
        self._boxes: list[tuple[float, float, float, float, int]] = []
        self._prompt_mask_preview: np.ndarray | None = None
        self._manual_mask_preview: np.ndarray | None = None
        self._manual_mask_editable = False
        self._brush_radius = 14
        self._drag_start = None
        self._drag_current = None
        self._drawing_manual_mask = False
        self._manual_mask_path: list[tuple[float, float]] = []
        self._painting_manual_mask = False
        self._manual_paint_mode = "add"
        self._last_paint_point: tuple[float, float] | None = None

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        if tool != "manual_mask":
            self._drawing_manual_mask = False
            self._manual_mask_path = []
            self._painting_manual_mask = False
            self._last_paint_point = None
            self._manual_mask_editable = False
        self.update()

    def set_brush_radius(self, radius: int) -> None:
        self._brush_radius = max(1, int(radius))
        self.update()

    def set_image(self, image_bgr: np.ndarray | None) -> None:
        self._image = image_bgr
        if self._image is None:
            self._pixmap = None
        else:
            rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            bytes_per_line = channels * width
            qimage = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            self._pixmap = QtGui.QPixmap.fromImage(qimage)
            if self._prompt_mask_preview is not None and self._prompt_mask_preview.shape != self._image.shape[:2]:
                resized = cv2.resize(
                    self._prompt_mask_preview.astype(np.uint8, copy=False),
                    (self._image.shape[1], self._image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                self._prompt_mask_preview = resized > 0
            if self._manual_mask_preview is not None and self._manual_mask_preview.shape != self._image.shape[:2]:
                resized = cv2.resize(
                    self._manual_mask_preview.astype(np.uint8, copy=False),
                    (self._image.shape[1], self._image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                self._manual_mask_preview = resized > 0
        self.update()

    def set_prompt_overlays(
        self,
        points: list[tuple[float, float, int]],
        boxes: list[tuple[float, float, float, float, int]],
    ) -> None:
        self._points = points
        self._boxes = boxes
        self.update()

    def set_prompt_mask_preview(self, mask: np.ndarray | None) -> None:
        if mask is None:
            self._prompt_mask_preview = None
        else:
            array = np.asarray(mask)
            self._prompt_mask_preview = array if array.dtype == np.bool_ else (array > 0)
        self.update()

    def set_manual_mask_preview(self, mask: np.ndarray | None) -> None:
        if mask is None:
            self._manual_mask_preview = None
        else:
            array = np.asarray(mask)
            self._manual_mask_preview = array if array.dtype == np.bool_ else (array > 0)
        self._manual_mask_editable = False
        self.update()

    def set_mask_preview(self, mask: np.ndarray | None) -> None:
        self.set_prompt_mask_preview(mask)
        self.update()

    def clear(self) -> None:
        self._image = None
        self._pixmap = None
        self._points = []
        self._boxes = []
        self._prompt_mask_preview = None
        self._manual_mask_preview = None
        self._drag_start = None
        self._drag_current = None
        self._drawing_manual_mask = False
        self._manual_mask_path = []
        self._painting_manual_mask = False
        self._last_paint_point = None
        self._manual_mask_editable = False
        self.update()

    def _image_rect(self) -> QtCore.QRectF | None:
        if self._pixmap is None:
            return None
        target = QtCore.QRectF(self.rect())
        scaled = self._pixmap.size().scaled(target.size().toSize(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        x = (target.width() - scaled.width()) / 2.0
        y = (target.height() - scaled.height()) / 2.0
        return QtCore.QRectF(x, y, scaled.width(), scaled.height())

    def _widget_to_image(self, pos: QtCore.QPointF) -> tuple[float, float] | None:
        rect = self._image_rect()
        if rect is None or self._pixmap is None or not rect.contains(pos):
            return None
        x = (pos.x() - rect.x()) * self._pixmap.width() / rect.width()
        y = (pos.y() - rect.y()) * self._pixmap.height() / rect.height()
        return (float(x), float(y))

    def _image_to_widget(self, x: float, y: float) -> QtCore.QPointF | None:
        rect = self._image_rect()
        if rect is None or self._pixmap is None:
            return None
        px = rect.x() + x * rect.width() / self._pixmap.width()
        py = rect.y() + y * rect.height() / self._pixmap.height()
        return QtCore.QPointF(px, py)

    def _ensure_editable_manual_mask(self) -> bool:
        if self._image is None:
            return False
        target_shape = self._image.shape[:2]
        if self._manual_mask_preview is None:
            self._manual_mask_preview = np.zeros(target_shape, dtype=np.bool_)
            self._manual_mask_editable = True
            return True
        if self._manual_mask_preview.shape != target_shape:
            resized = cv2.resize(
                self._manual_mask_preview.astype(np.uint8, copy=False),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            self._manual_mask_preview = resized > 0
            self._manual_mask_editable = True
            return True
        if not self._manual_mask_editable:
            self._manual_mask_preview = np.array(self._manual_mask_preview, dtype=np.bool_, copy=True)
            self._manual_mask_editable = True
        return True

    def _append_manual_mask_point(self, mapped: tuple[float, float] | None) -> None:
        if mapped is None:
            return
        if not self._manual_mask_path:
            self._manual_mask_path.append(mapped)
            return
        last_x, last_y = self._manual_mask_path[-1]
        if math.hypot(mapped[0] - last_x, mapped[1] - last_y) >= 1.5:
            self._manual_mask_path.append(mapped)

    def _paint_manual_segment(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        if not self._ensure_editable_manual_mask():
            return
        # Brush painting is applied to a temporary stroke mask first so add/remove
        # operations stay simple and deterministic on the stored manual mask layer.
        stroke = np.zeros_like(self._manual_mask_preview, dtype=np.uint8)
        start_pt = (int(round(start[0])), int(round(start[1])))
        end_pt = (int(round(end[0])), int(round(end[1])))
        cv2.line(stroke, start_pt, end_pt, 1, thickness=max(1, int(self._brush_radius * 2)))
        cv2.circle(stroke, end_pt, max(1, int(self._brush_radius)), 1, thickness=-1)
        if self._manual_paint_mode == "erase":
            self._manual_mask_preview[stroke > 0] = False
        else:
            self._manual_mask_preview[stroke > 0] = True

    def _commit_manual_mask_path(self) -> None:
        if not self._drawing_manual_mask or len(self._manual_mask_path) < 3 or not self._ensure_editable_manual_mask():
            self._drawing_manual_mask = False
            self._manual_mask_path = []
            self.update()
            return
        polygon = np.array([[int(round(x)), int(round(y))] for x, y in self._manual_mask_path], dtype=np.int32)
        fill = np.zeros_like(self._manual_mask_preview, dtype=np.uint8)
        cv2.fillPoly(fill, [polygon], 1)
        self._manual_mask_preview[fill > 0] = True
        self._drawing_manual_mask = False
        self._manual_mask_path = []
        self.manual_mask_changed.emit(self._manual_mask_preview.copy())
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        mapped = self._widget_to_image(event.position())
        if mapped is None:
            return
        if self._tool == "point":
            self.point_added.emit(mapped[0], mapped[1], 1)
        elif self._tool == "box":
            self._drag_start = mapped
            self._drag_current = mapped
            self.update()
        elif self._tool == "manual_mask" and event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Manual mask mode supports polygon creation by default, with
            # Shift/Ctrl converting the same tool into additive/subtractive brush edits.
            modifiers = event.modifiers()
            if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                self._painting_manual_mask = True
                self._manual_paint_mode = "erase"
                self._last_paint_point = mapped
                self._paint_manual_segment(mapped, mapped)
                self.update()
                return
            if modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self._painting_manual_mask = True
                self._manual_paint_mode = "add"
                self._last_paint_point = mapped
                self._paint_manual_segment(mapped, mapped)
                self.update()
                return
            self._drawing_manual_mask = True
            self._manual_mask_path = []
            self._append_manual_mask_point(mapped)
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        mapped = self._widget_to_image(event.position())
        if self._tool == "box" and self._drag_start is not None:
            if mapped is not None:
                self._drag_current = mapped
                self.update()
        elif self._tool == "manual_mask" and self._painting_manual_mask and mapped is not None and self._last_paint_point is not None:
            self._paint_manual_segment(self._last_paint_point, mapped)
            self._last_paint_point = mapped
            self.update()
        elif self._tool == "manual_mask" and self._drawing_manual_mask:
            self._append_manual_mask_point(mapped)
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._tool == "box" and self._drag_start is not None:
            mapped = self._widget_to_image(event.position())
            if mapped is not None:
                x1, y1 = self._drag_start
                x2, y2 = mapped
                self.box_added.emit(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), 1)
            self._drag_start = None
            self._drag_current = None
            self.update()
            return
        if self._tool == "manual_mask" and self._painting_manual_mask:
            mapped = self._widget_to_image(event.position())
            if mapped is not None and self._last_paint_point is not None:
                self._paint_manual_segment(self._last_paint_point, mapped)
            self._painting_manual_mask = False
            self._last_paint_point = None
            if self._manual_mask_preview is not None:
                self.manual_mask_changed.emit(self._manual_mask_preview.copy())
            self.update()
            return
        if self._tool == "manual_mask":
            self._append_manual_mask_point(self._widget_to_image(event.position()))
            self._commit_manual_mask_path()

    def _draw_mask_overlay(self, painter: QtGui.QPainter, rect: QtCore.QRectF, mask: np.ndarray | None, color: tuple[int, int, int], alpha_scale: int) -> None:
        if mask is None or self._pixmap is None:
            return
        resized = cv2.resize(np.asarray(mask, dtype=np.uint8), (self._pixmap.width(), self._pixmap.height()), interpolation=cv2.INTER_NEAREST)
        alpha = np.where(resized > 0, alpha_scale, 0).astype(np.uint8, copy=False)
        rgba = np.zeros((resized.shape[0], resized.shape[1], 4), dtype=np.uint8)
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = alpha
        qimage = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.shape[1] * 4,
            QtGui.QImage.Format.Format_RGBA8888,
        )
        painter.drawImage(rect, qimage)

    def _draw_manual_mask_path(self, painter: QtGui.QPainter) -> None:
        if self._pixmap is None or len(self._manual_mask_path) < 2:
            return
        color = QtGui.QColor("#ffd666")
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(color, max(2, self._brush_radius // 3), QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin))
        polygon = QtGui.QPolygonF()
        for x, y in self._manual_mask_path:
            point = self._image_to_widget(x, y)
            if point is not None:
                polygon.append(point)
        if polygon.size() >= 2:
            painter.drawPolyline(polygon)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#161616"))
        rect = self._image_rect()
        if self._pixmap is not None and rect is not None:
            painter.drawPixmap(rect.toRect(), self._pixmap)
            self._draw_mask_overlay(painter, rect, self._prompt_mask_preview, (20, 173, 255), 110)
            self._draw_mask_overlay(painter, rect, self._manual_mask_preview, (0, 210, 120), 135)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self._draw_manual_mask_path(painter)
            for x, y, label in self._points:
                point = self._image_to_widget(x, y)
                if point is None:
                    continue
                color = QtGui.QColor("#52c41a") if label == 1 else QtGui.QColor("#ff4d4f")
                painter.setBrush(color)
                painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1.5))
                painter.drawEllipse(point, 5.0, 5.0)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor("#40a9ff"), 2))
            for x1, y1, x2, y2, _label in self._boxes:
                p1 = self._image_to_widget(x1, y1)
                p2 = self._image_to_widget(x2, y2)
                if p1 is None or p2 is None:
                    continue
                painter.drawRect(QtCore.QRectF(p1, p2).normalized())
            if self._drag_start is not None and self._drag_current is not None:
                p1 = self._image_to_widget(*self._drag_start)
                p2 = self._image_to_widget(*self._drag_current)
                if p1 is not None and p2 is not None:
                    painter.setPen(QtGui.QPen(QtGui.QColor("#ffd666"), 2, QtCore.Qt.PenStyle.DashLine))
                    painter.drawRect(QtCore.QRectF(p1, p2).normalized())
            if self._manual_mask_preview is not None and np.any(self._manual_mask_preview):
                painter.setPen(QtGui.QPen(QtGui.QColor("#00d278"), 2))
                painter.drawText(rect.adjusted(10, 18, -10, -10), "manualMask id=0")
        else:
            painter.setPen(QtGui.QColor("#8c8c8c"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Open an image, folder, or video to begin")
