"""Main PySide6 application window."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .backend import SAM3Ultralytics
from .cache_store import CacheStore, load_cached_mask
from .gui_state import GUIState
from .gui_widgets import PreviewCanvas
from .gui_workers import BackendTask
from .io_utils import list_image_directory, normalize_mask_input, preview_mask, read_video_frame, to_bgr_image, video_frame_count
from .schemas import PredictionResult, SegmentationObject
from .visualization import render_overlay


class SAM3MainWindow(QtWidgets.QMainWindow):
    """Compact single-window desktop app for SAM 3 workflows."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("sam3_ultralytics")
        self.resize(1320, 860)
        self.state = GUIState()
        self.cache_store = CacheStore.create(Path.cwd() / ".sam3_cache")
        self.backend: SAM3Ultralytics | None = None
        self._backend_signature: tuple | None = None
        self.current_task: BackendTask | None = None
        self.current_task_mode: str | None = None
        self._streaming_batch_mode = False
        self._streaming_batch_total = 0
        self._sequence_run_context: dict | None = None
        self._manual_mask_clipboard: np.ndarray | None = None
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(180)
        self.play_timer.timeout.connect(self._advance_playback)
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._run_interaction_preview)
        self._build_ui()
        self._apply_defaults()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(420)
        splitter.addWidget(controls)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form = QtWidgets.QFormLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setEditable(True)
        self.model_browse_button = QtWidgets.QPushButton("Browse")
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(self.model_combo)
        model_row.addWidget(self.model_browse_button)
        form.addRow("Model", model_row)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cuda:0", "cpu"])
        form.addRow("Device", self.device_combo)

        self.cache_dir_edit = QtWidgets.QLineEdit()
        self.cache_browse_button = QtWidgets.QPushButton("Browse")
        cache_row = QtWidgets.QHBoxLayout()
        cache_row.addWidget(self.cache_dir_edit)
        cache_row.addWidget(self.cache_browse_button)
        form.addRow("Cache", cache_row)
        self.compact_cache_checkbox = QtWidgets.QCheckBox("Use compact cache archives (recommended)")
        self.compact_cache_checkbox.setChecked(True)
        form.addRow("", self.compact_cache_checkbox)
        self.clear_cache_button = QtWidgets.QPushButton("Clear Cache")
        form.addRow("", self.clear_cache_button)
        controls_layout.addLayout(form)

        source_row = QtWidgets.QGridLayout()
        self.open_image_button = QtWidgets.QPushButton("Open Image")
        self.open_directory_button = QtWidgets.QPushButton("Open Folder")
        self.open_video_button = QtWidgets.QPushButton("Open Video")
        source_row.addWidget(self.open_image_button, 0, 0)
        source_row.addWidget(self.open_directory_button, 0, 1)
        source_row.addWidget(self.open_video_button, 1, 0, 1, 2)
        controls_layout.addLayout(source_row)

        toolbox = QtWidgets.QToolBox()
        controls_layout.addWidget(toolbox)

        inference_page = QtWidgets.QWidget()
        inference_layout = QtWidgets.QVBoxLayout(inference_page)
        inference_form = QtWidgets.QFormLayout()
        self.run_scope_combo = QtWidgets.QComboBox()
        self.run_scope_combo.addItem("Current Image / Frame", "current")
        self.run_scope_combo.addItem("Entire Folder / Video", "all")
        self.confidence_spin = QtWidgets.QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)
        inference_form.addRow("Run Scope", self.run_scope_combo)
        inference_form.addRow("Confidence", self.confidence_spin)
        self.downscale_inference_checkbox = QtWidgets.QCheckBox("Downscale before inference")
        self.inference_scale_spin = QtWidgets.QDoubleSpinBox()
        self.inference_scale_spin.setRange(0.1, 1.0)
        self.inference_scale_spin.setDecimals(2)
        self.inference_scale_spin.setSingleStep(0.05)
        self.inference_scale_spin.setValue(1.0)
        self.inference_scale_spin.setEnabled(False)
        inference_form.addRow(self.downscale_inference_checkbox)
        inference_form.addRow("Scale Factor", self.inference_scale_spin)
        inference_layout.addLayout(inference_form)

        self.text_prompt_edit = QtWidgets.QLineEdit()
        self.text_prompt_edit.setPlaceholderText("person, bicycle")
        inference_layout.addWidget(QtWidgets.QLabel("Text Prompt"))
        inference_layout.addWidget(self.text_prompt_edit)

        mask_row = QtWidgets.QHBoxLayout()
        self.mask_button = QtWidgets.QPushButton("Load Mask")
        self.mask_clear_button = QtWidgets.QPushButton("Clear")
        mask_row.addWidget(self.mask_button)
        mask_row.addWidget(self.mask_clear_button)
        inference_layout.addLayout(mask_row)
        self.mask_label = QtWidgets.QLabel("No mask selected")
        self.mask_label.setWordWrap(True)
        inference_layout.addWidget(self.mask_label)
        self.mask_class_edit = QtWidgets.QLineEdit()
        self.mask_class_edit.setPlaceholderText("Optional target class")
        inference_layout.addWidget(QtWidgets.QLabel("Prompt Class"))
        inference_layout.addWidget(self.mask_class_edit)
        self.mask_id_edit = QtWidgets.QLineEdit()
        self.mask_id_edit.setPlaceholderText("Optional target ID")
        inference_layout.addWidget(QtWidgets.QLabel("Prompt ID"))
        inference_layout.addWidget(self.mask_id_edit)
        self.append_inference_checkbox = QtWidgets.QCheckBox("Append inferred masks")
        self.append_inference_checkbox.setChecked(False)
        inference_layout.addWidget(self.append_inference_checkbox)

        tool_row = QtWidgets.QGridLayout()
        self.point_tool_button = QtWidgets.QToolButton()
        self.point_tool_button.setText("Point Tool")
        self.point_tool_button.setCheckable(True)
        self.box_tool_button = QtWidgets.QToolButton()
        self.box_tool_button.setText("Box Tool")
        self.box_tool_button.setCheckable(True)
        tool_row.addWidget(self.point_tool_button, 0, 0)
        tool_row.addWidget(self.box_tool_button, 0, 1)
        inference_layout.addLayout(tool_row)

        clear_row = QtWidgets.QHBoxLayout()
        self.clear_prompts_button = QtWidgets.QPushButton("Clear Prompts")
        self.clear_all_button = QtWidgets.QPushButton("Clear All")
        clear_row.addWidget(self.clear_prompts_button)
        clear_row.addWidget(self.clear_all_button)
        inference_layout.addLayout(clear_row)
        self.prompt_summary_label = QtWidgets.QLabel("0 points, 0 boxes")
        inference_layout.addWidget(self.prompt_summary_label)
        toolbox.addItem(inference_page, "Inference")

        manual_page = QtWidgets.QWidget()
        manual_layout = QtWidgets.QVBoxLayout(manual_page)
        self.manual_mask_tool_button = QtWidgets.QToolButton()
        self.manual_mask_tool_button.setText("Manual Mask Tool")
        self.manual_mask_tool_button.setCheckable(True)
        manual_layout.addWidget(self.manual_mask_tool_button)
        self.manual_mask_label = QtWidgets.QLabel("No manual mask")
        self.manual_mask_label.setWordWrap(True)
        manual_layout.addWidget(self.manual_mask_label)
        self.manual_shortcut_label = QtWidgets.QLabel("Draw polygon: Left Mouse | Add: Shift+Left Mouse | Remove: Ctrl+Left Mouse")
        self.manual_shortcut_label.setWordWrap(True)
        manual_layout.addWidget(self.manual_shortcut_label)
        self.brush_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 80)
        self.brush_slider.setValue(14)
        self.brush_label = QtWidgets.QLabel("14 px")
        brush_row = QtWidgets.QHBoxLayout()
        brush_row.addWidget(QtWidgets.QLabel("Stroke"))
        brush_row.addWidget(self.brush_slider, stretch=1)
        brush_row.addWidget(self.brush_label)
        manual_layout.addLayout(brush_row)
        manual_button_row = QtWidgets.QGridLayout()
        self.copy_manual_mask_button = QtWidgets.QPushButton("Copy Mask")
        self.paste_manual_mask_button = QtWidgets.QPushButton("Paste To Current")
        self.copy_manual_mask_all_button = QtWidgets.QPushButton("Copy To All Frames")
        self.clear_manual_mask_button = QtWidgets.QPushButton("Clear Manual Mask")
        manual_button_row.addWidget(self.copy_manual_mask_button, 0, 0)
        manual_button_row.addWidget(self.paste_manual_mask_button, 0, 1)
        manual_button_row.addWidget(self.copy_manual_mask_all_button, 1, 0)
        manual_button_row.addWidget(self.clear_manual_mask_button, 1, 1)
        manual_layout.addLayout(manual_button_row)
        toolbox.addItem(manual_page, "Manual Masks")

        export_page = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(export_page)
        export_dir_row = QtWidgets.QHBoxLayout()
        self.export_dir_edit = QtWidgets.QLineEdit()
        self.export_dir_browse_button = QtWidgets.QPushButton("Browse")
        export_dir_row.addWidget(self.export_dir_edit)
        export_dir_row.addWidget(self.export_dir_browse_button)
        export_layout.addWidget(QtWidgets.QLabel("Export Directory"))
        export_layout.addLayout(export_dir_row)
        self.auto_export_masks_checkbox = QtWidgets.QCheckBox("Auto-export masks after inference")
        self.merge_masks_only_checkbox = QtWidgets.QCheckBox("Export merged masks only")
        self.invert_mask_export_checkbox = QtWidgets.QCheckBox("Invert exported masks")
        self.export_dilation_spin = QtWidgets.QSpinBox()
        self.export_dilation_spin.setRange(0, 256)
        self.export_dilation_spin.setValue(0)
        self.export_masks_button = QtWidgets.QPushButton("Export Masks")
        export_layout.addWidget(self.auto_export_masks_checkbox)
        export_layout.addWidget(self.merge_masks_only_checkbox)
        export_layout.addWidget(self.invert_mask_export_checkbox)
        export_layout.addWidget(QtWidgets.QLabel("Mask Dilation (px)"))
        export_layout.addWidget(self.export_dilation_spin)
        export_layout.addWidget(self.export_masks_button)
        toolbox.addItem(export_page, "Export")

        view_page = QtWidgets.QWidget()
        view_layout = QtWidgets.QFormLayout(view_page)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.show_labels_checkbox = QtWidgets.QCheckBox("Labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_masks_checkbox = QtWidgets.QCheckBox("Masks")
        self.show_masks_checkbox.setChecked(True)
        self.show_track_ids_checkbox = QtWidgets.QCheckBox("Track IDs")
        self.show_track_ids_checkbox.setChecked(True)
        self.filter_class_combo = QtWidgets.QComboBox()
        self.filter_id_combo = QtWidgets.QComboBox()
        self.filter_class_combo.addItem("All classes", None)
        self.filter_id_combo.addItem("All IDs", None)
        view_layout.addRow("Overlay Opacity", self.opacity_slider)
        view_layout.addRow(self.show_labels_checkbox)
        view_layout.addRow(self.show_masks_checkbox)
        view_layout.addRow(self.show_track_ids_checkbox)
        view_layout.addRow("Filter Class", self.filter_class_combo)
        view_layout.addRow("Filter ID", self.filter_id_combo)
        toolbox.addItem(view_page, "View")

        action_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.clear_masks_button = QtWidgets.QPushButton("Clear Masks")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.clear_masks_button)
        action_row.addWidget(self.cancel_button)
        controls_layout.addLayout(action_row)
        controls_layout.addStretch(1)

        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_canvas = PreviewCanvas()
        right_layout.addWidget(self.preview_canvas, stretch=1)

        playback_row = QtWidgets.QHBoxLayout()
        self.step_back_button = QtWidgets.QPushButton("Prev")
        self.step_back_button.setEnabled(False)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setEnabled(False)
        self.step_forward_button = QtWidgets.QPushButton("Next")
        self.step_forward_button.setEnabled(False)
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)
        self.frame_label = QtWidgets.QLabel("Frame 0/0")
        playback_row.addWidget(self.step_back_button)
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.step_forward_button)
        playback_row.addWidget(self.seek_slider, stretch=1)
        playback_row.addWidget(self.frame_label)
        right_layout.addLayout(playback_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        right_layout.addWidget(self.progress_bar)

        self.result_summary_label = QtWidgets.QLabel("No inference results yet.")
        self.result_summary_label.setWordWrap(True)
        right_layout.addWidget(self.result_summary_label)

        self.result_panel = QtWidgets.QPlainTextEdit()
        self.result_panel.setReadOnly(True)
        self.result_panel.setMaximumBlockCount(500)
        self.result_panel.setMinimumHeight(150)
        right_layout.addWidget(self.result_panel)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+X"), self, activated=self._clear_all)

        self.statusBar().showMessage("Ready")

        self.model_browse_button.clicked.connect(self._browse_model)
        self.cache_browse_button.clicked.connect(self._browse_cache_dir)
        self.clear_cache_button.clicked.connect(self._clear_cache_dir)
        self.compact_cache_checkbox.toggled.connect(self._set_compact_cache_mode)
        self.downscale_inference_checkbox.toggled.connect(self._toggle_inference_scale)
        self.open_image_button.clicked.connect(self._open_image)
        self.open_directory_button.clicked.connect(self._open_directory)
        self.open_video_button.clicked.connect(self._open_video)
        self.mask_button.clicked.connect(self._open_mask)
        self.mask_clear_button.clicked.connect(self._clear_mask)
        self.mask_class_edit.editingFinished.connect(self._store_current_prompt_target)
        self.mask_id_edit.editingFinished.connect(self._store_current_prompt_target)
        self.export_dir_browse_button.clicked.connect(self._browse_export_dir)
        self.export_masks_button.clicked.connect(self._export_masks_only)
        self.run_button.clicked.connect(self._run_inference)
        self.clear_masks_button.clicked.connect(self._clear_masks_only)
        self.cancel_button.clicked.connect(self._cancel_task)
        self.clear_prompts_button.clicked.connect(self._clear_prompts)
        self.clear_all_button.clicked.connect(self._clear_all)
        self.preview_canvas.point_added.connect(self._add_point)
        self.preview_canvas.box_added.connect(self._add_box)
        self.preview_canvas.manual_mask_changed.connect(self._set_manual_mask)
        self.step_back_button.clicked.connect(lambda: self._step_sequence(-1))
        self.play_button.clicked.connect(self._toggle_playback)
        self.step_forward_button.clicked.connect(lambda: self._step_sequence(1))
        self.seek_slider.valueChanged.connect(self._display_current_result)

        self.point_tool_button.toggled.connect(lambda checked: self._set_tool("point" if checked else "none"))
        self.box_tool_button.toggled.connect(lambda checked: self._set_tool("box" if checked else "none"))
        self.manual_mask_tool_button.toggled.connect(lambda checked: self._set_tool("manual_mask" if checked else "none"))
        self.brush_slider.valueChanged.connect(self._set_brush_size)
        self.copy_manual_mask_button.clicked.connect(self._copy_manual_mask_to_clipboard)
        self.paste_manual_mask_button.clicked.connect(self._paste_manual_mask_to_current_frame)
        self.copy_manual_mask_all_button.clicked.connect(self._copy_manual_mask_to_all_frames)
        self.clear_manual_mask_button.clicked.connect(self._clear_current_manual_mask)
        for widget in [
            self.opacity_slider,
            self.show_labels_checkbox,
            self.show_masks_checkbox,
            self.show_track_ids_checkbox,
        ]:
            signal = widget.valueChanged if isinstance(widget, QtWidgets.QSlider) else widget.toggled
            signal.connect(self._refresh_preview)
        self.filter_class_combo.currentIndexChanged.connect(self._refresh_preview)
        self.filter_id_combo.currentIndexChanged.connect(self._refresh_preview)

    def _apply_defaults(self) -> None:
        self.model_combo.addItem(r"D:\cache\models\sam3.pt")
        self.state.cache_dir = str(self.cache_store.root)
        self.state.compact_cache_enabled = self.cache_store.compact_archives
        self.cache_dir_edit.setText(str(self.cache_store.root))
        self.compact_cache_checkbox.setChecked(self.state.compact_cache_enabled)
        self.state.inference_scale_enabled = False
        self.state.inference_scale = 1.0
        self.downscale_inference_checkbox.setChecked(False)
        self.inference_scale_spin.setValue(1.0)
        self.opacity_slider.setValue(45)
        self._set_brush_size(self.brush_slider.value())
        self._refresh_view_filters()
        self._reset_progress()

    def _set_result_summary(self, text: str) -> None:
        self.result_summary_label.setText(text)

    def _iter_result_objects(self):
        results = self.state.results
        if results is None:
            return []
        items = results if isinstance(results, list) else [results]
        objects = []
        for result in items:
            if result is None:
                continue
            objects.extend(result.objects)
        return objects

    def _refresh_view_filters(self) -> None:
        selected_class = self.filter_class_combo.currentData()
        selected_id = self.filter_id_combo.currentData()
        classes = sorted({str(obj.label) for obj in self._iter_result_objects() if obj.label})
        ids = sorted({int(obj.track_id) for obj in self._iter_result_objects() if obj.track_id is not None})
        if self.state.mask_class:
            classes = sorted(set(classes) | {self.state.mask_class})
        if self.state.mask_id is not None:
            ids = sorted(set(ids) | {int(self.state.mask_id)})
        if self.state.manual_mask_input is not None or self.state.manual_masks_by_key:
            classes = sorted(set(classes) | {"manualMask"})
            ids = sorted(set(ids) | {0})

        self.filter_class_combo.blockSignals(True)
        self.filter_class_combo.clear()
        self.filter_class_combo.addItem("All classes", None)
        for value in classes:
            self.filter_class_combo.addItem(value, value)
        class_index = self.filter_class_combo.findData(selected_class)
        self.filter_class_combo.setCurrentIndex(class_index if class_index >= 0 else 0)
        self.filter_class_combo.blockSignals(False)

        self.filter_id_combo.blockSignals(True)
        self.filter_id_combo.clear()
        self.filter_id_combo.addItem("All IDs", None)
        for value in ids:
            self.filter_id_combo.addItem(str(value), value)
        id_index = self.filter_id_combo.findData(selected_id)
        self.filter_id_combo.setCurrentIndex(id_index if id_index >= 0 else 0)
        self.filter_id_combo.blockSignals(False)

    def _selected_view_track_ids(self) -> set[int] | None:
        value = self.filter_id_combo.currentData()
        if value is None:
            return None
        return {int(value)}

    def _selected_view_labels(self) -> set[str] | None:
        value = self.filter_class_combo.currentData()
        if value is None:
            return None
        return {str(value)}

    def _mask_visible_for_filters(self) -> bool:
        visible_ids = self._selected_view_track_ids()
        visible_labels = self._selected_view_labels()
        if visible_ids is not None and self.state.mask_id not in visible_ids:
            return False
        if visible_labels is not None and self.state.mask_class not in visible_labels:
            return False
        return True

    def _manual_mask_visible_for_filters(self) -> bool:
        visible_ids = self._selected_view_track_ids()
        visible_labels = self._selected_view_labels()
        if visible_ids is not None and 0 not in visible_ids:
            return False
        if visible_labels is not None and "manualMask" not in visible_labels:
            return False
        return True

    def _allocate_mask_id(self) -> int:
        mask_id = int(self.state.next_mask_id)
        self.state.next_mask_id += 1
        return mask_id

    def _ensure_current_mask_id(self) -> int | None:
        key = self._current_source_key()
        if key is None:
            return self.state.mask_id
        existing = self.state.mask_ids_by_key.get(key)
        if existing is None:
            existing = self._allocate_mask_id()
            self.state.mask_ids_by_key[key] = existing
        self.state.mask_id = existing
        self.mask_id_edit.setText(str(existing))
        return existing

    def _current_mask_id(self) -> int | None:
        return self.state.mask_id

    def _clear_log(self, message: str | None = None) -> None:
        self.result_panel.clear()
        if message:
            self.result_panel.appendPlainText(message)

    def _append_log(self, message: str) -> None:
        self.result_panel.appendPlainText(message)

    def _reset_progress(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")

    def _update_progress(self, current: int, total: int, message: str) -> None:
        if total > 0:
            value = int(round((current / total) * 100.0))
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, value)))
            self.progress_bar.setFormat(f"{message} ({current}/{total})")
            self.statusBar().showMessage(f"{message} ({current}/{total})")
            self._append_log(f"[{current}/{total}] {message}")
            return
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat(message)
        self.statusBar().showMessage(message)
        self._append_log(message)

    def _browse_model(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select SAM 3 checkpoint", "", "PyTorch (*.pt *.pth)")
        if path:
            if self.model_combo.findText(path) == -1:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def _browse_cache_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Cache Directory",
            self.cache_dir_edit.text() or str(self.cache_store.root),
        )
        if not path:
            return
        if self.current_task is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "Wait for the current task to finish before switching cache directories.")
            return
        self._switch_cache_dir(path)

    def _clear_cache_dir(self) -> None:
        if self.current_task is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "Wait for the current task to finish before clearing the cache.")
            return
        self.cache_store.clear()
        self.backend = None
        self._backend_signature = None
        self._clear_all()
        self.state.cache_dir = str(self.cache_store.root)
        self.cache_dir_edit.setText(self.state.cache_dir)
        self.statusBar().showMessage(f"Cleared cache: {self.cache_store.root}")
        self._append_log(f"Cleared cache directory: {self.cache_store.root}")

    def _set_compact_cache_mode(self, checked: bool) -> None:
        if self.current_task is not None:
            self.compact_cache_checkbox.blockSignals(True)
            self.compact_cache_checkbox.setChecked(self.cache_store.compact_archives)
            self.compact_cache_checkbox.blockSignals(False)
            QtWidgets.QMessageBox.warning(self, "Busy", "Wait for the current task to finish before changing cache layout.")
            return
        self.cache_store.compact_archives = bool(checked)
        self.state.compact_cache_enabled = bool(checked)
        mode_text = "compact cache archives enabled" if checked else "legacy cache layout enabled"
        self.statusBar().showMessage(mode_text)
        self._append_log(mode_text)

    def _toggle_inference_scale(self, checked: bool) -> None:
        self.state.inference_scale_enabled = bool(checked)
        self.inference_scale_spin.setEnabled(bool(checked))
        if not checked:
            self.state.inference_scale = 1.0

    def _current_inference_scale(self) -> float:
        if not self.downscale_inference_checkbox.isChecked():
            self.state.inference_scale_enabled = False
            self.state.inference_scale = 1.0
            return 1.0
        self.state.inference_scale_enabled = True
        self.state.inference_scale = float(self.inference_scale_spin.value())
        return self.state.inference_scale

    def _switch_cache_dir(self, path: str) -> None:
        self.cache_store.set_root(path)
        self.backend = None
        self._backend_signature = None
        self._clear_all()
        self.state.cache_dir = str(self.cache_store.root)
        self.state.compact_cache_enabled = self.cache_store.compact_archives
        self.cache_dir_edit.setText(self.state.cache_dir)
        self.statusBar().showMessage(f"Using cache: {self.cache_store.root}")
        self._append_log(f"Using cache directory: {self.cache_store.root}")

    def _current_source_key(self, index: int | None = None) -> str | None:
        if self.state.source_kind == "directory":
            if not self.state.source_items:
                return None
            current_index = self._current_source_index() if index is None else index
            current_index = min(max(current_index, 0), len(self.state.source_items) - 1)
            return self.state.source_items[current_index]
        if self.state.source_kind == "video":
            if not self.state.source_path:
                return None
            frame_index = self._current_source_index() if index is None else index
            return f"{self.state.source_path}::frame:{frame_index}"
        return self.state.source_path

    def _first_sequence_mask(self):
        if self.state.source_kind == "directory":
            key = self._current_source_key(0)
        elif self.state.source_kind == "video":
            key = self._current_source_key(0)
        else:
            key = self._current_source_key()
        if key is None:
            return None
        mask = self.state.mask_inputs_by_key.get(key)
        return mask

    @staticmethod
    def _cache_token(value: object) -> str:
        text = "" if value is None else str(value)
        return "".join(char if char.isalnum() else "_" for char in text)[:80] or "item"

    def _load_cached_mask(self, mask_ref: object) -> np.ndarray | None:
        mask = load_cached_mask(mask_ref)
        if mask is None:
            return None
        return np.asarray(mask, dtype=np.float32)

    def _cache_prompt_mask(self, key: str, mask: object) -> tuple[str, np.ndarray]:
        array = self._load_cached_mask(mask)
        if array is None:
            raise ValueError("Mask cache write requires a valid mask.")
        path = self.cache_store.write_mask("prompt", key, array)
        return path, array

    def _cache_manual_mask(self, key: str, mask: object) -> tuple[str, np.ndarray]:
        array = self._load_cached_mask(mask)
        if array is None:
            raise ValueError("Manual mask cache write requires a valid mask.")
        path = self.cache_store.write_mask("manual", key, array)
        return path, array

    def _cache_result(self, result: PredictionResult, key: str) -> PredictionResult:
        return self.cache_store.write_result("inference", key, result)

    def _cache_results(self, results):
        if results is None:
            return None
        if isinstance(results, list):
            cached_results: list[PredictionResult | None] = []
            for index, result in enumerate(results):
                if result is None:
                    cached_results.append(None)
                    continue
                key = f"{self._cache_token(result.source)}_{result.frame_index if result.frame_index is not None else index}"
                cached_results.append(self._cache_result(result, key))
            return cached_results
        key = f"{self._cache_token(results.source)}_{results.frame_index if results.frame_index is not None else 'single'}"
        return self._cache_result(results, key)

    def _store_current_prompt_target(self) -> bool:
        key = self._current_source_key()
        class_value = self.mask_class_edit.text().strip() or self.text_prompt_edit.text().strip() or None
        id_text = self.mask_id_edit.text().strip()
        if id_text:
            try:
                id_value = int(id_text)
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Invalid Prompt ID", "Prompt ID must be an integer.")
                self.mask_id_edit.setFocus()
                return False
        else:
            id_value = self.state.mask_id
            if id_value is None and (self.state.mask_input is not None or self.state.points or self.state.boxes):
                id_value = self._allocate_mask_id()
        self.state.mask_id = id_value
        self.state.mask_class = class_value
        self.mask_class_edit.blockSignals(True)
        self.mask_class_edit.setText(class_value or "")
        self.mask_class_edit.blockSignals(False)
        self.mask_id_edit.blockSignals(True)
        self.mask_id_edit.setText("" if id_value is None else str(id_value))
        self.mask_id_edit.blockSignals(False)
        if key is not None:
            if class_value:
                self.state.mask_classes_by_key[key] = class_value
            else:
                self.state.mask_classes_by_key.pop(key, None)
            if id_value is not None:
                self.state.mask_ids_by_key[key] = int(id_value)
            else:
                self.state.mask_ids_by_key.pop(key, None)
        if id_value is not None:
            self.state.next_mask_id = max(self.state.next_mask_id, int(id_value) + 1)
        self._refresh_view_filters()
        return True

    def _sync_current_mask_state(self) -> None:
        key = self._current_source_key()
        self.state.mask_path = None
        self.state.mask_input = None
        self.state.mask_source = None
        self.state.mask_class = None
        self.state.mask_id = None
        self.state.manual_mask_input = None
        self.state.points = []
        self.state.boxes = []
        if key is not None:
            mask_ref = self.state.mask_inputs_by_key.get(key)
            if mask_ref is not None:
                self.state.mask_input = self._load_cached_mask(mask_ref)
                self.state.mask_path = str(mask_ref)
            self.state.mask_source = self.state.mask_sources_by_key.get(key)
            self.state.mask_class = self.state.mask_classes_by_key.get(key)
            self.state.mask_id = self.state.mask_ids_by_key.get(key)
            manual_mask_ref = self.state.manual_masks_by_key.get(key)
            if manual_mask_ref is not None:
                self.state.manual_mask_input = self._load_cached_mask(manual_mask_ref)
            self.state.points = list(self.state.points_by_key.get(key, []))
            self.state.boxes = list(self.state.boxes_by_key.get(key, []))
        self.mask_class_edit.blockSignals(True)
        self.mask_class_edit.setText(self.state.mask_class or "")
        self.mask_class_edit.blockSignals(False)
        self.mask_id_edit.blockSignals(True)
        self.mask_id_edit.setText("" if self.state.mask_id is None else str(self.state.mask_id))
        self.mask_id_edit.blockSignals(False)
        if self.state.mask_input is None:
            self.mask_label.setText("No mask selected")
        elif self.state.mask_source:
            self.mask_label.setText(self.state.mask_source)
        else:
            non_zero = int((self.state.mask_input > 0).sum())
            source = self.state.mask_path or "mask"
            self.mask_label.setText(f"{source.title()} mask ({non_zero} px)")
        if self.state.manual_mask_input is None:
            self.manual_mask_label.setText("No manual mask")
        else:
            non_zero = int((self.state.manual_mask_input > 0).sum())
            self.manual_mask_label.setText(f"manualMask id=0 ({non_zero} px)")
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._refresh_view_filters()

    def _clear_all_masks(self) -> None:
        self.state.mask_inputs_by_key.clear()
        self.state.mask_paths_by_key.clear()
        self.state.mask_sources_by_key.clear()
        self.state.mask_ids_by_key.clear()
        self.state.mask_classes_by_key.clear()
        self.state.manual_masks_by_key.clear()
        self.state.mask_path = None
        self.state.mask_input = None
        self.state.mask_source = None
        self.state.mask_class = None
        self.state.mask_id = None
        self.state.manual_mask_input = None
        self.state.next_mask_id = 1
        self.mask_class_edit.blockSignals(True)
        self.mask_class_edit.clear()
        self.mask_class_edit.blockSignals(False)
        self.mask_id_edit.clear()
        self.mask_label.setText("No mask selected")
        self.manual_mask_label.setText("No manual mask")
        self.preview_canvas.set_prompt_mask_preview(None)
        self.preview_canvas.set_manual_mask_preview(None)
        self._refresh_view_filters()
        self._update_prompt_summary()

    def _clear_source_prompts(self) -> None:
        self.state.points.clear()
        self.state.boxes.clear()
        self.state.points_by_key.clear()
        self.state.boxes_by_key.clear()
        self._clear_all_masks()
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._update_prompt_summary()

    def _reset_result_view(self) -> None:
        self.state.results = None
        self.state.current_frame_index = 0
        self._refresh_view_filters()
        self._reset_playback()

    def _load_image_path(self, path: str) -> None:
        self._clear_source_prompts()
        self.state.source_path = path
        self.state.source_kind = "image"
        self.state.source_items = [path]
        self.state.source_frame_count = None
        self._reset_result_view()
        self.preview_canvas.set_image(to_bgr_image(path))
        self.preview_canvas.set_prompt_mask_preview(None)
        self.preview_canvas.set_manual_mask_preview(None)
        self._set_result_summary(f"Loaded image: {Path(path).name}")
        self._clear_log(f"Loaded image: {path}")
        self._configure_playback()
        self._sync_current_mask_state()

    def _load_directory_path(self, path: str) -> None:
        files = [str(item) for item in list_image_directory(path)]
        self._clear_source_prompts()
        self.state.source_path = path
        self.state.source_kind = "directory"
        self.state.source_items = files
        self.state.source_frame_count = None
        self._reset_result_view()
        self.preview_canvas.set_image(to_bgr_image(files[0]))
        self.preview_canvas.set_prompt_mask_preview(None)
        self.preview_canvas.set_manual_mask_preview(None)
        self._set_result_summary(f"Loaded folder with {len(files)} images")
        self._clear_log(f"Loaded folder: {path}\nImages: {len(files)}")
        self._configure_playback()
        self._sync_current_mask_state()

    def _load_video_path(self, path: str) -> None:
        self._clear_source_prompts()
        self.state.source_path = path
        self.state.source_kind = "video"
        self.state.source_items = [path]
        self.state.source_frame_count = video_frame_count(path)
        self._reset_result_view()
        self.preview_canvas.set_image(read_video_frame(path, 0))
        self.preview_canvas.set_prompt_mask_preview(None)
        self.preview_canvas.set_manual_mask_preview(None)
        self._set_result_summary(f"Loaded video with {self.state.source_frame_count or 0} frames")
        self._clear_log(f"Loaded video: {path}")
        self._configure_playback()
        self._sync_current_mask_state()

    def _open_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            self._load_image_path(path)

    def _open_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Image Directory", self.state.source_path or str(Path.cwd()))
        if path:
            try:
                self._load_directory_path(path)
            except Exception:
                QtWidgets.QMessageBox.critical(self, "Directory Error", traceback.format_exc())

    def _open_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
        if path:
            self._load_video_path(path)

    def _load_mask_from_path(self, path: str) -> None:
        mask_input, _metadata = normalize_mask_input(path)
        key = self._current_source_key()
        cache_key = key or path
        cached_path, cached_mask = self._cache_prompt_mask(cache_key, mask_input)
        self.state.mask_path = cached_path
        self.state.mask_input = cached_mask
        self.state.mask_source = path
        self._ensure_prompt_target_assignment()
        mask_id = self._current_mask_id()
        if key is not None:
            self.state.mask_inputs_by_key[key] = cached_path
            self.state.mask_paths_by_key[key] = path
            self.state.mask_sources_by_key[key] = path
        self.mask_label.setText(path)
        self._refresh_view_filters()
        self._append_log(f"Loaded mask: {path} (id={mask_id})")
        self._refresh_preview()

    def _open_mask(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Mask", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if path:
            try:
                self._load_mask_from_path(path)
            except Exception:
                QtWidgets.QMessageBox.critical(self, "Mask Error", traceback.format_exc())

    def _preferred_result_object(self):
        result = self._current_result()
        if result is None or not result.objects:
            return None
        if self.state.mask_id is not None:
            for obj in result.objects:
                if obj.track_id == self.state.mask_id:
                    return obj
        tracked_ids = result.tracking_metadata.get("active_track_ids") or []
        if tracked_ids:
            for obj in result.objects:
                if obj.track_id == tracked_ids[0]:
                    return obj
        return result.objects[0]

    def _ensure_prompt_target_assignment(self) -> bool:
        if not self._store_current_prompt_target():
            return False
        if self.state.mask_id is None:
            self.state.mask_id = self._allocate_mask_id()
            self.mask_id_edit.blockSignals(True)
            self.mask_id_edit.setText(str(self.state.mask_id))
            self.mask_id_edit.blockSignals(False)
            key = self._current_source_key()
            if key is not None:
                self.state.mask_ids_by_key[key] = self.state.mask_id
        return True

    def _set_manual_mask(self, mask: object) -> None:
        key = self._current_source_key()
        cached_key = key or "current"
        cached_path, cached_mask = self._cache_manual_mask(cached_key, mask)
        self.state.manual_mask_input = cached_mask
        if key is not None:
            self.state.manual_masks_by_key[key] = cached_path
        non_zero = int((self.state.manual_mask_input > 0).sum())
        self.manual_mask_label.setText(f"manualMask id=0 ({non_zero} px)")
        self._refresh_view_filters()
        self.preview_canvas.set_manual_mask_preview(self._current_manual_mask_preview())
        self._append_log(f"Updated manualMask id 0 for current view ({non_zero} px)")

    def _clear_current_manual_mask(self) -> None:
        key = self._current_source_key()
        if key is not None:
            self.state.manual_masks_by_key.pop(key, None)
        self.state.manual_mask_input = None
        self.manual_mask_label.setText("No manual mask")
        self.preview_canvas.set_manual_mask_preview(None)
        self._refresh_view_filters()
        self._refresh_preview()
        self._append_log("Cleared manualMask id 0 for current view")

    def _copy_manual_mask_to_clipboard(self) -> None:
        if self.state.manual_mask_input is None:
            QtWidgets.QMessageBox.warning(self, "No Manual Mask", "Create a manual mask first.")
            return
        mask = np.asarray(self.state.manual_mask_input, dtype=np.float32).copy()
        self._manual_mask_clipboard = mask
        uint8_mask = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        height, width = uint8_mask.shape
        qimage = QtGui.QImage(uint8_mask.data, width, height, width, QtGui.QImage.Format.Format_Grayscale8).copy()
        QtWidgets.QApplication.clipboard().setImage(qimage)
        self._append_log("Copied manualMask id 0 to clipboard")
        self.statusBar().showMessage("Copied manualMask id 0 to clipboard")

    def _paste_manual_mask_to_current_frame(self) -> None:
        mask = None
        if self._manual_mask_clipboard is not None:
            mask = np.asarray(self._manual_mask_clipboard, dtype=np.float32).copy()
        else:
            qimage = QtWidgets.QApplication.clipboard().image()
            if not qimage.isNull():
                gray = qimage.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
                width = gray.width()
                height = gray.height()
                bits = gray.bits()
                array = np.frombuffer(bits, dtype=np.uint8, count=gray.bytesPerLine() * height).reshape((height, gray.bytesPerLine()))
                mask = (array[:, :width] > 0).astype(np.float32)
        if mask is None:
            QtWidgets.QMessageBox.warning(self, "Clipboard Empty", "No manual mask is available in the clipboard.")
            return
        self._set_manual_mask(mask)
        self._refresh_preview()
        self._append_log("Pasted manualMask id 0 into current view")

    def _copy_manual_mask_to_all_frames(self) -> None:
        if self.state.manual_mask_input is None:
            QtWidgets.QMessageBox.warning(self, "No Manual Mask", "Create a manual mask first.")
            return
        targets: list[str] = []
        if self.state.source_kind == "directory":
            targets = list(self.state.source_items)
        elif self.state.source_kind == "video" and self.state.source_frame_count:
            targets = [self._current_source_key(index) for index in range(int(self.state.source_frame_count)) if self._current_source_key(index)]
        elif self._current_source_key() is not None:
            targets = [self._current_source_key()]
        mask = np.asarray(self.state.manual_mask_input, dtype=np.float32).copy()
        total = len(targets)
        for index, key in enumerate(targets, start=1):
            if key is not None:
                cached_path, _cached_mask = self._cache_manual_mask(key, mask)
                self.state.manual_masks_by_key[key] = cached_path
                self._append_log(f"[{index}/{total}] Copied manualMask id 0 to {self._format_source_key_label(key)}")
        self._refresh_view_filters()
        self._refresh_preview()
        self._append_log(f"Copied manualMask id 0 to {len(targets)} frame(s)")

    def _clear_mask(self, reset_label: bool = True, *, update_preview: bool = True) -> None:
        key = self._current_source_key()
        if key is not None:
            self.state.mask_inputs_by_key.pop(key, None)
            self.state.mask_paths_by_key.pop(key, None)
            self.state.mask_sources_by_key.pop(key, None)
            self.state.mask_classes_by_key.pop(key, None)
            self.state.mask_ids_by_key.pop(key, None)
        self.state.mask_path = None
        self.state.mask_input = None
        self.state.mask_source = None
        self.state.mask_class = None
        self.state.mask_id = None
        if reset_label:
            self.mask_label.setText("No mask selected")
        self.mask_class_edit.blockSignals(True)
        self.mask_class_edit.clear()
        self.mask_class_edit.blockSignals(False)
        self.mask_id_edit.clear()
        self.preview_canvas.set_prompt_mask_preview(None)
        self._refresh_view_filters()
        self._update_prompt_summary()
        if update_preview:
            self._refresh_preview()

    def _clear_masks_only(self) -> None:
        self._clear_all_masks()
        self.state.results = None
        self.state.current_frame_index = 0
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._set_result_summary("Masks cleared")
        self._refresh_preview()
        self._configure_playback()
        self.statusBar().showMessage("Cleared all masks and mask results")
        self._append_log("Cleared all masks and mask results")

    def _browse_export_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Directory", self.export_dir_edit.text() or str(Path.cwd()))
        if path:
            self.state.export_dir = path
            self.export_dir_edit.setText(path)

    def _set_tool(self, tool: str) -> None:
        mapping = {
            "point": self.point_tool_button,
            "box": self.box_tool_button,
            "manual_mask": self.manual_mask_tool_button,
        }
        if tool in mapping and mapping[tool].isChecked():
            for name, widget in mapping.items():
                if name != tool:
                    widget.setChecked(False)
        self.preview_canvas.set_tool(tool)

    def _set_brush_size(self, value: int) -> None:
        self.preview_canvas.set_brush_radius(value)
        self.brush_label.setText(f"{value} px")

    def _clear_prompts(self) -> None:
        key = self._current_source_key()
        if key is not None:
            self.state.points_by_key.pop(key, None)
            self.state.boxes_by_key.pop(key, None)
        self.state.points.clear()
        self.state.boxes.clear()
        self._clear_mask(reset_label=True, update_preview=False)
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._update_prompt_summary()
        self._refresh_preview()

    def _clear_all(self) -> None:
        self.text_prompt_edit.clear()
        self.state.points.clear()
        self.state.boxes.clear()
        self.state.points_by_key.clear()
        self.state.boxes_by_key.clear()
        self._clear_all_masks()
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._reset_result_view()
        self._set_result_summary("No inference results yet.")
        self._clear_log(self.state.source_path or "Cleared prompts and results")
        for widget in [self.point_tool_button, self.box_tool_button, self.manual_mask_tool_button]:
            widget.setChecked(False)
        self.preview_canvas.set_tool("none")
        self.statusBar().showMessage("Cleared prompts and results")
        if self.state.source_path and Path(self.state.source_path).exists():
            self._refresh_preview()
        elif self.state.source_kind == "directory" and self.state.source_items:
            self._refresh_preview()
        else:
            self.preview_canvas.clear()
        self._configure_playback()

    def _add_point(self, x: float, y: float, label: int) -> None:
        if not self._ensure_prompt_target_assignment():
            return
        self.state.points.append((x, y, label))
        key = self._current_source_key()
        if key is not None:
            self.state.points_by_key[key] = list(self.state.points)
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._update_prompt_summary()
        self._schedule_interaction_preview("point prompt")

    def _add_box(self, x1: float, y1: float, x2: float, y2: float, label: int) -> None:
        if not self._ensure_prompt_target_assignment():
            return
        self.state.boxes.append((x1, y1, x2, y2, label))
        key = self._current_source_key()
        if key is not None:
            self.state.boxes_by_key[key] = list(self.state.boxes)
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self._update_prompt_summary()
        self._schedule_interaction_preview("box prompt")

    def _update_prompt_summary(self) -> None:
        suffixes = []
        if self.state.mask_input is not None:
            suffixes.append("mask")
        if self.state.mask_id is not None:
            suffixes.append(f"id={self.state.mask_id}")
        suffix = ""
        if suffixes:
            suffix = ", " + ", ".join(suffixes)
        self.prompt_summary_label.setText(f"{len(self.state.points)} points, {len(self.state.boxes)} boxes{suffix}")

    def _backend_settings_signature(self) -> tuple:
        return (
            str(Path(self.model_combo.currentText().strip())),
            self.device_combo.currentText(),
            float(self.confidence_spin.value()),
            str(self.cache_store.yolo_dir),
        )

    def _create_backend(self) -> SAM3Ultralytics:
        signature = self._backend_settings_signature()
        # Keep a hot backend around for iterative GUI use. Reload only when a
        # runtime-affecting setting changes so repeated runs avoid model startup cost.
        if self.backend is not None and self._backend_signature == signature:
            return self.backend
        backend = SAM3Ultralytics(
            signature[0],
            device=signature[1],
            conf=signature[2],
            yolo_config_dir=signature[3],
        )
        backend.load()
        self.backend = backend
        self._backend_signature = signature
        return backend

    def _current_mask_argument(self):
        return self.state.mask_path or self.state.mask_input

    def _first_sequence_key(self) -> str | None:
        if self.state.source_kind == "directory" and self.state.source_items:
            return self.state.source_items[0]
        if self.state.source_kind == "video" and self.state.source_path:
            return f"{self.state.source_path}::frame:0"
        return self._current_source_key()

    def _current_mask_text(self) -> str | None:
        text = self.text_prompt_edit.text().strip()
        if text:
            return text
        mask_class = (self.state.mask_class or "").strip()
        return mask_class or None

    def _first_sequence_mask_text(self) -> str | None:
        text = self.text_prompt_edit.text().strip()
        if text:
            return text
        key = self._first_sequence_key()
        if key is None:
            return None
        value = self.state.mask_classes_by_key.get(key, "").strip()
        return value or None

    def _first_sequence_mask_id(self) -> int | None:
        key = self._first_sequence_key()
        if key is None:
            return self._current_mask_id()
        return self.state.mask_ids_by_key.get(key)

    def _first_sequence_points(self):
        key = self._first_sequence_key()
        if key is None:
            return None
        points = self.state.points_by_key.get(key)
        return list(points) if points else None

    def _first_sequence_boxes(self):
        key = self._first_sequence_key()
        if key is None:
            return None
        boxes = self.state.boxes_by_key.get(key)
        return list(boxes) if boxes else None

    def _directory_mask_map(self) -> dict[str, object]:
        return {
            key: value
            for key, value in self.state.mask_inputs_by_key.items()
            if key in self.state.source_items
        }

    def _directory_points_map(self) -> dict[str, list[tuple[float, float, int]]]:
        return {key: list(value) for key, value in self.state.points_by_key.items() if key in self.state.source_items and value}

    def _directory_boxes_map(self) -> dict[str, list[tuple[float, float, float, float, int]]]:
        return {key: list(value) for key, value in self.state.boxes_by_key.items() if key in self.state.source_items and value}

    def _directory_text_map(self) -> dict[str, str]:
        if self.text_prompt_edit.text().strip():
            return {}
        return {
            key: value
            for key, value in self.state.mask_classes_by_key.items()
            if key in self.state.source_items and value
        }

    def _video_mask_map(self) -> dict[int, object]:
        prefix = f"{self.state.source_path}::frame:"
        items: dict[int, object] = {}
        for key, value in self.state.mask_inputs_by_key.items():
            if not key.startswith(prefix):
                continue
            try:
                frame_index = int(key.split(":")[-1])
            except ValueError:
                continue
            items[frame_index] = value
        return items

    def _video_points_map(self) -> dict[int, list[tuple[float, float, int]]]:
        prefix = f"{self.state.source_path}::frame:"
        items: dict[int, list[tuple[float, float, int]]] = {}
        for key, value in self.state.points_by_key.items():
            if not key.startswith(prefix) or not value:
                continue
            try:
                frame_index = int(key.split(":")[-1])
            except ValueError:
                continue
            items[frame_index] = list(value)
        return items

    def _video_boxes_map(self) -> dict[int, list[tuple[float, float, float, float, int]]]:
        prefix = f"{self.state.source_path}::frame:"
        items: dict[int, list[tuple[float, float, float, float, int]]] = {}
        for key, value in self.state.boxes_by_key.items():
            if not key.startswith(prefix) or not value:
                continue
            try:
                frame_index = int(key.split(":")[-1])
            except ValueError:
                continue
            items[frame_index] = list(value)
        return items

    def _video_text_map(self) -> dict[int, str]:
        if self.text_prompt_edit.text().strip():
            return {}
        prefix = f"{self.state.source_path}::frame:"
        items: dict[int, str] = {}
        for key, value in self.state.mask_classes_by_key.items():
            if not key.startswith(prefix) or not value:
                continue
            try:
                frame_index = int(key.split(":")[-1])
            except ValueError:
                continue
            items[frame_index] = value
        return items

    def _prompt_kwargs(self) -> dict[str, object]:
        return {
            "text_prompt": self._current_mask_text(),
            "points": self.state.points or None,
            "boxes": self.state.boxes or None,
            "mask_input": self._current_mask_argument(),
            "mask_id": self._current_mask_id(),
            "mask_label": self.state.mask_class,
        }

    def _has_active_prompt(self) -> bool:
        prompt_kwargs = self._prompt_kwargs()
        return any(
            [
                prompt_kwargs["text_prompt"],
                prompt_kwargs["points"],
                prompt_kwargs["boxes"],
                prompt_kwargs["mask_input"],
            ]
        )

    def _schedule_interaction_preview(self, reason: str) -> None:
        self.preview_timer.stop()
        if not self.state.source_path or not self._has_active_prompt():
            return
        self.statusBar().showMessage(f"Updated {reason}. Press Run to infer.")

    def _run_interaction_preview(self) -> None:
        if self.current_task is not None or not self.state.source_path or not self._has_active_prompt():
            return
        prompt_kwargs = self._prompt_kwargs()
        current_index = self.seek_slider.value()
        inference_scale = self._current_inference_scale()

        def job(progress_callback=None, cancel_callback=None, item_start_callback=None, item_result_callback=None):
            backend = self._create_backend()
            self.backend = backend
            if self.state.source_kind == "image":
                return backend.predict_image(self.state.source_path, inference_scale=inference_scale, **prompt_kwargs)
            if self.state.source_kind == "directory":
                source = self.state.source_items[min(current_index, len(self.state.source_items) - 1)]
                return backend.predict_image(source, inference_scale=inference_scale, **prompt_kwargs)
            results = backend.predict_video_frames(
                self.state.source_path,
                frame_indices=[current_index],
                inference_scale=inference_scale,
                **prompt_kwargs,
            )
            return results[0] if results else None

        self.current_task = BackendTask(job)
        self.current_task_mode = "preview"
        self.current_task.signals.result.connect(self._handle_preview_result)
        self.current_task.signals.error.connect(self._handle_preview_error)
        self.current_task.signals.finished.connect(self._task_finished)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Previewing current view")
        self.statusBar().showMessage("Previewing current view")
        self.thread_pool.start(self.current_task)

    def _run_inference(self) -> None:
        if not self.state.source_path:
            QtWidgets.QMessageBox.warning(self, "Missing Source", "Open an image, folder, or video first.")
            return
        if not self.model_combo.currentText().strip():
            QtWidgets.QMessageBox.warning(self, "Missing Model", "Select a SAM 3 checkpoint.")
            return

        self.preview_timer.stop()
        export_dir = self.export_dir_edit.text().strip() or None
        auto_export_dir = export_dir if self.auto_export_masks_checkbox.isChecked() else None
        merged_mask_only = self.merge_masks_only_checkbox.isChecked()
        invert_mask = self.invert_mask_export_checkbox.isChecked()
        run_scope = self.run_scope_combo.currentData() or "current"
        current_index = self.seek_slider.value()
        prompt_kwargs = self._prompt_kwargs()
        base_text_prompt = self.text_prompt_edit.text().strip() or None
        first_sequence_text = self._first_sequence_mask_text()
        first_sequence_mask = self._first_sequence_mask()
        first_sequence_mask_id = self._first_sequence_mask_id()
        directory_mask_map = self._directory_mask_map()
        directory_points_map = self._directory_points_map()
        directory_boxes_map = self._directory_boxes_map()
        directory_text_map = self._directory_text_map()
        video_mask_map = self._video_mask_map()
        video_points_map = self._video_points_map()
        video_boxes_map = self._video_boxes_map()
        video_text_map = self._video_text_map()
        inference_scale = self._current_inference_scale()

        effective_run_scope = run_scope

        if self.state.source_kind == "directory" and effective_run_scope == "all":
            total_hint = len(self.state.source_items)
            run_label = f"Running inference across {total_hint} images"
        elif self.state.source_kind == "video" and effective_run_scope == "all":
            total_hint = int(self.state.source_frame_count or 0)
            run_label = f"Running inference across {total_hint} frames"
        else:
            total_hint = 1
            run_label = "Running inference on current view"

        streaming_batch_run = self.state.source_kind in {"directory", "video"} and effective_run_scope == "all"

        if streaming_batch_run:
            self._start_incremental_sequence_run(
                run_label=run_label,
                total_hint=total_hint,
                export_dir=export_dir,
                auto_export_dir=auto_export_dir,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                base_text_prompt=base_text_prompt,
                first_sequence_text=first_sequence_text,
                first_sequence_mask=first_sequence_mask,
                first_sequence_mask_id=first_sequence_mask_id,
                directory_mask_map=directory_mask_map,
                directory_points_map=directory_points_map,
                directory_boxes_map=directory_boxes_map,
                directory_text_map=directory_text_map,
                video_mask_map=video_mask_map,
                video_points_map=video_points_map,
                video_boxes_map=video_boxes_map,
                video_text_map=video_text_map,
                inference_scale=inference_scale,
            )
            return

        def job(progress_callback=None, cancel_callback=None, item_start_callback=None, item_result_callback=None):
            backend = self._create_backend()
            self.backend = backend
            if self.state.source_kind == "image":
                return backend.predict_image(
                    self.state.source_path,
                    export_mask_dir=auto_export_dir,
                    output_dir=None,
                    merged_mask_only=merged_mask_only,
                    invert_mask=invert_mask,
                    inference_scale=inference_scale,
                    **prompt_kwargs,
                )
            if self.state.source_kind == "directory":
                if effective_run_scope == "current":
                    source = self.state.source_items[min(current_index, len(self.state.source_items) - 1)]
                    return backend.predict_image(
                        source,
                        export_mask_dir=auto_export_dir,
                        output_dir=None,
                        merged_mask_only=merged_mask_only,
                        invert_mask=invert_mask,
                        inference_scale=inference_scale,
                        **prompt_kwargs,
                    )
                if first_sequence_mask is not None and progress_callback is not None:
                    progress_callback(0, len(self.state.source_items), "Using first-frame mask to constrain folder sequence")
                return backend.predict_image_sequence(
                    self.state.source_items,
                    text_prompt=base_text_prompt,
                    points=self._first_sequence_points(),
                    boxes=self._first_sequence_boxes(),
                    mask_input=first_sequence_mask,
                    mask_inputs=directory_mask_map or None,
                    points_by_source=directory_points_map or None,
                    boxes_by_source=directory_boxes_map or None,
                    text_prompts_by_source=directory_text_map or None,
                    reuse_first_mask=False,
                    first_mask_initializer_only=first_sequence_mask is not None,
                    mask_id=first_sequence_mask_id,
                    mask_label=first_sequence_text,
                    export_mask_dir=auto_export_dir,
                    output_dir=None,
                    merged_mask_only=merged_mask_only,
                    invert_mask=invert_mask,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                    item_start_callback=item_start_callback,
                    item_result_callback=item_result_callback,
                    inference_scale=inference_scale,
                )
            if effective_run_scope == "current":
                results = backend.predict_video_frames(
                    self.state.source_path,
                    frame_indices=[current_index],
                    export_mask_dir=auto_export_dir,
                    output_dir=export_dir if auto_export_dir else None,
                    merged_mask_only=merged_mask_only,
                    invert_mask=invert_mask,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                    item_start_callback=item_start_callback,
                    item_result_callback=item_result_callback,
                    inference_scale=inference_scale,
                    **prompt_kwargs,
                )
                return results[0] if results else None
            if first_sequence_mask is not None or video_mask_map or video_points_map or video_boxes_map:
                return backend.track_video(
                    self.state.source_path,
                    text_prompt=base_text_prompt,
                    points=self._first_sequence_points(),
                    boxes=self._first_sequence_boxes(),
                    mask_input=first_sequence_mask,
                    mask_inputs_by_frame=video_mask_map or None,
                    points_by_frame=video_points_map or None,
                    boxes_by_frame=video_boxes_map or None,
                    text_prompts_by_frame=video_text_map or None,
                    mask_id=first_sequence_mask_id,
                    mask_label=first_sequence_text,
                    export_mask_dir=auto_export_dir,
                    output_dir=export_dir if auto_export_dir else None,
                    merged_mask_only=merged_mask_only,
                    invert_mask=invert_mask,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_callback,
                    item_start_callback=item_start_callback,
                    item_result_callback=item_result_callback,
                    inference_scale=inference_scale,
                )
            return backend.track_video(
                self.state.source_path,
                text_prompt=base_text_prompt,
                points=self._first_sequence_points(),
                boxes=self._first_sequence_boxes(),
                mask_input=self._current_mask_argument(),
                mask_id=self._current_mask_id(),
                mask_label=self.state.mask_class,
                export_mask_dir=auto_export_dir,
                output_dir=export_dir if auto_export_dir else None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                item_start_callback=item_start_callback,
                item_result_callback=item_result_callback,
                inference_scale=inference_scale,
            )

        self.current_task = BackendTask(job)
        self.current_task_mode = "run"
        self._streaming_batch_mode = streaming_batch_run
        self._streaming_batch_total = total_hint if streaming_batch_run else 0
        self.current_task.signals.result.connect(self._handle_result)
        self.current_task.signals.error.connect(self._handle_error)
        self.current_task.signals.finished.connect(self._task_finished)
        self.current_task.signals.progress.connect(self._handle_progress)
        if streaming_batch_run:
            self.current_task.signals.item_started.connect(self._handle_batch_item_started)
            self.current_task.signals.item_result.connect(self._handle_batch_item_result)
            if not self.append_inference_checkbox.isChecked() or not isinstance(self.state.results, list) or len(self.state.results) != total_hint:
                self.state.results = [None] * total_hint
            if total_hint > 0:
                self.seek_slider.setValue(0)
            self._configure_playback()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(run_label)
        self.statusBar().showMessage(run_label)
        self._append_log(run_label)
        if total_hint <= 1:
            self.progress_bar.setRange(0, 0)
        self.thread_pool.start(self.current_task)

    def _start_incremental_sequence_run(
        self,
        *,
        run_label: str,
        total_hint: int,
        export_dir,
        auto_export_dir,
        merged_mask_only: bool,
        invert_mask: bool,
        base_text_prompt,
        first_sequence_text,
        first_sequence_mask,
        first_sequence_mask_id,
        directory_mask_map,
        directory_points_map,
        directory_boxes_map,
        directory_text_map,
        video_mask_map,
        video_points_map,
        video_boxes_map,
        video_text_map,
        inference_scale: float,
    ) -> None:
        items = list(self.state.source_items) if self.state.source_kind == "directory" else list(range(int(self.state.source_frame_count or 0)))
        normalized_directory_mask_map = {
            key: loaded
            for key, value in (directory_mask_map or {}).items()
            if (loaded := self._load_cached_mask(value)) is not None
        }
        normalized_video_mask_map = {
            key: loaded
            for key, value in (video_mask_map or {}).items()
            if (loaded := self._load_cached_mask(value)) is not None
        }
        self._sequence_run_context = {
            "kind": self.state.source_kind,
            "items": items,
            "total": total_hint,
            "results": [None] * total_hint,
            "next_index": 0,
            "sequence_mask": self._load_cached_mask(first_sequence_mask),
            "propagate_sequence_mask": self._load_cached_mask(first_sequence_mask) is not None,
            "mask_id": first_sequence_mask_id,
            "mask_label": first_sequence_text,
            "base_text_prompt": base_text_prompt,
            "first_points": self._first_sequence_points(),
            "first_boxes": self._first_sequence_boxes(),
            "directory_mask_map": normalized_directory_mask_map,
            "directory_points_map": directory_points_map or {},
            "directory_boxes_map": directory_boxes_map or {},
            "directory_text_map": directory_text_map or {},
            "video_mask_map": normalized_video_mask_map,
            "video_points_map": video_points_map or {},
            "video_boxes_map": video_boxes_map or {},
            "video_text_map": video_text_map or {},
            "inference_scale": inference_scale,
            "export_dir": export_dir,
            "auto_export_dir": auto_export_dir,
            "merged_mask_only": merged_mask_only,
            "invert_mask": invert_mask,
            "error": None,
        }
        self.current_task_mode = "run"
        self._streaming_batch_mode = True
        self._streaming_batch_total = total_hint
        if not self.append_inference_checkbox.isChecked() or not isinstance(self.state.results, list) or len(self.state.results) != total_hint:
            self.state.results = [None] * total_hint
        if total_hint > 0:
            self.seek_slider.setValue(0)
        self._configure_playback()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(run_label)
        self.statusBar().showMessage(run_label)
        self._append_log(run_label)
        # Process long folder/video runs one item at a time so the GUI can
        # repaint the current frame, show the loaded result, and then advance.
        self._launch_next_sequence_item()

    def _launch_next_sequence_item(self) -> None:
        context = self._sequence_run_context
        if context is None:
            return
        index = int(context.get("next_index", 0))
        total = int(context.get("total", 0))
        if index >= total:
            self._finalize_incremental_sequence_run()
            return
        kind = context["kind"]
        label = context["items"][index] if kind == "directory" else f"frame:{context['items'][index]}"
        self._handle_batch_item_started(index, total, str(label))

        def job(progress_callback=None, cancel_callback=None):
            backend = self._create_backend()
            self.backend = backend
            if kind == "directory":
                source = context["items"][index]
                # Folder runs are intentionally per-image inference, not hidden
                # tracking, so each image only uses prompts explicitly assigned to it.
                return backend.predict_image(
                    source,
                    text_prompt=context["directory_text_map"].get(str(source), context["base_text_prompt"]),
                    points=context["directory_points_map"].get(str(source), context["first_points"]),
                    boxes=context["directory_boxes_map"].get(str(source), context["first_boxes"]),
                    mask_input=context["directory_mask_map"].get(str(source)),
                    mask_id=context["mask_id"],
                    mask_label=context["mask_label"],
                    inference_scale=context["inference_scale"],
                )

            frame_index = int(context["items"][index])
            if context["propagate_sequence_mask"] or context["video_mask_map"] or context["video_points_map"] or context["video_boxes_map"]:
                results = backend.track_video_frames(
                    self.state.source_path,
                    frame_indices=[frame_index],
                    text_prompt=context["base_text_prompt"],
                    points=context["first_points"],
                    boxes=context["first_boxes"],
                    mask_input=context["sequence_mask"],
                    mask_inputs_by_frame=context["video_mask_map"] or None,
                    points_by_frame=context["video_points_map"] or None,
                    boxes_by_frame=context["video_boxes_map"] or None,
                    text_prompts_by_frame=context["video_text_map"] or None,
                    mask_id=context["mask_id"],
                    mask_label=context["mask_label"],
                    inference_scale=context["inference_scale"],
                    cancel_callback=cancel_callback,
                )
                return results[0] if results else None
            results = backend.predict_video_frames(
                self.state.source_path,
                frame_indices=[frame_index],
                text_prompt=context["video_text_map"].get(frame_index, context["base_text_prompt"]),
                points=context["video_points_map"].get(frame_index, context["first_points"]),
                boxes=context["video_boxes_map"].get(frame_index, context["first_boxes"]),
                mask_input=context["video_mask_map"].get(frame_index),
                mask_id=context["mask_id"],
                mask_label=context["mask_label"],
                inference_scale=context["inference_scale"],
                cancel_callback=cancel_callback,
            )
            return results[0] if results else None

        self.current_task = BackendTask(job)
        self.current_task.signals.result.connect(lambda result, i=index, lab=str(label): self._handle_sequence_item_result(i, total, lab, result))
        self.current_task.signals.error.connect(self._handle_sequence_item_error)
        self.thread_pool.start(self.current_task)

    def _handle_sequence_item_result(self, index: int, total: int, label: str, result) -> None:
        context = self._sequence_run_context
        if context is None:
            return
        self.current_task = None
        self._handle_batch_item_result(index, total, result, label)
        if result is not None and context.get("propagate_sequence_mask"):
            next_mask = result.prompt_mask if result.prompt_mask is not None else self.backend._first_object_mask(result)
            if next_mask is not None:
                context["sequence_mask"] = np.asarray(next_mask, dtype=np.float32).copy()
        if isinstance(self.state.results, list):
            context["results"] = list(self.state.results)
        context["next_index"] = index + 1
        self._update_progress(index + 1, total, f"Processed {self._format_source_key_label(label)}")
        if context["next_index"] >= total:
            self._finalize_incremental_sequence_run()
            return
        # Queue the next item back through the event loop so the frame/result
        # refresh becomes visible before the next inference starts.
        QtCore.QTimer.singleShot(0, self._launch_next_sequence_item)

    def _handle_sequence_item_error(self, message: str) -> None:
        context = self._sequence_run_context
        self.current_task = None
        if context is not None:
            context["error"] = message
        self._handle_error(message)

    def _finalize_incremental_sequence_run(self) -> None:
        context = self._sequence_run_context
        if context is None:
            return
        self.state.results = context["results"]
        self._sequence_run_context = None
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Completed")
        self._append_log(f"Inference completed with {len(self.state.results)} result items")
        self._refresh_view_filters()
        self._configure_playback()
        self._update_result_panel()
        self._refresh_preview()
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.current_task_mode = None
        self._streaming_batch_mode = False
        self._streaming_batch_total = 0

    def _cancel_task(self) -> None:
        if self.current_task is not None:
            self.current_task.cancel()
            self.statusBar().showMessage("Cancellation requested...")
            self._append_log("Cancellation requested")

    @staticmethod
    def _clone_segmentation_object(obj: SegmentationObject, object_index: int) -> SegmentationObject:
        return SegmentationObject(
            mask=np.asarray(obj.mask).copy(),
            box=obj.box,
            score=obj.score,
            label=obj.label,
            track_id=obj.track_id,
            object_index=object_index,
        )

    def _merge_prediction_result(self, existing: PredictionResult | None, incoming: PredictionResult) -> PredictionResult:
        if existing is None:
            return incoming
        objects: list[SegmentationObject] = []
        for obj in existing.objects:
            objects.append(self._clone_segmentation_object(obj, len(objects) + 1))
        for obj in incoming.objects:
            objects.append(self._clone_segmentation_object(obj, len(objects) + 1))
        image = incoming.image if incoming.image is not None else existing.image
        prompt_mask = incoming.prompt_mask if incoming.prompt_mask is not None else existing.prompt_mask
        return PredictionResult(
            source=incoming.source or existing.source,
            frame_index=incoming.frame_index if incoming.frame_index is not None else existing.frame_index,
            mode=incoming.mode or existing.mode,
            image_size=incoming.image_size or existing.image_size,
            inference_image_size=incoming.inference_image_size or existing.inference_image_size,
            objects=objects,
            prompt_metadata=dict(incoming.prompt_metadata or existing.prompt_metadata),
            tracking_metadata=dict(incoming.tracking_metadata or existing.tracking_metadata),
            timings=dict(incoming.timings or existing.timings),
            image=None if image is None else np.asarray(image).copy(),
            prompt_mask=None if prompt_mask is None else np.asarray(prompt_mask).copy(),
        )

    def _apply_inference_result(self, result):
        if not self.append_inference_checkbox.isChecked() or self.state.results is None:
            return result
        if isinstance(result, list):
            if not isinstance(self.state.results, list):
                return result
            merged_results: list[PredictionResult] = []
            existing_results = self.state.results
            for index, incoming in enumerate(result):
                existing = existing_results[index] if index < len(existing_results) else None
                merged_results.append(self._merge_prediction_result(existing, incoming))
            return merged_results
        if isinstance(self.state.results, list):
            results = list(self.state.results)
            index = min(self._current_source_index(), len(results) - 1)
            if 0 <= index < len(results):
                results[index] = self._merge_prediction_result(results[index], result)
                return results
            return results
        return self._merge_prediction_result(self.state.results, result)

    def _handle_result(self, result) -> None:
        if self._streaming_batch_mode and isinstance(result, list):
            if not self.append_inference_checkbox.isChecked():
                self.state.results = self._cache_results(result)
        else:
            self.state.results = self._cache_results(self._apply_inference_result(result))
        self._refresh_view_filters()
        if isinstance(self.state.results, list):
            current_index = min(self._current_source_index(), len(self.state.results) - 1)
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(max(current_index, 0))
            self.seek_slider.blockSignals(False)
            self.state.current_frame_index = max(current_index, 0)
        else:
            self.state.current_frame_index = 0
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Completed")
        if isinstance(self.state.results, list):
            self._append_log(f"Inference completed with {len(self.state.results)} result items")
        elif self.state.results is not None:
            self._append_log(f"Inference completed with {len(self.state.results.objects)} objects")
        self._configure_playback()
        self._update_result_panel()
        self._refresh_preview()

    def _handle_batch_item_started(self, index: int, total: int, label: str) -> None:
        if total <= 0:
            return
        if 0 <= index < total:
            self.seek_slider.setValue(index)
            self.state.current_frame_index = index
            self._display_current_result()
        message = f"Running {self._format_source_key_label(label)}"
        self.statusBar().showMessage(message)
        self._append_log(f"[{index + 1}/{total}] Running {self._format_source_key_label(label)}")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def _handle_batch_item_result(self, index: int, total: int, result, label: str) -> None:
        if result is None or total <= 0:
            return
        if not isinstance(self.state.results, list) or len(self.state.results) != total:
            self.state.results = [None] * total
        existing = self.state.results[index] if 0 <= index < len(self.state.results) else None
        merged_result = self._merge_prediction_result(existing, result) if self.append_inference_checkbox.isChecked() and existing is not None else result
        self.state.results[index] = self._cache_result(merged_result, f"{label}_{index}")
        self.seek_slider.setValue(index)
        self.state.current_frame_index = index
        self._refresh_view_filters()
        self._display_current_result()
        self._append_log(f"[{index + 1}/{total}] Loaded result for {self._format_source_key_label(label)}")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        if index + 1 < total:
            self._advance_batch_preview(index + 1)

    def _handle_preview_result(self, result) -> None:
        if result is None:
            return
        cached_result = self._cache_result(result, f"preview_{self._current_source_key() or 'current'}")
        if isinstance(self.state.results, list) and self.state.source_kind in {"directory", "video"}:
            results = list(self.state.results)
            index = min(self._current_source_index(), len(results) - 1)
            if 0 <= index < len(results):
                results[index] = cached_result
                self.state.results = results
            else:
                self.state.results = cached_result
        else:
            self.state.results = cached_result
        self._refresh_view_filters()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Preview updated")
        self._append_log(f"Interactive preview updated with {len(cached_result.objects)} objects")
        self._configure_playback()
        self._update_result_panel()
        self._refresh_preview()

    def _handle_preview_error(self, message: str) -> None:
        self._append_log("Interactive preview failed")
        self._append_log(message)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Preview failed")
        self.statusBar().showMessage("Interactive preview failed")

    def _handle_error(self, message: str) -> None:
        self._set_result_summary("Inference failed")
        self._append_log("Inference failed")
        self._append_log(message)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Failed")
        self.statusBar().showMessage("Inference failed")
        QtWidgets.QMessageBox.critical(self, "Inference Error", message)

    def _task_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.current_task = None
        self.current_task_mode = None
        self._streaming_batch_mode = False
        self._streaming_batch_total = 0
        if self.progress_bar.maximum() == 0:
            self._reset_progress()

    def _handle_progress(self, current: int, total: int, message: str) -> None:
        self._update_progress(current, total, message)

    def _update_result_panel(self) -> None:
        result = self._current_result()
        if result is None:
            self._set_result_summary("No inference results yet.")
            return
        if not self._result_matches_current_view(result):
            self._set_result_summary("No inference result for the current view.")
            return
        lines = []
        if isinstance(self.state.results, list) and self.state.source_kind == "directory":
            lines.append(f"Batch Images: {len(self.state.results)}")
        lines.extend(
            [
                f"Source: {result.source}",
                f"Mode: {result.mode}",
                f"Image Size: {result.image_size}",
                f"Inference Size: {result.inference_image_size}",
                f"Objects: {len(result.objects)}",
                f"Prompts: {result.prompt_metadata}",
                f"Tracking: {result.tracking_metadata}",
                f"Timings: {result.timings}",
            ]
        )
        self._set_result_summary("\n".join(lines))

    def _current_result(self):
        if self.state.results is None:
            return None
        if isinstance(self.state.results, list):
            index = self.seek_slider.value()
            if 0 <= index < len(self.state.results):
                return self.state.results[index]
            return None
        return self.state.results

    def _current_source_index(self) -> int:
        return max(self.seek_slider.value(), 0)

    def _playback_count(self) -> int:
        if isinstance(self.state.results, list):
            return len(self.state.results)
        if self.state.source_kind == "directory":
            return len(self.state.source_items)
        if self.state.source_kind == "video":
            return int(self.state.source_frame_count or 0)
        return 0

    def _playback_label_prefix(self) -> str:
        return "Frame" if self.state.source_kind == "video" else "Image"

    def _configure_playback(self) -> None:
        count = self._playback_count()
        if count <= 0:
            self._reset_playback()
            return
        current_index = min(self._current_source_index(), count - 1)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setEnabled(True)
        self.seek_slider.setRange(0, max(count - 1, 0))
        self.seek_slider.setValue(current_index)
        self.seek_slider.blockSignals(False)
        can_advance = count > 1
        self.play_button.setEnabled(can_advance)
        self.step_back_button.setEnabled(can_advance)
        self.step_forward_button.setEnabled(can_advance)
        self.frame_label.setText(f"{self._playback_label_prefix()} {current_index + 1}/{count}")

    def _result_matches_current_view(self, result) -> bool:
        if result is None:
            return False
        if isinstance(self.state.results, list):
            return True
        if self.state.source_kind == "directory":
            current_index = self._current_source_index()
            return bool(self.state.source_items) and current_index < len(self.state.source_items) and result.source == self.state.source_items[current_index]
        if self.state.source_kind == "video":
            return (result.frame_index or 0) == self._current_source_index()
        return True

    def _display_current_result(self) -> None:
        self.state.current_frame_index = self.seek_slider.value()
        self._sync_current_mask_state()
        count = self._playback_count()
        if count > 0:
            self.frame_label.setText(f"{self._playback_label_prefix()} {self.seek_slider.value() + 1}/{count}")
        self._update_result_panel()
        self._refresh_preview()

    def _current_mask_preview(self):
        if self.state.mask_input is None or not self._mask_visible_for_filters():
            return None
        return preview_mask(self.state.mask_input)

    def _current_manual_mask_preview(self):
        if self.state.manual_mask_input is None or not self._manual_mask_visible_for_filters():
            return None
        return preview_mask(self.state.manual_mask_input)

    def _refresh_preview(self) -> None:
        result = self._current_result()
        if result is not None and not self._result_matches_current_view(result):
            result = None

        if result is not None:
            if result.image is not None:
                frame = result.image
            elif self.state.source_kind == "video" and self.state.source_path:
                frame = read_video_frame(self.state.source_path, result.frame_index or 0)
            elif result.source:
                frame = to_bgr_image(result.source)
            elif self.state.source_path and self.state.source_kind != "video":
                frame = to_bgr_image(self.state.source_path)
            else:
                return
            overlay = render_overlay(
                frame,
                result,
                opacity=self.opacity_slider.value() / 100.0,
                show_labels=self.show_labels_checkbox.isChecked(),
                show_masks=self.show_masks_checkbox.isChecked(),
                show_track_ids=self.show_track_ids_checkbox.isChecked(),
                visible_track_ids=self._selected_view_track_ids(),
                visible_labels=self._selected_view_labels(),
            )
            self.preview_canvas.set_image(overlay)
        elif self.state.source_kind == "video" and self.state.source_path:
            self.preview_canvas.set_image(read_video_frame(self.state.source_path, self._current_source_index()))
        elif self.state.source_kind == "directory" and self.state.source_items:
            current_index = min(self._current_source_index(), len(self.state.source_items) - 1)
            self.preview_canvas.set_image(to_bgr_image(self.state.source_items[current_index]))
        elif self.state.source_path:
            self.preview_canvas.set_image(to_bgr_image(self.state.source_path))
        self.preview_canvas.set_prompt_overlays(self.state.points, self.state.boxes)
        self.preview_canvas.set_prompt_mask_preview(self._current_mask_preview())
        self.preview_canvas.set_manual_mask_preview(self._current_manual_mask_preview())
        self._update_prompt_summary()

    def _step_sequence(self, delta: int) -> None:
        count = self._playback_count()
        if count <= 0:
            return
        next_index = (self.seek_slider.value() + delta) % count
        self.seek_slider.setValue(next_index)

    def _toggle_playback(self) -> None:
        if self._playback_count() <= 1:
            return
        self.state.playing = not self.state.playing
        self.play_button.setText("Pause" if self.state.playing else "Play")
        if self.state.playing:
            self.play_timer.start()
        else:
            self.play_timer.stop()

    def _advance_playback(self) -> None:
        if self._playback_count() <= 1:
            return
        self._step_sequence(1)

    def _reset_playback(self) -> None:
        self.play_timer.stop()
        self.state.playing = False
        self.play_button.setText("Play")
        self.play_button.setEnabled(False)
        self.step_back_button.setEnabled(False)
        self.step_forward_button.setEnabled(False)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setRange(0, 0)
        self.frame_label.setText("Frame 0/0")

    def _export_masks_only(self) -> None:
        if self.backend is None or self.state.results is None:
            QtWidgets.QMessageBox.warning(self, "No Results", "Run inference before exporting masks.")
            return
        if self.current_task is not None:
            QtWidgets.QMessageBox.warning(self, "Busy", "Wait for the current task to finish first.")
            return
        export_dir = self.export_dir_edit.text().strip() or self.state.export_dir
        if not export_dir:
            QtWidgets.QMessageBox.warning(self, "Missing Export Directory", "Select an export directory first.")
            return
        def job(progress_callback=None, cancel_callback=None):
            if cancel_callback is not None and cancel_callback():
                return None
            return self.backend.save_results(
                self.state.results,
                output_dir=None,
                mask_dir=export_dir,
                save_overlay=False,
                save_json=False,
                merged_mask_only=self.merge_masks_only_checkbox.isChecked(),
                invert_mask=self.invert_mask_export_checkbox.isChecked(),
                manual_masks_by_key=self.state.manual_masks_by_key or None,
                dilation_pixels=int(self.export_dilation_spin.value()),
                progress_callback=progress_callback,
            )

        self.current_task = BackendTask(job)
        self.current_task_mode = "export"
        self.current_task.signals.result.connect(lambda result: self._handle_export_result(result, export_dir))
        self.current_task.signals.error.connect(self._handle_export_error)
        self.current_task.signals.finished.connect(self._task_finished)
        self.current_task.signals.progress.connect(self._handle_progress)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Exporting masks")
        self.statusBar().showMessage("Exporting masks")
        self._append_log(f"Exporting masks to {export_dir}")
        self.thread_pool.start(self.current_task)

    def _handle_export_result(self, _result, export_dir: str) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Export completed")
        self.statusBar().showMessage(f"Masks exported to {export_dir}")
        self._append_log(f"Exported masks to {export_dir}")

    def _handle_export_error(self, message: str) -> None:
        self._append_log("Mask export failed")
        self._append_log(message)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Export failed")
        self.statusBar().showMessage("Mask export failed")
        QtWidgets.QMessageBox.critical(self, "Export Error", message)

    def _format_source_key_label(self, key: str) -> str:
        if "::frame:" in key:
            source, frame_text = key.rsplit("::frame:", 1)
            if frame_text.isdigit():
                return f"{Path(source).name} frame {int(frame_text) + 1}"
            return f"{Path(source).name} {frame_text}"
        return Path(key).name if key else "current view"

    def _advance_batch_preview(self, index: int) -> None:
        if not self._streaming_batch_mode or self.current_task_mode != "run" or self.current_task is None:
            return
        if self._streaming_batch_total <= 0:
            return
        target_index = max(0, min(index, self._streaming_batch_total - 1))
        self.seek_slider.setValue(target_index)
        self.state.current_frame_index = target_index
        self._display_current_result()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

























































