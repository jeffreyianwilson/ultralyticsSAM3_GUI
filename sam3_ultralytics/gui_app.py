"""Main PySide6 application window."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .backend import SAM3Ultralytics
from .cache_store import ArchiveMaskArray, CacheStore, load_cached_mask, load_cached_result
from .gui_state import GUIState, ViewFilterState
from .gui_widgets import PreviewCanvas
from .gui_workers import BackendTask
from .io_utils import list_image_directory, normalize_mask_input, preview_mask, read_video_frame, to_bgr_image, video_frame_count
from .exceptions import InferenceCancelledError
from .project_io import (
    PROJECT_SUFFIX,
    decode_path,
    decode_view_filter_state,
    encode_path,
    encode_view_filter_state,
    load_project_document,
    project_cache_dir,
    save_project_document,
)
from .schemas import PredictionResult, SegmentationObject
from .visualization import render_overlay


class SAM3MainWindow(QtWidgets.QMainWindow):
    """Compact single-window desktop app for SAM 3 workflows."""

    def __init__(self) -> None:
        super().__init__()
        self._base_window_title = "sam3_ultralytics"
        self.setWindowTitle(self._base_window_title)
        self.resize(1320, 860)
        self._initial_maximize_pending = True
        self.default_cache_root = Path.cwd() / ".sam3_cache"
        self.state = GUIState()
        self.cache_store = CacheStore.create(self.default_cache_root)
        self.backend: SAM3Ultralytics | None = None
        self._backend_signature: tuple | None = None
        self.current_task: BackendTask | None = None
        self.current_task_mode: str | None = None
        self._streaming_batch_mode = False
        self._streaming_batch_total = 0
        self._sequence_run_context: dict | None = None
        self._manual_mask_clipboard: np.ndarray | None = None
        self._preview_frame_cache_key: object | None = None
        self._preview_frame_cache_image: np.ndarray | None = None
        self._preview_overlay_cache_key: object | None = None
        self._preview_overlay_cache_image: np.ndarray | None = None
        # Use a dedicated single-thread pool so backend/model access does not
        # hop across worker threads between sequence items.
        self.thread_pool = QtCore.QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.thread_pool.setExpiryTimeout(-1)
        self._run_cache_epoch = 0
        self._suspend_dirty_tracking = False
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
        file_menu = self.menuBar().addMenu("&File")
        self.new_project_action = file_menu.addAction("New Project")
        self.new_project_action.setShortcut(QtGui.QKeySequence.StandardKey.New)
        self.open_project_action = file_menu.addAction("Open Project...")
        self.open_project_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.save_project_action = file_menu.addAction("Save Project")
        self.save_project_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_project_as_action = file_menu.addAction("Save Project As...")
        self.save_project_as_action.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        controls_scroll.setMinimumWidth(340)
        controls_scroll.setMaximumWidth(420)
        splitter.addWidget(controls_scroll)
        self.controls_scroll = controls_scroll
        controls = QtWidgets.QWidget()
        self.controls_panel = controls
        controls_scroll.setWidget(controls)
        controls.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        controls_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        controls_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        form = QtWidgets.QFormLayout()
        form.setVerticalSpacing(6)
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
        self.controls_toolbox = toolbox
        toolbox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        if toolbox.layout() is not None:
            toolbox.layout().setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        controls_layout.addWidget(toolbox)

        inference_page = QtWidgets.QWidget()
        inference_page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        inference_root = QtWidgets.QVBoxLayout(inference_page)
        inference_root.setContentsMargins(0, 0, 0, 0)
        inference_root.setSpacing(0)
        self.inference_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.inference_splitter.setChildrenCollapsible(False)
        inference_root.addWidget(self.inference_splitter)
        inference_content = QtWidgets.QWidget()
        self.inference_splitter.addWidget(inference_content)
        inference_layout = QtWidgets.QVBoxLayout(inference_content)
        inference_layout.setContentsMargins(6, 6, 6, 6)
        inference_layout.setSpacing(6)
        inference_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
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
        inference_spacer = QtWidgets.QWidget()
        inference_spacer.setMinimumHeight(0)
        self.inference_splitter.addWidget(inference_spacer)
        self.inference_splitter.setStretchFactor(0, 1)
        self.inference_splitter.setStretchFactor(1, 0)
        toolbox.addItem(inference_page, "Inference")

        manual_page = QtWidgets.QWidget()
        manual_page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        manual_root = QtWidgets.QVBoxLayout(manual_page)
        manual_root.setContentsMargins(0, 0, 0, 0)
        manual_root.setSpacing(0)
        self.manual_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.manual_splitter.setChildrenCollapsible(False)
        manual_root.addWidget(self.manual_splitter)
        manual_content = QtWidgets.QWidget()
        self.manual_splitter.addWidget(manual_content)
        manual_layout = QtWidgets.QVBoxLayout(manual_content)
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(6)
        manual_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
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
        manual_range_row = QtWidgets.QHBoxLayout()
        self.copy_manual_prev_spin = QtWidgets.QSpinBox()
        self.copy_manual_prev_spin.setRange(0, 100000)
        self.copy_manual_prev_spin.setValue(0)
        self.copy_manual_next_spin = QtWidgets.QSpinBox()
        self.copy_manual_next_spin.setRange(0, 100000)
        self.copy_manual_next_spin.setValue(0)
        self.copy_manual_range_button = QtWidgets.QPushButton("Copy Range")
        manual_range_row.addWidget(QtWidgets.QLabel("Prev"))
        manual_range_row.addWidget(self.copy_manual_prev_spin)
        manual_range_row.addWidget(QtWidgets.QLabel("Next"))
        manual_range_row.addWidget(self.copy_manual_next_spin)
        manual_range_row.addWidget(self.copy_manual_range_button)
        manual_layout.addLayout(manual_range_row)
        manual_spacer = QtWidgets.QWidget()
        manual_spacer.setMinimumHeight(0)
        self.manual_splitter.addWidget(manual_spacer)
        self.manual_splitter.setStretchFactor(0, 1)
        self.manual_splitter.setStretchFactor(1, 0)
        toolbox.addItem(manual_page, "Manual Masks")

        view_page = QtWidgets.QWidget()
        view_page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        view_root = QtWidgets.QVBoxLayout(view_page)
        view_root.setContentsMargins(0, 0, 0, 0)
        view_root.setSpacing(0)
        self.view_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.view_splitter.setChildrenCollapsible(False)
        view_root.addWidget(self.view_splitter)
        view_content = QtWidgets.QWidget()
        self.view_splitter.addWidget(view_content)
        view_layout = QtWidgets.QVBoxLayout(view_content)
        view_layout.setContentsMargins(6, 6, 6, 6)
        view_layout.setSpacing(6)
        view_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        view_form = QtWidgets.QFormLayout()
        view_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        view_form.setVerticalSpacing(6)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.show_labels_checkbox = QtWidgets.QCheckBox("Labels")
        self.show_labels_checkbox.setChecked(True)
        self.show_masks_checkbox = QtWidgets.QCheckBox("Masks")
        self.show_masks_checkbox.setChecked(True)
        self.show_track_ids_checkbox = QtWidgets.QCheckBox("Track IDs")
        self.show_track_ids_checkbox.setChecked(True)
        view_form.addRow("Overlay Opacity", self.opacity_slider)
        view_form.addRow(self.show_labels_checkbox)
        view_form.addRow(self.show_masks_checkbox)
        view_form.addRow(self.show_track_ids_checkbox)
        view_layout.addLayout(view_form)
        self.filter_class_list = QtWidgets.QListWidget()
        self.filter_id_list = QtWidgets.QListWidget()
        self.filter_instance_list = QtWidgets.QListWidget()
        for widget in [self.filter_class_list, self.filter_id_list, self.filter_instance_list]:
            widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            widget.setUniformItemSizes(True)
            widget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.filter_class_button = QtWidgets.QToolButton()
        self.filter_class_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.filter_class_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.filter_class_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.filter_id_button = QtWidgets.QToolButton()
        self.filter_id_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.filter_id_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.filter_id_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.filter_instance_button = QtWidgets.QToolButton()
        self.filter_instance_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.filter_instance_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.filter_instance_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        filter_button_style = (
            "QToolButton {"
            " text-align: left;"
            " padding-left: 6px;"
            " padding-right: 20px;"
            " min-height: 24px;"
            " border: 1px solid #5f5f5f;"
            " border-radius: 4px;"
            " background-color: #343434;"
            "}"
            "QToolButton:hover {"
            " border-color: #8a8a8a;"
            "}"
            "QToolButton:pressed {"
            " background-color: #3d3d3d;"
            "}"
            "QToolButton::menu-indicator {"
            " subcontrol-origin: padding;"
            " subcontrol-position: right center;"
            " right: 6px;"
            "}"
        )
        self.filter_class_button.setStyleSheet(filter_button_style)
        self.filter_id_button.setStyleSheet(filter_button_style)
        self.filter_instance_button.setStyleSheet(filter_button_style)
        self.filter_class_menu = QtWidgets.QMenu(self)
        self.filter_id_menu = QtWidgets.QMenu(self)
        self.filter_instance_menu = QtWidgets.QMenu(self)
        self._attach_filter_menu(self.filter_class_button, self.filter_class_menu, self.filter_class_list)
        self._attach_filter_menu(self.filter_id_button, self.filter_id_menu, self.filter_id_list)
        self._attach_filter_menu(self.filter_instance_button, self.filter_instance_menu, self.filter_instance_list)
        view_layout.addWidget(QtWidgets.QLabel("Filter Class"))
        view_layout.addWidget(self.filter_class_button)
        view_layout.addWidget(QtWidgets.QLabel("Filter ID"))
        view_layout.addWidget(self.filter_id_button)
        view_layout.addWidget(QtWidgets.QLabel("Filter Instance"))
        view_layout.addWidget(self.filter_instance_button)
        export_separator = QtWidgets.QFrame()
        export_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        export_separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        view_layout.addWidget(export_separator)

        export_dir_row = QtWidgets.QHBoxLayout()
        self.export_dir_edit = QtWidgets.QLineEdit()
        self.export_dir_browse_button = QtWidgets.QPushButton("Browse")
        export_dir_row.addWidget(self.export_dir_edit)
        export_dir_row.addWidget(self.export_dir_browse_button)
        view_layout.addWidget(QtWidgets.QLabel("Export Directory"))
        view_layout.addLayout(export_dir_row)
        self.auto_export_masks_checkbox = QtWidgets.QCheckBox("Auto-export masks after inference")
        self.merge_masks_only_checkbox = QtWidgets.QCheckBox("Export merged masks only")
        self.preserve_source_filename_checkbox = QtWidgets.QCheckBox("Use original source filename for merged masks")
        self.invert_mask_export_checkbox = QtWidgets.QCheckBox("Invert exported masks")
        self.export_filtered_view_checkbox = QtWidgets.QCheckBox("Export only masks visible in View")
        self.export_filtered_view_checkbox.setChecked(True)
        self.export_filtered_view_checkbox.setEnabled(False)
        self.export_dilation_spin = QtWidgets.QSpinBox()
        self.export_dilation_spin.setRange(0, 256)
        self.export_dilation_spin.setValue(0)
        self.export_masks_button = QtWidgets.QPushButton("Export Masks")
        view_layout.addWidget(self.auto_export_masks_checkbox)
        view_layout.addWidget(self.merge_masks_only_checkbox)
        view_layout.addWidget(self.preserve_source_filename_checkbox)
        view_layout.addWidget(self.invert_mask_export_checkbox)
        view_layout.addWidget(self.export_filtered_view_checkbox)
        view_layout.addWidget(QtWidgets.QLabel("Mask Dilation (px)"))
        view_layout.addWidget(self.export_dilation_spin)
        view_layout.addWidget(self.export_masks_button)
        view_spacer = QtWidgets.QWidget()
        view_spacer.setMinimumHeight(0)
        self.view_splitter.addWidget(view_spacer)
        self.view_splitter.setStretchFactor(0, 1)
        self.view_splitter.setStretchFactor(1, 0)
        toolbox.addItem(view_page, "View/Export")

        metadata_page = QtWidgets.QWidget()
        metadata_page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        metadata_layout = QtWidgets.QVBoxLayout(metadata_page)
        metadata_layout.setContentsMargins(6, 6, 6, 6)
        metadata_layout.setSpacing(6)
        metadata_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)
        self.metadata_panel = QtWidgets.QPlainTextEdit()
        self.metadata_panel.setReadOnly(True)
        self.metadata_panel.setMaximumHeight(120)
        metadata_layout.addWidget(self.metadata_panel)
        toolbox.addItem(metadata_page, "Metadata")

        action_row = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.cancel_button)
        controls_layout.addLayout(action_row)

        right = QtWidgets.QWidget()
        splitter.addWidget(right)
        splitter.setCollapsible(0, False)
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
        self.frame_jump_edit = QtWidgets.QLineEdit()
        self.frame_jump_edit.setPlaceholderText("Frame #")
        self.frame_jump_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.frame_jump_edit.setFixedWidth(84)
        self.frame_jump_edit.setEnabled(False)
        self.seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.seek_slider.setEnabled(False)
        self.frame_label = QtWidgets.QLabel("Frame 0/0")
        playback_row.addWidget(self.step_back_button)
        playback_row.addWidget(self.play_button)
        playback_row.addWidget(self.step_forward_button)
        playback_row.addWidget(self.frame_jump_edit)
        playback_row.addWidget(self.seek_slider, stretch=1)
        playback_row.addWidget(self.frame_label)
        right_layout.addLayout(playback_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        right_layout.addWidget(self.progress_bar)

        self.result_panel = QtWidgets.QPlainTextEdit()
        self.result_panel.setReadOnly(True)
        self.result_panel.setMaximumBlockCount(500)
        self.result_panel.setMinimumHeight(150)
        right_layout.addWidget(self.result_panel)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+X"), self, activated=self._clear_all)

        self.statusBar().hide()
        splitter.setSizes([380, 1200])

        self.new_project_action.triggered.connect(self._new_project)
        self.open_project_action.triggered.connect(self._open_project)
        self.save_project_action.triggered.connect(self._save_project)
        self.save_project_as_action.triggered.connect(self._save_project_as)
        self.model_browse_button.clicked.connect(self._browse_model)
        self.cache_browse_button.clicked.connect(self._browse_cache_dir)
        self.clear_cache_button.clicked.connect(self._clear_cache_dir)
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
        self.frame_jump_edit.returnPressed.connect(self._jump_to_frame)

        self.point_tool_button.toggled.connect(lambda checked: self._set_tool("point" if checked else "none"))
        self.box_tool_button.toggled.connect(lambda checked: self._set_tool("box" if checked else "none"))
        self.manual_mask_tool_button.toggled.connect(lambda checked: self._set_tool("manual_mask" if checked else "none"))
        self.brush_slider.valueChanged.connect(self._set_brush_size)
        self.copy_manual_mask_button.clicked.connect(self._copy_manual_mask_to_clipboard)
        self.paste_manual_mask_button.clicked.connect(self._paste_manual_mask_to_current_frame)
        self.copy_manual_mask_all_button.clicked.connect(self._copy_manual_mask_to_all_frames)
        self.copy_manual_range_button.clicked.connect(self._copy_manual_mask_to_range)
        self.clear_manual_mask_button.clicked.connect(self._clear_current_manual_mask)
        for widget in [
            self.opacity_slider,
            self.show_labels_checkbox,
            self.show_masks_checkbox,
            self.show_track_ids_checkbox,
        ]:
            signal = widget.valueChanged if isinstance(widget, QtWidgets.QSlider) else widget.toggled
            signal.connect(self._refresh_preview)
        self.filter_class_list.itemChanged.connect(self._handle_class_filter_change)
        self.filter_id_list.itemChanged.connect(self._handle_id_filter_change)
        self.filter_instance_list.itemChanged.connect(self._handle_instance_filter_change)
        toolbox.currentChanged.connect(lambda _index: self._update_left_panel_layout())
        splitter.splitterMoved.connect(lambda _pos, _index: self._update_left_panel_layout())

        self.model_combo.currentTextChanged.connect(lambda _value: self._mark_project_dirty())
        self.device_combo.currentTextChanged.connect(lambda _value: self._mark_project_dirty())
        self.run_scope_combo.currentIndexChanged.connect(lambda _index: self._mark_project_dirty())
        self.confidence_spin.valueChanged.connect(lambda _value: self._mark_project_dirty())
        self.downscale_inference_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.inference_scale_spin.valueChanged.connect(lambda _value: self._mark_project_dirty())
        self.text_prompt_edit.textChanged.connect(lambda _text: self._mark_project_dirty())
        self.append_inference_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.export_dir_edit.textChanged.connect(lambda _text: self._mark_project_dirty())
        self.auto_export_masks_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.merge_masks_only_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.preserve_source_filename_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.invert_mask_export_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.export_filtered_view_checkbox.toggled.connect(lambda _checked: self._mark_project_dirty())
        self.export_dilation_spin.valueChanged.connect(lambda _value: self._mark_project_dirty())

    def _apply_defaults(self) -> None:
        self.model_combo.addItem(r"D:\cache\models\sam3.pt")
        self.state.cache_dir = str(self.cache_store.root)
        self.cache_dir_edit.setText(str(self.cache_store.root))
        self.cache_dir_edit.setReadOnly(False)
        self.cache_browse_button.setEnabled(True)
        self.state.inference_scale_enabled = False
        self.state.inference_scale = 1.0
        self.downscale_inference_checkbox.setChecked(False)
        self.inference_scale_spin.setValue(1.0)
        self.opacity_slider.setValue(45)
        self._set_brush_size(self.brush_slider.value())
        self._refresh_view_filters()
        self._reset_progress()
        self._set_result_summary("No inference results yet.")
        self._update_window_title()
        QtCore.QTimer.singleShot(0, self._update_left_panel_layout)

    def _set_result_summary(self, text: str) -> None:
        self.metadata_panel.setPlainText(text)

    def _project_display_name(self) -> str:
        return self.state.project_name or "Unsaved Session"

    def _update_window_title(self) -> None:
        suffix = self._project_display_name()
        dirty = "*" if self.state.dirty else ""
        self.setWindowTitle(f"{self._base_window_title} - {suffix}{dirty}")

    def _mark_project_dirty(self) -> None:
        if self._suspend_dirty_tracking:
            return
        if not self.state.dirty:
            self.state.dirty = True
            self._update_window_title()

    def _mark_project_clean(self) -> None:
        self.state.dirty = False
        self._update_window_title()

    def _project_cache_root(self, project_path: str | Path) -> Path:
        return project_cache_dir(project_path)

    @staticmethod
    def _project_name_from_path(project_path: str | Path) -> str:
        name = Path(project_path).name
        if name.endswith(PROJECT_SUFFIX):
            return name[: -len(PROJECT_SUFFIX)]
        return Path(project_path).stem

    @staticmethod
    def _normalize_project_save_path(path: str) -> Path:
        candidate = Path(path)
        if candidate.suffix:
            return candidate
        return candidate.with_name(f"{candidate.name}{PROJECT_SUFFIX}")

    def _set_project_cache_mode(self, active: bool) -> None:
        self.cache_dir_edit.setReadOnly(active)
        self.cache_browse_button.setEnabled(not active)

    def _has_active_project(self) -> bool:
        return bool(self.state.project_path)

    def _busy_for_project_action(self) -> bool:
        if self.current_task is None:
            return False
        QtWidgets.QMessageBox.warning(self, "Busy", "Wait for the current task to finish before changing projects.")
        return True

    def _confirm_discard_unsaved_changes(self) -> bool:
        if not self.state.dirty:
            return True
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Unsaved Changes")
        dialog.setText(f"Save changes to {self._project_display_name()} before continuing?")
        save_button = dialog.addButton("Save", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        discard_button = dialog.addButton("Discard", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = dialog.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(save_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is cancel_button:
            return False
        if clicked is discard_button:
            return True
        if clicked is save_button:
            return self._save_project()
        return True

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._initial_maximize_pending:
            self._initial_maximize_pending = False
            self.showMaximized()
        QtCore.QTimer.singleShot(0, self._update_left_panel_layout)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_left_panel_layout()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._confirm_discard_unsaved_changes():
            event.ignore()
            return
        super().closeEvent(event)

    def _iter_result_objects(self):
        result = self._current_result()
        if result is None or not self._result_matches_current_view(result):
            return []
        return self._result_objects(result, apply_filters=False, include_suppressed=False)

    def _result_source_scope(self, result: PredictionResult) -> str:
        if result.mode == "video":
            return str(result.source or self.state.source_path or "")
        return str(result.source or self.state.source_path or "")

    def _result_frame_key(self, result: PredictionResult) -> str:
        if result.mode == "video":
            return f"{self._result_source_scope(result)}::frame:{int(result.frame_index or 0)}"
        return str(result.source or self._current_source_key() or self.state.source_path or "")

    @staticmethod
    def _object_sort_key(obj: SegmentationObject) -> tuple[float, float, float, int]:
        if obj.box is not None:
            x1, y1, x2, y2 = obj.box
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            area = max(0.0, (x2 - x1) * (y2 - y1))
            return (center_x, center_y, area, obj.object_index)
        mask = np.asarray(obj.mask)
        ys, xs = np.where(mask if mask.dtype == np.bool_ else (mask > 0))
        if ys.size:
            return (float(xs.mean()), float(ys.mean()), float(mask.sum()), obj.object_index)
        return (0.0, 0.0, 0.0, obj.object_index)

    @staticmethod
    def _is_numeric_label(label: str | None) -> bool:
        if label is None:
            return False
        text = str(label).strip()
        return bool(text) and text.isdigit()

    @staticmethod
    def _prompt_text_labels(result: PredictionResult) -> list[str]:
        raw = result.prompt_metadata.get("texts") if isinstance(result.prompt_metadata, dict) else None
        if isinstance(raw, str):
            values = [part.strip() for part in raw.split(",") if part.strip()]
            return values
        if isinstance(raw, (list, tuple)):
            values: list[str] = []
            for item in raw:
                text = str(item).strip()
                if text:
                    values.append(text)
            return values
        return []

    def _display_label_for_object(self, result: PredictionResult, obj: SegmentationObject) -> str:
        raw_label = str(obj.label).strip() if obj.label is not None else ""
        if raw_label and not self._is_numeric_label(raw_label):
            return raw_label
        prompt_labels = self._prompt_text_labels(result)
        if prompt_labels:
            if raw_label.isdigit():
                numeric_index = int(raw_label)
                if 0 <= numeric_index < len(prompt_labels):
                    return prompt_labels[numeric_index]
            if len(prompt_labels) == 1:
                return prompt_labels[0]
        return "Unlabeled"

    def _manual_instance_key(self, frame_key: str | None = None) -> str:
        resolved = frame_key or self._current_source_key() or "frame"
        return f"{resolved}|manualMask|0"

    def _object_instance_key(self, result: PredictionResult, obj: SegmentationObject) -> str:
        box = obj.box or (0.0, 0.0, 0.0, 0.0)
        rounded_box = tuple(int(round(value)) for value in box)
        label = self._display_label_for_object(result, obj)
        return f"{self._result_frame_key(result)}|{obj.object_index}|{obj.track_id}|{label}|{rounded_box}"

    def _object_track_scope_key(self, result: PredictionResult, track_id: int) -> str:
        return f"{self._result_source_scope(result)}|{int(track_id)}"

    def _suppressed_instance_keys(self, result: PredictionResult) -> set[str]:
        return self.state.suppressed_objects_by_key.get(self._result_frame_key(result), set())

    def _suppressed_track_ids(self, result: PredictionResult) -> set[int]:
        return self.state.suppressed_track_ids_by_source.get(self._result_source_scope(result), set())

    def _is_object_suppressed(self, result: PredictionResult, obj: SegmentationObject) -> bool:
        if self._object_instance_key(result, obj) in self._suppressed_instance_keys(result):
            return True
        return obj.track_id is not None and int(obj.track_id) in self._suppressed_track_ids(result)

    def _filter_state(self) -> ViewFilterState:
        return self.state.view_filters

    @staticmethod
    def _effective_selection(options: list[object], selected: set[object], all_selected: bool) -> set[object] | None:
        if not options:
            return None
        if all_selected:
            return None
        return set(selected)

    @staticmethod
    def _set_checkable_items(
        widget: QtWidgets.QListWidget,
        entries: list[tuple[str, object]],
        selected_values: set[object],
        *,
        all_label: str,
        all_selected: bool,
    ) -> None:
        widget.blockSignals(True)
        widget.clear()
        all_item = QtWidgets.QListWidgetItem(all_label)
        all_item.setFlags(all_item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
        all_item.setData(QtCore.Qt.ItemDataRole.UserRole, None)
        all_item.setCheckState(QtCore.Qt.CheckState.Checked if all_selected else QtCore.Qt.CheckState.Unchecked)
        widget.addItem(all_item)
        for label, value in entries:
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
            checked = all_selected or value in selected_values
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            widget.addItem(item)
        widget.blockSignals(False)

    @staticmethod
    def _sync_selection_with_options(
        options: list[object],
        selected: set[object],
        *,
        all_selected: bool,
    ) -> tuple[set[object], bool]:
        option_values = set(options)
        if not options:
            return set(), True
        if all_selected:
            return set(option_values), True
        filtered = set(selected) & option_values
        if len(filtered) == len(option_values):
            return set(option_values), True
        return filtered, False

    @staticmethod
    def _apply_toggle_to_selection(
        options: list[object],
        selected: set[object],
        all_selected: bool,
        *,
        changed_value: object,
        checked: bool,
    ) -> tuple[set[object], bool]:
        option_values = set(options)
        if changed_value is None:
            if checked:
                return set(option_values), True
            return set(), False
        updated = set(option_values) if all_selected else set(selected)
        if checked:
            updated.add(changed_value)
        else:
            updated.discard(changed_value)
        updated &= option_values
        if len(updated) == len(option_values):
            return set(option_values), True
        return updated, False

    def _current_frame_result(self) -> PredictionResult | None:
        result = self._current_result()
        if result is None or not self._result_matches_current_view(result):
            return None
        return result

    def _current_frame_objects(self, *, include_suppressed: bool = False) -> list[SegmentationObject]:
        result = self._current_frame_result()
        if result is None:
            return []
        objects: list[SegmentationObject] = []
        for obj in result.objects:
            if not include_suppressed and self._is_object_suppressed(result, obj):
                continue
            objects.append(obj)
        return objects

    def _manual_mask_present_for_current_frame(self) -> bool:
        key = self._current_source_key()
        if self.state.manual_mask_input is not None:
            return True
        if key is None:
            return False
        return key in self.state.manual_masks_by_key

    def _attach_filter_menu(
        self,
        button: QtWidgets.QToolButton,
        menu: QtWidgets.QMenu,
        widget: QtWidgets.QListWidget,
    ) -> None:
        container = QtWidgets.QWidget(menu)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)
        layout.addWidget(widget)
        container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        widget_action = QtWidgets.QWidgetAction(menu)
        widget_action.setDefaultWidget(container)
        menu.addAction(widget_action)
        widget.setProperty("owner_button", button)
        button.setMenu(menu)

    def _resize_filter_list(self, widget: QtWidgets.QListWidget) -> None:
        row_height = widget.sizeHintForRow(0)
        if row_height <= 0:
            row_height = widget.fontMetrics().height() + 8
        frame = widget.frameWidth() * 2
        item_count = max(widget.count(), 1)
        visible_count = min(item_count, 10)
        height = frame + (row_height * visible_count)
        owner_button = widget.property("owner_button")
        button_width = owner_button.width() if isinstance(owner_button, QtWidgets.QAbstractButton) else 220
        width = max(button_width, 220)
        widget.setFixedHeight(height)
        widget.setFixedWidth(width)
        parent_widget = widget.parentWidget()
        if parent_widget is not None:
            parent_widget.setFixedWidth(width + 12)
            parent_widget.setFixedHeight(height + 12)
        if isinstance(owner_button, QtWidgets.QAbstractButton) and owner_button.menu() is not None:
            owner_button.menu().setFixedSize(width + 12, height + 12)

    @staticmethod
    def _button_text_for_values(
        default_label: str,
        selected_values: set[object] | None,
        formatter,
    ) -> str:
        if selected_values is None:
            return default_label
        if len(selected_values) == 0:
            return "None"
        labels = [formatter(value) for value in selected_values]
        if not labels:
            return default_label
        if len(labels) <= 2:
            return ", ".join(labels)
        return f"{len(labels)} selected"

    @staticmethod
    def _set_toolbutton_elided_text(button: QtWidgets.QToolButton, text: str) -> None:
        button.setText(text)

    def _update_left_panel_layout(self) -> None:
        if not hasattr(self, "controls_panel"):
            return
        current = self.controls_toolbox.currentWidget() if hasattr(self, "controls_toolbox") else None
        if current is not None:
            current.adjustSize()
        tab_buttons = self.controls_toolbox.findChildren(QtWidgets.QAbstractButton, "qt_toolbox_toolboxbutton")
        tabs_height = sum(button.sizeHint().height() for button in tab_buttons)
        page_height = 0
        if current is not None:
            page_height = max(current.sizeHint().height(), current.minimumSizeHint().height())
            if current.layout() is not None:
                page_height = max(page_height, current.layout().sizeHint().height())
        frame = self.controls_toolbox.frameWidth() * 2
        other_height = 0
        layout = self.controls_panel.layout()
        if layout is not None:
            for index in range(layout.count()):
                item = layout.itemAt(index)
                widget = item.widget()
                child_layout = item.layout()
                if widget is self.controls_toolbox:
                    continue
                if widget is not None:
                    other_height += widget.sizeHint().height()
                elif child_layout is not None:
                    other_height += child_layout.sizeHint().height()
            spacing_count = max(layout.count() - 1, 0)
            other_height += layout.spacing() * spacing_count
            other_height += layout.contentsMargins().top() + layout.contentsMargins().bottom()
        available_height = max(self.controls_scroll.viewport().height() - other_height, tabs_height + 80)
        self.controls_toolbox.setFixedHeight(min(tabs_height + page_height + frame + 4, available_height))
        self.controls_toolbox.updateGeometry()
        if layout is not None:
            self.controls_panel.setFixedHeight(layout.sizeHint().height())
        self.controls_panel.adjustSize()
        self.controls_scroll.widget().updateGeometry()
        self._refresh_filter_button_labels()

    def _refresh_filter_button_labels(self) -> None:
        filters = self._filter_state()
        active_class_values = self._effective_selection(filters.class_options, filters.selected_classes, filters.all_classes_selected)
        active_id_values = self._effective_selection(filters.id_options, filters.selected_ids, filters.all_ids_selected)
        active_instance_values = self._effective_selection(
            [key for _label, key in filters.instance_options],
            filters.selected_instances,
            filters.all_instances_selected,
        )
        instance_labels_by_key = {key: label for label, key in filters.instance_options}
        self._set_toolbutton_elided_text(
            self.filter_class_button,
            self._button_text_for_values("All classes", active_class_values, lambda value: str(value)),
        )
        self._set_toolbutton_elided_text(
            self.filter_id_button,
            self._button_text_for_values("All IDs", active_id_values, lambda value: str(int(value))),
        )
        self._set_toolbutton_elided_text(
            self.filter_instance_button,
            self._button_text_for_values(
                "All instances",
                active_instance_values,
                lambda value: instance_labels_by_key.get(str(value), str(value)),
            ),
        )

    def _selected_view_instance_keys(self) -> set[str] | None:
        filters = self._filter_state()
        values = self._effective_selection(
            [key for _label, key in filters.instance_options],
            filters.selected_instances,
            filters.all_instances_selected,
        )
        if values is None:
            return None
        return {str(value) for value in values}

    def _result_objects(
        self,
        result: PredictionResult | None,
        *,
        apply_filters: bool,
        include_suppressed: bool,
        filter_state: ViewFilterState | None = None,
    ) -> list[SegmentationObject]:
        if result is None:
            return []
        objects: list[SegmentationObject] = []
        if apply_filters:
            if filter_state is None:
                visible_track_ids = self._selected_view_track_ids()
                visible_labels = self._selected_view_labels()
                visible_instance_keys = self._selected_view_instance_keys()
            else:
                visible_track_ids = self._effective_selection(
                    filter_state.id_options,
                    filter_state.selected_ids,
                    filter_state.all_ids_selected,
                )
                if visible_track_ids is not None:
                    visible_track_ids = {int(value) for value in visible_track_ids}
                visible_labels = self._effective_selection(
                    filter_state.class_options,
                    filter_state.selected_classes,
                    filter_state.all_classes_selected,
                )
                if visible_labels is not None:
                    visible_labels = {str(value) for value in visible_labels}
                visible_instance_keys = self._effective_selection(
                    [key for _label, key in filter_state.instance_options],
                    filter_state.selected_instances,
                    filter_state.all_instances_selected,
                )
                if visible_instance_keys is not None:
                    visible_instance_keys = {str(value) for value in visible_instance_keys}
        else:
            visible_track_ids = None
            visible_labels = None
            visible_instance_keys = None
        for obj in result.objects:
            if not include_suppressed and self._is_object_suppressed(result, obj):
                continue
            if visible_track_ids is not None and obj.track_id not in visible_track_ids:
                continue
            if visible_labels is not None and self._display_label_for_object(result, obj) not in visible_labels:
                continue
            if visible_instance_keys is not None and self._object_instance_key(result, obj) not in visible_instance_keys:
                continue
            objects.append(obj)
        return objects

    def _instance_filter_candidates(self, result: PredictionResult | None) -> list[tuple[str, str]]:
        selected_track_ids = self._selected_view_track_ids()
        selected_labels = self._selected_view_labels()
        label_counts: dict[str, int] = {}
        items: list[tuple[str, str]] = []
        if result is not None:
            candidates = self._result_objects(result, apply_filters=False, include_suppressed=False)
            if selected_track_ids is not None:
                candidates = [obj for obj in candidates if obj.track_id in selected_track_ids]
            if selected_labels is not None:
                candidates = [obj for obj in candidates if self._display_label_for_object(result, obj) in selected_labels]
            candidates = sorted(candidates, key=self._object_sort_key)
            for obj in candidates:
                key = self._object_instance_key(result, obj)
                label = self._display_label_for_object(result, obj)
                label_counts[label] = label_counts.get(label, 0) + 1
                parts = [f"{label} #{label_counts[label]}"]
                if obj.track_id is not None:
                    parts.append(f"id={obj.track_id}")
                if obj.score is not None:
                    parts.append(f"{obj.score:.2f}")
                items.append((" | ".join(parts), key))
        if self._manual_mask_present_for_current_frame():
            manual_key = self._manual_instance_key(self._result_frame_key(result) if result is not None else self._current_source_key())
            include_manual = True
            if selected_track_ids is not None and 0 not in selected_track_ids:
                include_manual = False
            if selected_labels is not None and "manualMask" not in selected_labels:
                include_manual = False
            if include_manual:
                items.append(("manualMask #1 | id=0", manual_key))
        return items

    def _filtered_result_copy(self, result: PredictionResult, *, apply_view_filters: bool) -> PredictionResult:
        filter_state = None
        if apply_view_filters:
            frame_key = self._result_frame_key(result)
            filter_state = self.state.view_filters_by_frame.get(frame_key)
        objects = self._result_objects(
            result,
            apply_filters=apply_view_filters,
            include_suppressed=False,
            filter_state=filter_state,
        )
        return PredictionResult(
            source=result.source,
            frame_index=result.frame_index,
            mode=result.mode,
            image_size=result.image_size,
            inference_image_size=result.inference_image_size,
            objects=list(objects),
            prompt_metadata=dict(result.prompt_metadata),
            tracking_metadata=dict(result.tracking_metadata),
            timings=dict(result.timings),
            image=result.image,
            prompt_mask=result.prompt_mask,
        )

    def _results_for_export(self, *, apply_view_filters: bool):
        results = self.state.results
        if isinstance(results, list):
            return [None if item is None else self._filtered_result_copy(item, apply_view_filters=apply_view_filters) for item in results]
        if isinstance(results, PredictionResult):
            return self._filtered_result_copy(results, apply_view_filters=apply_view_filters)
        return results

    def _manual_masks_for_export(self, *, apply_view_filters: bool) -> dict[str, object] | None:
        mapping = self.state.manual_masks_by_key or None
        if mapping is None:
            return None
        if not apply_view_filters:
            return mapping

        filtered: dict[str, object] = {}
        for frame_key, mask_ref in mapping.items():
            view_filters = self.state.view_filters_by_frame.get(frame_key)
            if view_filters is None:
                filtered[frame_key] = mask_ref
                continue
            label_values = self._effective_selection(
                view_filters.class_options,
                view_filters.selected_classes,
                view_filters.all_classes_selected,
            )
            if label_values is not None and "manualMask" not in label_values:
                continue
            id_values = self._effective_selection(
                view_filters.id_options,
                view_filters.selected_ids,
                view_filters.all_ids_selected,
            )
            if id_values is not None and 0 not in {int(value) for value in id_values}:
                continue
            instance_values = self._effective_selection(
                [key for _label, key in view_filters.instance_options],
                view_filters.selected_instances,
                view_filters.all_instances_selected,
            )
            if instance_values is not None and self._manual_instance_key(frame_key) not in {str(value) for value in instance_values}:
                continue
            filtered[frame_key] = mask_ref
        return filtered or None

    @staticmethod
    def _iter_export_result_items(results) -> list[PredictionResult]:
        if isinstance(results, list):
            return [item for item in results if isinstance(item, PredictionResult)]
        if isinstance(results, PredictionResult):
            return [results]
        return []

    def _export_would_overwrite_source_imagery(self, export_dir: str | Path, results=None) -> bool:
        if not self.preserve_source_filename_checkbox.isChecked():
            return False
        try:
            target_dir = Path(export_dir).resolve()
        except OSError:
            return False
        candidate_results = self._iter_export_result_items(results)
        if candidate_results:
            for result in candidate_results:
                if result.mode != "image" or not isinstance(result.source, str):
                    continue
                source_path = Path(result.source)
                if source_path.suffix and source_path.parent.resolve() == target_dir:
                    return True
            return False
        if self.state.source_kind == "image" and self.state.source_path:
            source_path = Path(self.state.source_path)
            return bool(source_path.suffix and source_path.parent.resolve() == target_dir)
        if self.state.source_kind == "directory" and self.state.source_path:
            return Path(self.state.source_path).resolve() == target_dir
        return False

    def _warn_source_overwrite_export(self) -> None:
        QtWidgets.QMessageBox.warning(
            self,
            "Export Cancelled",
            "Masks cannot overwrite source imagery. Choose a different export directory or disable original source filenames.",
        )

    def _refresh_view_filters(self) -> None:
        result = self._current_frame_result()
        frame_key = self._result_frame_key(result) if result is not None else self._current_source_key()
        filters_map = self.state.view_filters_by_frame
        is_new_frame_state = False
        if frame_key is None:
            filters = self.state.view_filters
            if filters.frame_key is not None:
                filters = ViewFilterState(frame_key=None)
                self.state.view_filters = filters
                is_new_frame_state = True
        else:
            existing = filters_map.get(frame_key)
            if existing is None:
                existing = ViewFilterState(frame_key=frame_key)
                filters_map[frame_key] = existing
                is_new_frame_state = True
            filters = existing
            self.state.view_filters = filters
        filters.frame_key = frame_key

        base_objects = self._current_frame_objects(include_suppressed=False)
        classes = sorted({self._display_label_for_object(result, obj) for obj in base_objects}) if result is not None else []
        ids = sorted({int(obj.track_id) for obj in base_objects if obj.track_id is not None})
        if self.state.mask_class:
            classes = sorted(set(classes) | {str(self.state.mask_class)})
        if self.state.mask_id is not None:
            ids = sorted(set(ids) | {int(self.state.mask_id)})
        if self._manual_mask_present_for_current_frame():
            classes = sorted(set(classes) | {"manualMask"})
            ids = sorted(set(ids) | {0})

        filters.class_options = classes
        filters.id_options = ids

        if is_new_frame_state:
            filters.all_classes_selected = True
            filters.all_ids_selected = True
            filters.selected_classes = set(classes)
            filters.selected_ids = set(ids)
            filters.all_instances_selected = True
            filters.selected_instances = set()
        else:
            filters.selected_classes, filters.all_classes_selected = self._sync_selection_with_options(
                filters.class_options,
                filters.selected_classes,
                all_selected=filters.all_classes_selected,
            )
            filters.selected_ids, filters.all_ids_selected = self._sync_selection_with_options(
                filters.id_options,
                filters.selected_ids,
                all_selected=filters.all_ids_selected,
            )

        instance_entries = self._instance_filter_candidates(result)
        filters.instance_options = instance_entries
        instance_keys = [key for _label, key in instance_entries]
        if is_new_frame_state:
            filters.selected_instances = set(instance_keys)
            filters.all_instances_selected = True
        else:
            filters.selected_instances, filters.all_instances_selected = self._sync_selection_with_options(
                instance_keys,
                filters.selected_instances,
                all_selected=filters.all_instances_selected,
            )

        self._set_checkable_items(
            self.filter_class_list,
            [(value, value) for value in filters.class_options],
            filters.selected_classes,
            all_label="All Classes",
            all_selected=filters.all_classes_selected,
        )
        self._set_checkable_items(
            self.filter_id_list,
            [(str(value), value) for value in filters.id_options],
            filters.selected_ids,
            all_label="All IDs",
            all_selected=filters.all_ids_selected,
        )
        self._set_checkable_items(
            self.filter_instance_list,
            filters.instance_options,
            filters.selected_instances,
            all_label="All Instances",
            all_selected=filters.all_instances_selected,
        )
        for widget in [self.filter_class_list, self.filter_id_list, self.filter_instance_list]:
            self._resize_filter_list(widget)
        self._refresh_filter_button_labels()

    def _selected_view_track_ids(self) -> set[int] | None:
        filters = self._filter_state()
        values = self._effective_selection(filters.id_options, filters.selected_ids, filters.all_ids_selected)
        if values is None:
            return None
        return {int(value) for value in values}

    def _selected_view_labels(self) -> set[str] | None:
        filters = self._filter_state()
        values = self._effective_selection(filters.class_options, filters.selected_classes, filters.all_classes_selected)
        if values is None:
            return None
        return {str(value) for value in values}

    def _current_selected_objects(self) -> list[tuple[PredictionResult, SegmentationObject]]:
        result = self._current_result()
        if result is None or not self._result_matches_current_view(result):
            return []
        selected_keys = self._selected_view_instance_keys()
        if selected_keys is None:
            return []
        selected_objects: list[tuple[PredictionResult, SegmentationObject]] = []
        for obj in self._result_objects(result, apply_filters=False, include_suppressed=True):
            if self._object_instance_key(result, obj) in selected_keys:
                selected_objects.append((result, obj))
        return selected_objects

    def _delete_selected_mask(self) -> None:
        selected = self._current_selected_objects()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No Instance Selected", "Check one or more instances in the View panel first.")
            return
        deleted_count = 0
        for result, obj in selected:
            frame_key = self._result_frame_key(result)
            self.state.suppressed_objects_by_key.setdefault(frame_key, set()).add(self._object_instance_key(result, obj))
            deleted_count += 1
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._append_log(f"Deleted {deleted_count} selected mask(s)")
        self.statusBar().showMessage("Deleted selected mask")

    def _delete_selected_track(self) -> None:
        selected = self._current_selected_objects()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No Instance Selected", "Check one or more instances in the View panel first.")
            return
        track_scopes: dict[str, set[int]] = {}
        for result, obj in selected:
            if obj.track_id is not None:
                scope = self._result_source_scope(result)
                track_scopes.setdefault(scope, set()).add(int(obj.track_id))
        if not track_scopes:
            QtWidgets.QMessageBox.warning(self, "No Track ID", "None of the selected masks have track IDs.")
            return
        deleted_tracks = 0
        for scope, track_ids in track_scopes.items():
            self.state.suppressed_track_ids_by_source.setdefault(scope, set()).update(track_ids)
            deleted_tracks += len(track_ids)
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._append_log(f"Deleted {deleted_tracks} selected track(s)")
        self.statusBar().showMessage("Deleted selected track")

    def _restore_deleted_masks(self) -> None:
        self.state.suppressed_objects_by_key.clear()
        self.state.suppressed_track_ids_by_source.clear()
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._append_log("Restored deleted masks")
        self.statusBar().showMessage("Restored deleted masks")

    def _mask_visible_for_filters(self) -> bool:
        visible_ids = self._selected_view_track_ids()
        visible_labels = self._selected_view_labels()
        prompt_class = str(self.state.mask_class).strip() if self.state.mask_class else "Unlabeled"
        if visible_ids is not None and self.state.mask_id not in visible_ids:
            return False
        if visible_labels is not None and prompt_class not in visible_labels:
            return False
        return True

    def _manual_mask_visible_for_filters(self) -> bool:
        visible_ids = self._selected_view_track_ids()
        visible_labels = self._selected_view_labels()
        visible_instances = self._selected_view_instance_keys()
        if visible_ids is not None and 0 not in visible_ids:
            return False
        if visible_labels is not None and "manualMask" not in visible_labels:
            return False
        if visible_instances is not None and self._manual_instance_key() not in visible_instances:
            return False
        return True

    def _has_active_view_filters(self) -> bool:
        selected_labels = self._selected_view_labels()
        selected_ids = self._selected_view_track_ids()
        selected_instances = self._selected_view_instance_keys()
        if selected_labels is not None and len(selected_labels) == 0:
            return False
        if selected_ids is not None and len(selected_ids) == 0:
            return False
        if selected_instances is not None and len(selected_instances) == 0:
            return False
        # Any non-empty filter scope (including "All ...") should keep masks visible.
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

    def _apply_filter_change(self, kind: str, item: QtWidgets.QListWidgetItem) -> None:
        filters = self._filter_state()
        if kind == "class":
            options = [value for value in filters.class_options]
            selected = filters.selected_classes
            all_selected = filters.all_classes_selected
        elif kind == "id":
            options = [value for value in filters.id_options]
            selected = filters.selected_ids
            all_selected = filters.all_ids_selected
        else:
            options = [key for _label, key in filters.instance_options]
            selected = filters.selected_instances
            all_selected = filters.all_instances_selected

        changed_value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        updated_selected, updated_all = self._apply_toggle_to_selection(
            options,
            selected,
            all_selected,
            changed_value=changed_value,
            checked=checked,
        )
        if kind == "class":
            filters.selected_classes = {str(value) for value in updated_selected}
            filters.all_classes_selected = updated_all
        elif kind == "id":
            filters.selected_ids = {int(value) for value in updated_selected}
            filters.all_ids_selected = updated_all
        else:
            filters.selected_instances = {str(value) for value in updated_selected}
            filters.all_instances_selected = updated_all

    def _handle_class_filter_change(self, item: QtWidgets.QListWidgetItem) -> None:
        self._apply_filter_change("class", item)
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._mark_project_dirty()

    def _handle_id_filter_change(self, item: QtWidgets.QListWidgetItem) -> None:
        self._apply_filter_change("id", item)
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._mark_project_dirty()

    def _handle_instance_filter_change(self, item: QtWidgets.QListWidgetItem) -> None:
        self._apply_filter_change("instance", item)
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._mark_project_dirty()

    def _append_log(self, message: str) -> None:
        self.result_panel.appendPlainText(message)

    def _reset_progress(self) -> None:
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")

    @staticmethod
    def _format_processing_time_ms(value: float | None) -> str | None:
        if value is None:
            return None
        milliseconds = float(value)
        if milliseconds >= 1000.0:
            return f"{milliseconds / 1000.0:.2f}s"
        return f"{milliseconds:.1f}ms"

    def _result_processing_time_label(self, result: PredictionResult | None) -> str | None:
        if result is None:
            return None
        timings = dict(result.timings or {})
        total_ms = timings.get("backend_total_ms")
        if total_ms is None:
            components = [timings.get(name) for name in ("preprocess", "inference", "postprocess")]
            if any(component is not None for component in components):
                total_ms = sum(float(component or 0.0) for component in components)
        formatted = self._format_processing_time_ms(total_ms)
        if formatted is None:
            return None
        return f" in {formatted}"

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
        scope_label = "the current unsaved session" if not self._has_active_project() else f"project '{self._project_display_name()}'"
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Clear Cache")
        dialog.setText(f"This will completely clear the cache for {scope_label}.")
        dialog.setInformativeText("This cannot be undone.")
        continue_button = dialog.addButton("Continue", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_button = dialog.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        if dialog.clickedButton() is not continue_button:
            return
        self.cache_store.clear()
        self.backend = None
        self._backend_signature = None
        self._clear_all()
        self.state.cache_dir = str(self.cache_store.root)
        self.cache_dir_edit.setText(self.state.cache_dir)
        self.statusBar().showMessage(f"Cleared cache: {self.cache_store.root}")
        self._append_log(f"Cleared cache directory: {self.cache_store.root}")

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
        if self._has_active_project():
            QtWidgets.QMessageBox.warning(self, "Project Cache", "Saved projects always use their sidecar cache directory.")
            return
        self.cache_store.set_root(path)
        self.backend = None
        self._backend_signature = None
        self._clear_all()
        self.state.cache_dir = str(self.cache_store.root)
        self.cache_dir_edit.setText(self.state.cache_dir)
        self.statusBar().showMessage(f"Using cache: {self.cache_store.root}")
        self._append_log(f"Using cache directory: {self.cache_store.root}")
        self._mark_project_dirty()

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
        array = np.asarray(mask)
        if array.dtype == np.bool_:
            return array
        if np.issubdtype(array.dtype, np.floating):
            return array > 0.5
        return array > 0

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

    @staticmethod
    def _serialize_path_mapping(mapping: dict[str, str], base_dir: Path) -> dict[str, dict]:
        return {
            key: encoded
            for key, value in mapping.items()
            if (encoded := encode_path(value, base_dir=base_dir)) is not None
        }

    @staticmethod
    def _deserialize_path_mapping(mapping: dict[str, dict] | None, base_dir: Path) -> dict[str, str]:
        if not mapping:
            return {}
        return {
            key: decoded
            for key, value in mapping.items()
            if (decoded := decode_path(value, base_dir=base_dir)) is not None
        }

    def _result_cache_ref(self, result: PredictionResult | None) -> str | None:
        if result is None:
            return None
        for obj in result.objects:
            mask = obj.mask
            if isinstance(mask, ArchiveMaskArray):
                return str(mask.path)
        prompt_mask = result.prompt_mask
        if isinstance(prompt_mask, ArchiveMaskArray):
            return str(prompt_mask.path)
        return None

    def _serialize_result_refs(self, results, base_dir: Path):
        if results is None:
            return None
        if isinstance(results, list):
            payload: list[dict | None] = []
            for result in results:
                encoded = encode_path(self._result_cache_ref(result), base_dir=base_dir) if result is not None else None
                payload.append(encoded)
            return payload
        return encode_path(self._result_cache_ref(results), base_dir=base_dir)

    def _deserialize_result_refs(self, payload, base_dir: Path):
        missing: list[str] = []

        def load_one(item):
            path = decode_path(item, base_dir=base_dir)
            if not path:
                return None
            if not Path(path).exists():
                missing.append(path)
                return None
            return load_cached_result(path)

        if payload is None:
            return None, missing
        if isinstance(payload, list):
            return [load_one(item) for item in payload], missing
        return load_one(payload), missing

    def _clone_cache_assets_to_store(self, store: CacheStore) -> None:
        prompt_refs: dict[str, str] = {}
        for key, ref in list(self.state.mask_inputs_by_key.items()):
            mask = self._load_cached_mask(ref)
            if mask is None:
                continue
            prompt_refs[key] = store.write_mask("prompt", key, mask)
        self.state.mask_inputs_by_key = prompt_refs

        manual_refs: dict[str, str] = {}
        for key, ref in list(self.state.manual_masks_by_key.items()):
            mask = self._load_cached_mask(ref)
            if mask is None:
                continue
            manual_refs[key] = store.write_mask("manual", key, mask)
        self.state.manual_masks_by_key = manual_refs

        def migrate_result(result: PredictionResult | None, index: int) -> PredictionResult | None:
            if result is None:
                return None
            key = f"{self._cache_token(result.source)}_{result.frame_index if result.frame_index is not None else index}"
            return store.write_result("inference", key, result)

        if isinstance(self.state.results, list):
            self.state.results = [migrate_result(result, index) for index, result in enumerate(self.state.results)]
        elif isinstance(self.state.results, PredictionResult):
            self.state.results = migrate_result(self.state.results, 0)

    def _build_project_payload(self, project_path: Path) -> dict:
        base_dir = project_path.parent
        return {
            "project": {
                "name": self._project_name_from_path(project_path),
                "cache_dir": encode_path(str(self.cache_store.root), base_dir=base_dir),
            },
            "runtime": {
                "model_path": self.model_combo.currentText().strip() or None,
                "device": self.device_combo.currentText(),
                "run_scope": self.run_scope_combo.currentData() or "current",
                "confidence": float(self.confidence_spin.value()),
                "inference_scale_enabled": bool(self.downscale_inference_checkbox.isChecked()),
                "inference_scale": float(self.inference_scale_spin.value()),
                "text_prompt": self.text_prompt_edit.text(),
                "append_inferred_masks": bool(self.append_inference_checkbox.isChecked()),
                "export_dir": encode_path(self.export_dir_edit.text().strip() or None, base_dir=base_dir),
                "auto_export_masks": bool(self.auto_export_masks_checkbox.isChecked()),
                "merge_masks_only": bool(self.merge_masks_only_checkbox.isChecked()),
                "preserve_source_filenames": bool(self.preserve_source_filename_checkbox.isChecked()),
                "invert_exported_masks": bool(self.invert_mask_export_checkbox.isChecked()),
                "export_visible_only": bool(self.export_filtered_view_checkbox.isChecked()),
                "mask_dilation": int(self.export_dilation_spin.value()),
                "overlay_opacity": int(self.opacity_slider.value()),
                "show_labels": bool(self.show_labels_checkbox.isChecked()),
                "show_masks": bool(self.show_masks_checkbox.isChecked()),
                "show_track_ids": bool(self.show_track_ids_checkbox.isChecked()),
                "manual_brush_px": int(self.brush_slider.value()),
            },
            "source": {
                "kind": self.state.source_kind,
                "path": encode_path(self.state.source_path, base_dir=base_dir),
                "current_frame_index": int(self._current_source_index()),
            },
            "session": {
                "source_frame_count": self.state.source_frame_count,
                "mask_paths_by_key": self._serialize_path_mapping(self.state.mask_paths_by_key, base_dir),
                "mask_inputs_by_key": self._serialize_path_mapping(self.state.mask_inputs_by_key, base_dir),
                "mask_sources_by_key": self._serialize_path_mapping(self.state.mask_sources_by_key, base_dir),
                "mask_ids_by_key": {key: int(value) for key, value in self.state.mask_ids_by_key.items()},
                "mask_classes_by_key": dict(self.state.mask_classes_by_key),
                "manual_masks_by_key": self._serialize_path_mapping(self.state.manual_masks_by_key, base_dir),
                "points_by_key": {key: [list(item) for item in value] for key, value in self.state.points_by_key.items()},
                "boxes_by_key": {key: [list(item) for item in value] for key, value in self.state.boxes_by_key.items()},
                "next_mask_id": int(self.state.next_mask_id),
                "results": self._serialize_result_refs(self.state.results, base_dir),
                "suppressed_objects_by_key": {key: sorted(value) for key, value in self.state.suppressed_objects_by_key.items()},
                "suppressed_track_ids_by_source": {key: sorted(int(item) for item in value) for key, value in self.state.suppressed_track_ids_by_source.items()},
                "view_filters_by_frame": {key: encode_view_filter_state(value) for key, value in self.state.view_filters_by_frame.items()},
            },
        }

    def _restore_project_payload(self, payload: dict, project_path: Path) -> None:
        base_dir = project_path.parent
        runtime = dict(payload.get("runtime") or {})
        source = dict(payload.get("source") or {})
        session = dict(payload.get("session") or {})
        project_section = dict(payload.get("project") or {})

        self._suspend_dirty_tracking = True
        try:
            model_path = runtime.get("model_path")
            if model_path:
                if self.model_combo.findText(str(model_path)) == -1:
                    self.model_combo.addItem(str(model_path))
                self.model_combo.setCurrentText(str(model_path))
            device = runtime.get("device")
            if device:
                self.device_combo.setCurrentText(str(device))
            run_scope = runtime.get("run_scope", "current")
            run_index = max(self.run_scope_combo.findData(run_scope), 0)
            self.run_scope_combo.setCurrentIndex(run_index)
            self.confidence_spin.setValue(float(runtime.get("confidence", 0.25)))
            self.downscale_inference_checkbox.setChecked(bool(runtime.get("inference_scale_enabled", False)))
            self.inference_scale_spin.setValue(float(runtime.get("inference_scale", 1.0)))
            self.text_prompt_edit.setText(str(runtime.get("text_prompt") or ""))
            self.append_inference_checkbox.setChecked(bool(runtime.get("append_inferred_masks", False)))
            self.export_dir_edit.setText(decode_path(runtime.get("export_dir"), base_dir=base_dir) or "")
            self.auto_export_masks_checkbox.setChecked(bool(runtime.get("auto_export_masks", False)))
            self.merge_masks_only_checkbox.setChecked(bool(runtime.get("merge_masks_only", False)))
            self.preserve_source_filename_checkbox.setChecked(bool(runtime.get("preserve_source_filenames", False)))
            self.invert_mask_export_checkbox.setChecked(bool(runtime.get("invert_exported_masks", False)))
            self.export_filtered_view_checkbox.setChecked(bool(runtime.get("export_visible_only", False)))
            self.export_dilation_spin.setValue(int(runtime.get("mask_dilation", 0)))
            self.opacity_slider.setValue(int(runtime.get("overlay_opacity", 45)))
            self.show_labels_checkbox.setChecked(bool(runtime.get("show_labels", True)))
            self.show_masks_checkbox.setChecked(bool(runtime.get("show_masks", True)))
            self.show_track_ids_checkbox.setChecked(bool(runtime.get("show_track_ids", True)))
            self.brush_slider.setValue(int(runtime.get("manual_brush_px", self.brush_slider.value())))

            source_path = decode_path(source.get("path"), base_dir=base_dir)
            source_kind = str(source.get("kind") or "image")
            if source_path:
                if source_kind == "directory":
                    self._load_directory_path(source_path)
                elif source_kind == "video":
                    self._load_video_path(source_path)
                else:
                    self._load_image_path(source_path)
            else:
                self._clear_all()

            self.state.mask_paths_by_key = self._deserialize_path_mapping(session.get("mask_paths_by_key"), base_dir)
            self.state.mask_inputs_by_key = self._deserialize_path_mapping(session.get("mask_inputs_by_key"), base_dir)
            self.state.mask_sources_by_key = self._deserialize_path_mapping(session.get("mask_sources_by_key"), base_dir)
            self.state.mask_ids_by_key = {key: int(value) for key, value in dict(session.get("mask_ids_by_key") or {}).items()}
            self.state.mask_classes_by_key = {key: str(value) for key, value in dict(session.get("mask_classes_by_key") or {}).items()}
            self.state.manual_masks_by_key = self._deserialize_path_mapping(session.get("manual_masks_by_key"), base_dir)
            self.state.points_by_key = {
                key: [tuple(item) for item in value]
                for key, value in dict(session.get("points_by_key") or {}).items()
            }
            self.state.boxes_by_key = {
                key: [tuple(item) for item in value]
                for key, value in dict(session.get("boxes_by_key") or {}).items()
            }
            self.state.next_mask_id = int(session.get("next_mask_id", 1))
            self.state.suppressed_objects_by_key = {
                key: set(str(item) for item in value)
                for key, value in dict(session.get("suppressed_objects_by_key") or {}).items()
            }
            self.state.suppressed_track_ids_by_source = {
                key: {int(item) for item in value}
                for key, value in dict(session.get("suppressed_track_ids_by_source") or {}).items()
            }
            self.state.view_filters_by_frame = {
                key: decode_view_filter_state(value)
                for key, value in dict(session.get("view_filters_by_frame") or {}).items()
            }
            self.state.results, missing_results = self._deserialize_result_refs(session.get("results"), base_dir)
            missing_assets = [path for mapping in [self.state.mask_inputs_by_key, self.state.manual_masks_by_key] for path in mapping.values() if not Path(path).exists()]
            missing_assets.extend(missing_results)
            if missing_assets:
                self._append_log(f"Project opened with missing cache assets: {len(missing_assets)}")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing Cache Assets",
                    f"Some cached project assets were missing and could not be restored.\n\nMissing files: {len(missing_assets)}",
                )

            self.state.project_path = str(project_path)
            self.state.project_cache_dir = decode_path(project_section.get("cache_dir"), base_dir=base_dir) or str(self._project_cache_root(project_path))
            self.state.project_name = project_section.get("name") or self._project_name_from_path(project_path)
            self.state.cache_dir = str(self.cache_store.root)
            frame_index = int(source.get("current_frame_index", 0))
            self._configure_playback()
            if self._playback_count() > 0:
                self.seek_slider.setValue(max(0, min(frame_index, self._playback_count() - 1)))
            self._sync_current_mask_state()
            self._refresh_view_filters()
            self._update_result_panel()
            self._refresh_preview()
        finally:
            self._suspend_dirty_tracking = False
        self._mark_project_clean()
        self._set_project_cache_mode(True)
        self.cache_dir_edit.setText(str(self.cache_store.root))
        self._append_log(f"Opened project: {project_path}")

    def _reset_session(self) -> None:
        self._suspend_dirty_tracking = True
        try:
            self._invalidate_preview_caches()
            self.state = GUIState()
            self.state.cache_dir = str(self.cache_store.root)
            self.backend = None
            self._backend_signature = None
            self.preview_timer.stop()
            self.play_timer.stop()
            self.text_prompt_edit.clear()
            self.mask_class_edit.clear()
            self.mask_id_edit.clear()
            self.export_dir_edit.clear()
            self.append_inference_checkbox.setChecked(False)
            self.auto_export_masks_checkbox.setChecked(False)
            self.merge_masks_only_checkbox.setChecked(False)
            self.preserve_source_filename_checkbox.setChecked(False)
            self.invert_mask_export_checkbox.setChecked(False)
            self.export_filtered_view_checkbox.setChecked(False)
            self.export_dilation_spin.setValue(0)
            self.opacity_slider.setValue(45)
            self.show_labels_checkbox.setChecked(True)
            self.show_masks_checkbox.setChecked(True)
            self.show_track_ids_checkbox.setChecked(True)
            self.downscale_inference_checkbox.setChecked(False)
            self.inference_scale_spin.setValue(1.0)
            self.run_scope_combo.setCurrentIndex(max(self.run_scope_combo.findData("current"), 0))
            self.confidence_spin.setValue(0.25)
            self.preview_canvas.clear()
            self.preview_canvas.set_prompt_overlays([], [])
            self.preview_canvas.set_prompt_mask_preview(None)
            self.preview_canvas.set_manual_mask_preview(None)
            self._manual_mask_clipboard = None
            self._sequence_run_context = None
            self._reset_result_view()
            self._configure_playback()
            self._set_result_summary("No inference results yet.")
            self._clear_log("Project reset")
            self._set_project_cache_mode(False)
            self.cache_dir_edit.setText(str(self.cache_store.root))
        finally:
            self._suspend_dirty_tracking = False
        self._mark_project_clean()

    def _save_project(self) -> bool:
        if self._busy_for_project_action():
            return False
        if not self.state.project_path:
            return self._save_project_as()
        project_path = Path(self.state.project_path)
        project_cache = self._project_cache_root(project_path)
        if str(project_cache) != str(self.cache_store.root):
            target_store = CacheStore.create(project_cache)
            self._clone_cache_assets_to_store(target_store)
            self.cache_store = target_store
            self.backend = None
            self._backend_signature = None
            self.cache_dir_edit.setText(str(self.cache_store.root))
        payload = self._build_project_payload(project_path)
        save_project_document(project_path, payload)
        self.state.project_cache_dir = str(project_cache)
        self.state.project_name = self._project_name_from_path(project_path)
        self.state.cache_dir = str(self.cache_store.root)
        self._mark_project_clean()
        self._set_project_cache_mode(True)
        self._append_log(f"Saved project: {project_path}")
        return True

    def _save_project_as(self) -> bool:
        if self._busy_for_project_action():
            return False
        suggested = self.state.project_path or str(Path(self.state.source_path).with_name(f"{Path(self.state.source_path).stem}{PROJECT_SUFFIX}") if self.state.source_path else Path.cwd() / f"session{PROJECT_SUFFIX}")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Project As", suggested, "SAM3 Project (*.sam3proj.json)")
        if not path:
            return False
        project_path = self._normalize_project_save_path(path)
        project_cache = self._project_cache_root(project_path)
        target_store = CacheStore.create(project_cache)
        self._clone_cache_assets_to_store(target_store)
        self.cache_store = target_store
        self.backend = None
        self._backend_signature = None
        self.state.project_path = str(project_path)
        self.state.project_cache_dir = str(project_cache)
        self.state.project_name = self._project_name_from_path(project_path)
        self.state.cache_dir = str(self.cache_store.root)
        self.cache_dir_edit.setText(str(self.cache_store.root))
        save_project_document(project_path, self._build_project_payload(project_path))
        self._mark_project_clean()
        self._set_project_cache_mode(True)
        self._append_log(f"Saved project: {project_path}")
        return True

    def _open_project(self) -> None:
        if self._busy_for_project_action():
            return
        if not self._confirm_discard_unsaved_changes():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Project", str(Path.cwd()), "SAM3 Project (*.sam3proj.json)")
        if not path:
            return
        try:
            project_path = Path(path)
            payload = load_project_document(project_path)
            project_cache = self._project_cache_root(project_path)
            self.cache_store = CacheStore.create(project_cache)
            self.backend = None
            self._backend_signature = None
            self.cache_dir_edit.setText(str(self.cache_store.root))
            self._restore_project_payload(payload, project_path)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Project Error", traceback.format_exc())

    def _new_project(self) -> None:
        if self._busy_for_project_action():
            return
        if not self._confirm_discard_unsaved_changes():
            return
        self.cache_store = CacheStore.create(self.default_cache_root)
        self._reset_session()
        self._append_log("Started new project session")

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
        self._mark_project_dirty()
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
        self.state.suppressed_objects_by_key.clear()
        self.state.suppressed_track_ids_by_source.clear()
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
        self._invalidate_preview_caches()
        self.state.results = None
        self.state.suppressed_objects_by_key.clear()
        self.state.suppressed_track_ids_by_source.clear()
        self.state.view_filters_by_frame.clear()
        self.state.view_filters = ViewFilterState()
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
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
        self._mark_project_dirty()

    def _copy_manual_mask_to_clipboard(self) -> None:
        if self.state.manual_mask_input is None:
            QtWidgets.QMessageBox.warning(self, "No Manual Mask", "Create a manual mask first.")
            return
        mask = np.asarray(self.state.manual_mask_input, dtype=np.bool_).copy()
        self._manual_mask_clipboard = mask
        uint8_mask = mask.astype(np.uint8, copy=False) * 255
        height, width = uint8_mask.shape
        qimage = QtGui.QImage(uint8_mask.data, width, height, width, QtGui.QImage.Format.Format_Grayscale8).copy()
        QtWidgets.QApplication.clipboard().setImage(qimage)
        self._append_log("Copied manualMask id 0 to clipboard")
        self.statusBar().showMessage("Copied manualMask id 0 to clipboard")

    def _paste_manual_mask_to_current_frame(self) -> None:
        mask = None
        if self._manual_mask_clipboard is not None:
            mask = np.asarray(self._manual_mask_clipboard, dtype=np.bool_).copy()
        else:
            qimage = QtWidgets.QApplication.clipboard().image()
            if not qimage.isNull():
                gray = qimage.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
                width = gray.width()
                height = gray.height()
                bits = gray.bits()
                array = np.frombuffer(bits, dtype=np.uint8, count=gray.bytesPerLine() * height).reshape((height, gray.bytesPerLine()))
                mask = array[:, :width] > 0
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
        mask = np.asarray(self.state.manual_mask_input, dtype=np.bool_).copy()
        total = len(targets)
        for index, key in enumerate(targets, start=1):
            if key is not None:
                cached_path, _cached_mask = self._cache_manual_mask(key, mask)
                self.state.manual_masks_by_key[key] = cached_path
                self._append_log(f"[{index}/{total}] Copied manualMask id 0 to {self._format_source_key_label(key)}")
        self._refresh_view_filters()
        self._refresh_preview()
        self._append_log(f"Copied manualMask id 0 to {len(targets)} frame(s)")
        self._mark_project_dirty()

    def _copy_manual_mask_to_range(self) -> None:
        if self.state.manual_mask_input is None:
            QtWidgets.QMessageBox.warning(self, "No Manual Mask", "Create a manual mask first.")
            return
        if self.state.source_kind not in {"directory", "video"}:
            QtWidgets.QMessageBox.warning(self, "Unsupported Source", "Range copy requires a folder or video source.")
            return

        prev_count = int(self.copy_manual_prev_spin.value())
        next_count = int(self.copy_manual_next_spin.value())
        if prev_count <= 0 and next_count <= 0:
            QtWidgets.QMessageBox.warning(self, "Invalid Range", "Set Prev and/or Next to a value greater than 0.")
            return

        current_index = self._current_source_index()
        if self.state.source_kind == "directory":
            total_count = len(self.state.source_items)
        else:
            total_count = int(self.state.source_frame_count or 0)
        if total_count <= 0:
            return

        start_index = max(0, current_index - prev_count)
        end_index = min(total_count - 1, current_index + next_count)
        target_indices = [index for index in range(start_index, end_index + 1) if index != current_index]
        if not target_indices:
            self._append_log("No neighboring frames in selected range; nothing copied.")
            return

        mask = np.asarray(self.state.manual_mask_input, dtype=np.bool_).copy()
        total = len(target_indices)
        for item_index, frame_index in enumerate(target_indices, start=1):
            key = self._current_source_key(frame_index)
            if key is None:
                continue
            cached_path, _cached_mask = self._cache_manual_mask(key, mask)
            self.state.manual_masks_by_key[key] = cached_path
            self._append_log(f"[{item_index}/{total}] Copied manualMask id 0 to {self._format_source_key_label(key)}")

        self._refresh_view_filters()
        self._refresh_preview()
        self._append_log(
            f"Copied manualMask id 0 to {len(target_indices)} frame(s) around current "
            f"({self._playback_label_prefix()} {current_index + 1})"
        )
        self._mark_project_dirty()

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
        self._mark_project_dirty()
        self._mark_project_dirty()

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
        self._mark_project_dirty()
        self._mark_project_dirty()

    def _browse_export_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Export Directory", self.export_dir_edit.text() or str(Path.cwd()))
        if path:
            self.state.export_dir = path
            self.export_dir_edit.setText(path)
            self._mark_project_dirty()
            self._mark_project_dirty()

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
        self._mark_project_dirty()
        self._mark_project_dirty()

    def _clear_all(self) -> None:
        self._invalidate_preview_caches()
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
        self._mark_project_dirty()
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
                return backend.predict_image(
                    self.state.source_path,
                    inference_scale=inference_scale,
                    cancel_callback=cancel_callback,
                    **prompt_kwargs,
                )
            if self.state.source_kind == "directory":
                source = self.state.source_items[min(current_index, len(self.state.source_items) - 1)]
                return backend.predict_image(
                    source,
                    inference_scale=inference_scale,
                    cancel_callback=cancel_callback,
                    **prompt_kwargs,
                )
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
        self.current_task.signals.cancelled.connect(self._handle_cancelled)
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
        self._run_cache_epoch += 1
        export_dir = self.export_dir_edit.text().strip() or None
        auto_export_dir = export_dir if self.auto_export_masks_checkbox.isChecked() else None
        if auto_export_dir and self._export_would_overwrite_source_imagery(auto_export_dir):
            self._warn_source_overwrite_export()
            return
        preserve_source_filenames = self.preserve_source_filename_checkbox.isChecked()
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
                preserve_source_filenames=preserve_source_filenames,
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
                    preserve_source_filenames=preserve_source_filenames,
                    cancel_callback=cancel_callback,
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
                        preserve_source_filenames=preserve_source_filenames,
                        cancel_callback=cancel_callback,
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
                    preserve_source_filenames=preserve_source_filenames,
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
                    preserve_source_filenames=preserve_source_filenames,
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
                    preserve_source_filenames=preserve_source_filenames,
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
                preserve_source_filenames=preserve_source_filenames,
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
        self.current_task.signals.cancelled.connect(self._handle_cancelled)
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
        preserve_source_filenames: bool,
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
            "preserve_source_filenames": preserve_source_filenames,
            "error": None,
            "cancelled": False,
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
        if context.get("cancelled"):
            self._handle_cancelled("Inference cancelled.")
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
                    preserve_source_filenames=context["preserve_source_filenames"],
                    cancel_callback=cancel_callback,
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
                    preserve_source_filenames=context["preserve_source_filenames"],
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
                preserve_source_filenames=context["preserve_source_filenames"],
                cancel_callback=cancel_callback,
            )
            return results[0] if results else None

        self.current_task = BackendTask(job)
        self.current_task.signals.result.connect(lambda result, i=index, lab=str(label): self._handle_sequence_item_result(i, total, lab, result))
        self.current_task.signals.error.connect(self._handle_sequence_item_error)
        self.current_task.signals.cancelled.connect(self._handle_cancelled)
        self.thread_pool.start(self.current_task)

    def _handle_sequence_item_result(self, index: int, total: int, label: str, result) -> None:
        context = self._sequence_run_context
        if context is None:
            return
        self.current_task = None
        if context.get("cancelled"):
            self._handle_cancelled("Inference cancelled.")
            return
        self._handle_batch_item_result(index, total, result, label)
        if result is not None and context.get("propagate_sequence_mask"):
            next_mask = result.prompt_mask if result.prompt_mask is not None else self.backend._first_object_mask(result)
            if next_mask is not None:
                context["sequence_mask"] = np.asarray(next_mask) > 0
        if isinstance(self.state.results, list):
            context["results"] = list(self.state.results)
        context["next_index"] = index + 1
        timing_suffix = self._result_processing_time_label(result)
        processed_message = f"Processed {self._format_source_key_label(label)}"
        if timing_suffix:
            processed_message += timing_suffix
        self._update_progress(index + 1, total, processed_message)
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
        if self._sequence_run_context is not None:
            self._sequence_run_context["cancelled"] = True
        if self.current_task is not None:
            self.current_task.cancel()
            self.statusBar().showMessage("Cancellation requested...")
            self._append_log("Cancellation requested")

    def _handle_cancelled(self, message: str) -> None:
        self.current_task = None
        self.current_task_mode = None
        self._streaming_batch_mode = False
        self._streaming_batch_total = 0
        self.cancel_button.setEnabled(False)
        self.run_button.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Cancelled")
        self.statusBar().showMessage(message)
        if self._sequence_run_context is not None:
            context = self._sequence_run_context
            if isinstance(context.get("results"), list):
                self.state.results = context["results"]
            self._sequence_run_context = None
            self._configure_playback()
            self._refresh_view_filters()
            self._update_result_panel()
            self._refresh_preview()
        self._append_log(message)

    @staticmethod
    def _clone_segmentation_object(
        obj: SegmentationObject,
        object_index: int,
        *,
        copy_mask: bool = False,
    ) -> SegmentationObject:
        mask = obj.mask
        if copy_mask:
            array = np.asarray(obj.mask)
            mask = array.copy() if array.dtype == np.bool_ else (array > 0).copy()
        return SegmentationObject(
            mask=mask,
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
            objects.append(self._clone_segmentation_object(obj, len(objects) + 1, copy_mask=False))
        for obj in incoming.objects:
            objects.append(self._clone_segmentation_object(obj, len(objects) + 1, copy_mask=False))
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
            image=image,
            prompt_mask=prompt_mask,
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
        if not self.append_inference_checkbox.isChecked():
            self.state.suppressed_objects_by_key.clear()
            self.state.suppressed_track_ids_by_source.clear()
        if self._streaming_batch_mode and isinstance(result, list):
            if not self.append_inference_checkbox.isChecked():
                self.state.results = self._cache_results(result)
        else:
            applied_result = self._apply_inference_result(result)
            if self.append_inference_checkbox.isChecked() and isinstance(applied_result, list):
                # Keep existing cached frame results intact and avoid re-caching
                # every frame when append mode updates only one frame.
                self.state.results = applied_result
            else:
                self.state.results = self._cache_results(applied_result)
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
        self._mark_project_dirty()

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
        if not self.append_inference_checkbox.isChecked():
            self.state.suppressed_objects_by_key.clear()
            self.state.suppressed_track_ids_by_source.clear()
        if not isinstance(self.state.results, list) or len(self.state.results) != total:
            self.state.results = [None] * total
        existing = self.state.results[index] if 0 <= index < len(self.state.results) else None
        cache_key = f"run{self._run_cache_epoch}_{index}_{self._cache_token(label)}"
        cached_incoming = self._cache_result(result, cache_key)
        if self.append_inference_checkbox.isChecked() and existing is not None:
            self.state.results[index] = self._merge_prediction_result(existing, cached_incoming)
        else:
            self.state.results[index] = cached_incoming
        self.seek_slider.setValue(index)
        self.state.current_frame_index = index
        self._display_current_result()
        self._append_log(f"[{index + 1}/{total}] Loaded result for {self._format_source_key_label(label)}")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        if index + 1 < total:
            self._advance_batch_preview(index + 1)
        self._mark_project_dirty()

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
        self._mark_project_dirty()

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
        visible_objects = self._result_objects(result, apply_filters=True, include_suppressed=False)
        hidden_count = len(result.objects) - len(self._result_objects(result, apply_filters=False, include_suppressed=False))
        lines = []
        if isinstance(self.state.results, list) and self.state.source_kind == "directory":
            lines.append(f"Batch Images: {len(self.state.results)}")
        lines.extend(
            [
                f"Source: {result.source}",
                f"Mode: {result.mode}",
                f"Image Size: {result.image_size}",
                f"Inference Size: {result.inference_image_size}",
                f"Objects: {len(visible_objects)} visible / {len(result.objects)} total",
                f"Deleted: {hidden_count}",
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
        self.frame_jump_edit.setEnabled(True)
        self.frame_jump_edit.setText(str(current_index + 1))

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
            current_one_based = self.seek_slider.value() + 1
            self.frame_label.setText(f"{self._playback_label_prefix()} {current_one_based}/{count}")
            self.frame_jump_edit.setText(str(current_one_based))
        self._refresh_view_filters()
        self._update_result_panel()
        self._refresh_preview()
        self._mark_project_dirty()

    def _current_mask_preview(self):
        if self.state.mask_input is None or not self._mask_visible_for_filters():
            return None
        return preview_mask(self.state.mask_input)

    def _current_manual_mask_preview(self):
        if self.state.manual_mask_input is None or not self._manual_mask_visible_for_filters():
            return None
        return preview_mask(self.state.manual_mask_input)

    def _invalidate_preview_caches(self) -> None:
        self._preview_frame_cache_key = None
        self._preview_frame_cache_image = None
        self._preview_overlay_cache_key = None
        self._preview_overlay_cache_image = None

    def _preview_frame_cache_key_for_result(self, result: PredictionResult | None) -> object | None:
        if result is not None:
            if result.image is not None and (not result.source or not Path(str(result.source)).exists()):
                return ("inline", result.source, result.frame_index, id(result.image))
            if result.mode == "video" and self.state.source_path:
                return ("video", str(self.state.source_path), int(result.frame_index or 0))
            if result.source:
                return ("image", str(result.source))
        if self.state.source_kind == "video" and self.state.source_path:
            return ("video", str(self.state.source_path), self._current_source_index())
        if self.state.source_kind == "directory" and self.state.source_items:
            current_index = min(self._current_source_index(), len(self.state.source_items) - 1)
            return ("image", str(self.state.source_items[current_index]))
        if self.state.source_path:
            return ("image", str(self.state.source_path))
        return None

    def _resolve_preview_frame(self, result: PredictionResult | None) -> np.ndarray | None:
        frame_key = self._preview_frame_cache_key_for_result(result)
        if frame_key is not None and frame_key == self._preview_frame_cache_key and self._preview_frame_cache_image is not None:
            return self._preview_frame_cache_image

        frame = None
        if result is not None:
            if result.image is not None:
                frame = result.image
            elif self.state.source_kind == "video" and self.state.source_path:
                frame = read_video_frame(self.state.source_path, result.frame_index or 0)
            elif result.source:
                frame = to_bgr_image(result.source)
        elif self.state.source_kind == "video" and self.state.source_path:
            frame = read_video_frame(self.state.source_path, self._current_source_index())
        elif self.state.source_kind == "directory" and self.state.source_items:
            current_index = min(self._current_source_index(), len(self.state.source_items) - 1)
            frame = to_bgr_image(self.state.source_items[current_index])
        elif self.state.source_path:
            frame = to_bgr_image(self.state.source_path)

        self._preview_frame_cache_key = frame_key
        self._preview_frame_cache_image = None if frame is None else np.asarray(frame)
        return self._preview_frame_cache_image

    def _preview_overlay_cache_signature(
        self,
        result: PredictionResult,
        visible_objects: list[SegmentationObject],
        *,
        show_masks: bool,
    ) -> tuple[object, ...]:
        result_ref = self._result_cache_ref(result) or ("live", id(result))
        object_keys = tuple(self._object_instance_key(result, obj) for obj in visible_objects)
        return (
            self._preview_frame_cache_key_for_result(result),
            result_ref,
            id(result),
            object_keys,
            round(self.opacity_slider.value() / 100.0, 4),
            self.show_labels_checkbox.isChecked(),
            bool(show_masks),
            self.show_track_ids_checkbox.isChecked(),
        )

    def _refresh_preview(self) -> None:
        result = self._current_result()
        if result is not None and not self._result_matches_current_view(result):
            result = None

        frame = self._resolve_preview_frame(result)
        if frame is None:
            return

        if result is not None:
            visible_objects = self._result_objects(result, apply_filters=True, include_suppressed=False)
            show_masks = self.show_masks_checkbox.isChecked() or self._has_active_view_filters()
            overlay_key = self._preview_overlay_cache_signature(result, visible_objects, show_masks=show_masks)
            if overlay_key == self._preview_overlay_cache_key and self._preview_overlay_cache_image is not None:
                overlay = self._preview_overlay_cache_image
            else:
                overlay = render_overlay(
                    frame,
                    result,
                    opacity=self.opacity_slider.value() / 100.0,
                    show_labels=self.show_labels_checkbox.isChecked(),
                    show_masks=show_masks,
                    show_track_ids=self.show_track_ids_checkbox.isChecked(),
                    objects=visible_objects,
                )
                self._preview_overlay_cache_key = overlay_key
                self._preview_overlay_cache_image = overlay
            self.preview_canvas.set_image(overlay)
        else:
            self._preview_overlay_cache_key = None
            self._preview_overlay_cache_image = None
            self.preview_canvas.set_image(frame)
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

    def _jump_to_frame(self) -> None:
        count = self._playback_count()
        if count <= 0:
            self.frame_jump_edit.clear()
            return
        text = self.frame_jump_edit.text().strip()
        if not text:
            self.frame_jump_edit.setText(str(self.seek_slider.value() + 1))
            return
        try:
            one_based = int(text)
        except ValueError:
            self.frame_jump_edit.setText(str(self.seek_slider.value() + 1))
            return
        one_based = max(1, min(one_based, count))
        self.seek_slider.setValue(one_based - 1)

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
        self.frame_jump_edit.setEnabled(False)
        self.frame_jump_edit.clear()
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
        apply_view_filters = True
        export_results = self._results_for_export(apply_view_filters=apply_view_filters)
        if isinstance(export_results, list):
            export_results = [item for item in export_results if item is not None]
        if self._export_would_overwrite_source_imagery(export_dir, export_results):
            self._warn_source_overwrite_export()
            return
        export_manual_masks = self._manual_masks_for_export(apply_view_filters=apply_view_filters)

        def job(progress_callback=None, cancel_callback=None):
            if cancel_callback is not None and cancel_callback():
                return None
            return self.backend.save_results(
                export_results,
                output_dir=None,
                mask_dir=export_dir,
                save_overlay=False,
                save_json=False,
                merged_mask_only=self.merge_masks_only_checkbox.isChecked(),
                invert_mask=self.invert_mask_export_checkbox.isChecked(),
                manual_masks_by_key=export_manual_masks,
                dilation_pixels=int(self.export_dilation_spin.value()),
                preserve_source_filenames=self.preserve_source_filename_checkbox.isChecked(),
                progress_callback=progress_callback,
            )

        self.current_task = BackendTask(job)
        self.current_task_mode = "export"
        self.current_task.signals.result.connect(lambda result: self._handle_export_result(result, export_dir))
        self.current_task.signals.error.connect(self._handle_export_error)
        self.current_task.signals.cancelled.connect(self._handle_cancelled)
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
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

























































