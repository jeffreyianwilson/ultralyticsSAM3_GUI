import pathlib

import cv2
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtWidgets

import sam3_ultralytics.gui_app as gui_app_module
from sam3_ultralytics.gui_app import SAM3MainWindow
from sam3_ultralytics.gui_workers import BackendTask
from sam3_ultralytics.schemas import PredictionResult, SegmentationObject


class DummyBackend:
    def __init__(self):
        self.called = False
        self.last_kwargs = {}

    def save_results(self, *args, **kwargs):
        self.called = True
        self.last_kwargs = kwargs
        progress_callback = kwargs.get("progress_callback")
        results = args[0] if args else None
        total = len(results) if isinstance(results, list) else 1
        if progress_callback is not None:
            if isinstance(results, list):
                for index, result in enumerate(results, start=1):
                    progress_callback(index, total, f"Exported {pathlib.Path(result.source or f'item_{index}').stem}")
            else:
                progress_callback(1, 1, f"Exported {pathlib.Path(getattr(results, 'source', 'mask')).stem}")
        return {"masks": [str(pathlib.Path(kwargs["mask_dir"]) / "mask.png")]}


class DummyTrackingBackend(DummyBackend):
    def __init__(self):
        super().__init__()
        self.predict_image_called = False
        self.predict_image_sequence_called = False
        self.track_image_sequence_called = False
        self.track_video_called = False

    def predict_image(self, source, **kwargs):
        self.predict_image_called = True
        self.last_kwargs = kwargs
        return _result()

    def predict_image_sequence(self, sources, **kwargs):
        self.predict_image_sequence_called = True
        self.last_kwargs = kwargs
        return [_result() for _ in sources]

    def track_image_sequence(self, sources, **kwargs):
        self.track_image_sequence_called = True
        self.last_kwargs = kwargs
        return [_result() for _ in sources]

    def track_video(self, source, **kwargs):
        self.track_video_called = True
        self.last_kwargs = kwargs
        return [_result(), _result()]


def _result() -> PredictionResult:
    return PredictionResult(
        source="sample.png",
        frame_index=None,
        mode="image",
        image_size=(8, 8),
        objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=None, object_index=1)],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )


def _patch_mask_prompt(monkeypatch, value: str = "person", target_id: int = 1):
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: (value, True))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", lambda *args, **kwargs: (target_id, True))


def test_app_window_smoke(qapp):
    window = SAM3MainWindow()
    assert window.windowTitle() == "sam3_ultralytics"


def test_widget_initialization(qapp):
    window = SAM3MainWindow()
    assert window.preview_canvas is not None
    assert window.model_combo is not None
    assert window.cache_dir_edit is not None
    assert window.clear_cache_button.text() == "Clear Cache"
    assert window.run_button.text() == "Run"
    assert window.clear_masks_button.text() == "Clear Masks"
    assert window.open_directory_button.text() == "Open Folder"
    assert window.mask_button.text() == "Load Mask"
    assert window.manual_mask_tool_button.text() == "Manual Mask Tool"
    assert window.clear_all_button.text() == "Clear All"
    assert window.step_back_button.text() == "Prev"
    assert window.step_forward_button.text() == "Next"
    assert window.mask_class_edit.placeholderText() == "Optional target class"
    assert window.run_scope_combo.itemText(1) == "Entire Folder / Video"
    assert window.confidence_spin.value() == 0.25
    assert window.append_inference_checkbox.isChecked() is False
    assert window.export_dilation_spin.value() == 0
    assert window.filter_class_combo.currentText() == "All classes"
    assert window.filter_id_combo.currentText() == "All IDs"
    assert window.progress_bar is not None


def test_prompt_state_updates_use_rollout_target(qapp, monkeypatch):
    window = SAM3MainWindow()
    window.mask_class_edit.setText("person")
    window.mask_id_edit.setText("7")
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("point prompts should not prompt")))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("point prompts should not prompt")))
    window._add_point(1, 2, 1)
    window._add_box(1, 2, 6, 7, 1)
    assert len(window.state.points) == 1
    assert len(window.state.boxes) == 1
    assert window.state.mask_id == 7
    assert window.state.mask_class == "person"
    assert window.prompt_summary_label.text() == "1 points, 1 boxes, id=7"


def test_interaction_prompts_do_not_auto_schedule_preview(qapp, monkeypatch):
    window = SAM3MainWindow()
    window.state.source_path = "sample.png"
    window.mask_id_edit.setText("3")
    started = []
    monkeypatch.setattr(window.preview_timer, "start", lambda: started.append(True))
    window._add_point(1, 2, 1)
    window._add_box(1, 2, 6, 7, 1)
    assert started == []
    assert "Press Run to infer." in window.statusBar().currentMessage()


def test_mask_state_updates(qapp, tmp_path):
    window = SAM3MainWindow()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.full((8, 8), 255, dtype=np.uint8))
    window._load_mask_from_path(str(mask_path))
    assert window.state.mask_path is not None
    assert window.state.mask_path.endswith(".npy")
    assert pathlib.Path(window.state.mask_path).exists()
    assert window.state.mask_input is not None
    assert window.mask_label.text() == str(mask_path)


def test_directory_source_updates_state_and_playback(qapp, tmp_path):
    window = SAM3MainWindow()
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    assert window.state.source_kind == "directory"
    assert len(window.state.source_items) == 2
    assert window.play_button.isEnabled() is True
    assert window.step_back_button.isEnabled() is True
    assert window.step_forward_button.isEnabled() is True
    assert window.seek_slider.isEnabled() is True


def test_directory_step_buttons_cycle_images(qapp, tmp_path):
    window = SAM3MainWindow()
    for index in range(3):
        image = np.full((8, 8, 3), index, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), image)
    window._load_directory_path(str(tmp_path))
    window._step_sequence(1)
    assert window.seek_slider.value() == 1
    assert window.frame_label.text() == "Image 2/3"
    window._step_sequence(-1)
    assert window.seek_slider.value() == 0


def test_per_view_masks_do_not_persist_between_directory_images(qapp, tmp_path, monkeypatch):
    window = SAM3MainWindow()
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    window._step_sequence(1)
    assert window.state.manual_mask_input is None
    assert window.manual_mask_label.text() == "No manual mask"
    window._step_sequence(-1)
    assert window.state.manual_mask_input is not None
    assert "manualMask id=0" in window.manual_mask_label.text()


def test_manual_mask_updates_state_with_fixed_metadata(qapp, tmp_path):
    window = SAM3MainWindow()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    window.preview_canvas.set_image(np.zeros((32, 32, 3), dtype=np.uint8))
    window._set_tool("manual_mask")
    window._set_manual_mask(np.ones((32, 32), dtype=np.float32))
    assert window.state.manual_mask_input is not None
    key = window._current_source_key()
    assert key is not None
    assert pathlib.Path(window.state.manual_masks_by_key[key]).exists()
    assert "manualMask id=0" in window.manual_mask_label.text()
    prompt_kwargs = window._prompt_kwargs()
    assert prompt_kwargs["mask_input"] is None


def test_clear_masks_only_clears_all_stored_masks(qapp, tmp_path):
    window = SAM3MainWindow()
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    window._step_sequence(1)
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    window._clear_masks_only()
    assert window.state.mask_inputs_by_key == {}
    assert window.state.manual_masks_by_key == {}
    assert window.state.manual_mask_input is None


def test_clear_masks_only_resets_displayed_results(qapp, tmp_path):
    window = SAM3MainWindow()
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    cv2.imwrite(str(first), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(second), np.zeros((8, 8, 3), dtype=np.uint8))
    window.state.results = [_result(), _result()]
    window.state.source_kind = "directory"
    window.state.source_items = [str(first), str(second)]
    window._configure_playback()
    window._clear_masks_only()
    assert window.state.results is None
    assert "Masks cleared" in window.result_summary_label.text()

def test_clear_all_resets_prompt_state(qapp):
    window = SAM3MainWindow()
    window.state.source_path = "sample.png"
    window.text_prompt_edit.setText("person")
    window._add_point(1, 2, 1)
    window._add_box(1, 2, 3, 4, 1)
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    window.state.results = _result()
    window._clear_all()
    assert window.text_prompt_edit.text() == ""
    assert window.state.mask_input is None
    assert window.state.manual_mask_input is None
    assert window.state.results is None
    assert window.state.points == []
    assert window.state.boxes == []


def test_brush_size_updates_canvas(qapp):
    window = SAM3MainWindow()
    window._set_brush_size(23)
    assert window.brush_label.text() == "23 px"
    assert window.preview_canvas._brush_radius == 23


def test_manual_mask_does_not_affect_prompt_kwargs(qapp, tmp_path):
    window = SAM3MainWindow()
    window.text_prompt_edit.setText("bus")
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    prompt_kwargs = window._prompt_kwargs()
    assert prompt_kwargs["text_prompt"] == "bus"
    assert prompt_kwargs["mask_input"] is None


def test_manual_mask_uses_fixed_id_and_class(qapp, tmp_path):
    window = SAM3MainWindow()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    assert window.state.manual_mask_input is not None
    key = window._current_source_key()
    assert key is not None
    assert pathlib.Path(window.state.manual_masks_by_key[key]).exists()
    assert "manualMask id=0" in window.manual_mask_label.text()


def test_subsequent_manual_mask_edits_update_without_prompt(qapp, tmp_path, monkeypatch):
    window = SAM3MainWindow()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    first_mask = np.zeros((8, 8), dtype=np.float32)
    first_mask[:4, :4] = 1.0
    window._set_manual_mask(first_mask)
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manual masks should not prompt")))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getInt", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manual masks should not prompt")))

    second_mask = first_mask.copy()
    second_mask[4:, 4:] = 1.0
    window._set_manual_mask(second_mask)
    assert np.array_equal(window.state.manual_mask_input, second_mask)


def test_view_filters_populate_and_hide_nonmatching_prompt_mask(qapp):
    window = SAM3MainWindow()
    window.state.mask_input = np.ones((8, 8), dtype=np.float32)
    window.state.mask_id = 1
    window.state.mask_class = "person"
    window.state.manual_mask_input = np.ones((8, 8), dtype=np.float32)
    result = PredictionResult(
        source="sample.png",
        frame_index=None,
        mode="image",
        image_size=(8, 8),
        objects=[
            SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=1, object_index=1),
            SegmentationObject(mask=np.tri(8, 8, dtype=bool), box=(1, 1, 6, 6), score=0.8, label="car", track_id=2, object_index=2),
        ],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    window._handle_result(result)
    assert window.filter_class_combo.findData("person") >= 0
    assert window.filter_class_combo.findData("car") >= 0
    assert window.filter_class_combo.findData("manualMask") >= 0
    assert window.filter_id_combo.findData(1) >= 0
    assert window.filter_id_combo.findData(2) >= 0
    assert window.filter_id_combo.findData(0) >= 0
    window.filter_class_combo.setCurrentIndex(window.filter_class_combo.findData("car"))
    assert window._current_mask_preview() is None
    window.filter_class_combo.setCurrentIndex(window.filter_class_combo.findData("person"))
    window.filter_id_combo.setCurrentIndex(window.filter_id_combo.findData(1))
    assert window._current_mask_preview() is not None


def test_interaction_preview_runs_current_view_inference(qapp, tmp_path, monkeypatch):
    window = SAM3MainWindow()
    backend = DummyTrackingBackend()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    window.mask_class_edit.setText("jeff")
    window.mask_id_edit.setText("1")
    window._add_box(1, 2, 6, 7, 1)
    monkeypatch.setattr(window, "_create_backend", lambda: backend)
    monkeypatch.setattr(window.thread_pool, "start", lambda task: task.run())
    window._run_interaction_preview()
    assert backend.predict_image_called is True
    assert backend.last_kwargs["boxes"] == [(1, 2, 6, 7, 1)]


def test_first_frame_mask_run_uses_batch_sequence_inference_for_directories(qapp, tmp_path, monkeypatch):
    window = SAM3MainWindow()
    window.text_prompt_edit.setText("jeff")
    window.mask_id_edit.setText("1")
    backend = DummyTrackingBackend()
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(mask_path), np.full((8, 8), 255, dtype=np.uint8))
    window._load_mask_from_path(str(mask_path))
    window.run_scope_combo.setCurrentIndex(1)
    monkeypatch.setattr(window, "_create_backend", lambda: backend)
    monkeypatch.setattr(window.thread_pool, "start", lambda task: task.run())
    window._run_inference()
    assert backend.predict_image_called is True
    assert backend.track_image_sequence_called is False
    assert backend.last_kwargs["mask_id"] == 1
    assert backend.last_kwargs["mask_label"] == "jeff"
    assert window.run_scope_combo.currentData() == "all"


def test_manual_mask_clipboard_copy_paste(qapp, tmp_path):
    window = SAM3MainWindow()
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    mask = np.tri(8, 8, dtype=np.float32)
    window._set_manual_mask(mask)
    window._copy_manual_mask_to_clipboard()
    window._clear_current_manual_mask()
    window._paste_manual_mask_to_current_frame()
    assert window.state.manual_mask_input is not None
    assert np.array_equal(window.state.manual_mask_input, mask)


def test_manual_mask_copy_to_all_frames_logs_progress(qapp, tmp_path):
    window = SAM3MainWindow()
    for index in range(3):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    window._copy_manual_mask_to_all_frames()
    log_text = window.result_panel.toPlainText()
    assert "[1/3] Copied manualMask id 0 to image_0.png" in log_text
    assert "[3/3] Copied manualMask id 0 to image_2.png" in log_text


def test_append_inference_merges_current_results(qapp):
    window = SAM3MainWindow()
    first = _result()
    second = PredictionResult(
        source="sample.png",
        frame_index=None,
        mode="image",
        image_size=(8, 8),
        objects=[SegmentationObject(mask=np.tri(8, 8, dtype=bool), box=(1, 1, 6, 6), score=0.8, label="car", track_id=2, object_index=1)],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    window._handle_result(first)
    window.append_inference_checkbox.setChecked(True)
    window._handle_result(second)
    assert window.state.results is not None
    assert len(window.state.results.objects) == 2
    assert hasattr(window.state.results.objects[0].mask, "path")


def test_clear_cache_dir_clears_cached_masks_and_results(qapp, tmp_path):
    window = SAM3MainWindow()
    window._switch_cache_dir(str(tmp_path / "cache"))
    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    window._set_manual_mask(np.ones((8, 8), dtype=np.float32))
    window._handle_result(_result())
    assert any(pathlib.Path(window.state.cache_dir).rglob("*.npy"))
    window._clear_cache_dir()
    assert not any(pathlib.Path(window.state.cache_dir).rglob("*.npy"))
    assert window.state.results is None
    assert window.state.manual_mask_input is None

def test_progress_updates_log_and_bar(qapp):
    window = SAM3MainWindow()
    window._handle_progress(2, 5, "Processed image_002.png")
    assert window.progress_bar.value() == 40
    assert "Processed image_002.png" in window.result_panel.toPlainText()


def test_batch_item_started_advances_to_current_frame(qapp, tmp_path):
    window = SAM3MainWindow()
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    window.seek_slider.setValue(0)
    window._handle_batch_item_started(1, 2, "image_1.png")
    assert window.seek_slider.value() == 1


def test_batch_item_result_updates_current_before_advancing(qapp, monkeypatch, tmp_path):
    window = SAM3MainWindow()
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"image_{index}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_directory_path(str(tmp_path))
    window.state.results = [None, None]
    window._streaming_batch_mode = True
    window._streaming_batch_total = 2
    window.current_task_mode = "run"
    window.current_task = object()
    preview_indices = []
    monkeypatch.setattr(window, "_refresh_preview", lambda: preview_indices.append(window.seek_slider.value()))
    monkeypatch.setattr(QtWidgets.QApplication, "processEvents", lambda *args, **kwargs: None)
    window._handle_batch_item_result(0, 2, _result(), "image_0.png")
    assert preview_indices[0] == 0
    assert window.seek_slider.value() == 1


def test_mocked_worker_behavior(qapp):
    values = []
    task = BackendTask(lambda progress_callback=None, cancel_callback=None: "done")
    task.signals.result.connect(values.append)
    task.run()
    assert values == ["done"]


def test_export_action_with_temporary_files(qapp, tmp_path, monkeypatch):
    window = SAM3MainWindow()
    window.backend = DummyBackend()
    window.state.results = [_result(), _result()]
    window.state.source_kind = "directory"
    window.state.manual_masks_by_key = {"sample.png": np.ones((8, 8), dtype=np.float32)}
    window.export_dir_edit.setText(str(tmp_path))
    window.merge_masks_only_checkbox.setChecked(True)
    window.invert_mask_export_checkbox.setChecked(True)
    window.export_dilation_spin.setValue(5)
    monkeypatch.setattr(window.thread_pool, "start", lambda task: task.run())
    window._export_masks_only()
    assert window.backend.called is True
    assert window.backend.last_kwargs["merged_mask_only"] is True
    assert window.backend.last_kwargs["invert_mask"] is True
    assert window.backend.last_kwargs["manual_masks_by_key"] == window.state.manual_masks_by_key
    assert window.backend.last_kwargs["dilation_pixels"] == 5
    assert "Exported masks to" in window.result_panel.toPlainText()


def test_backend_is_reused_until_runtime_settings_change(qapp, tmp_path, monkeypatch):
    created = []

    class FakeBackend:
        def __init__(self, model_path, *, device, conf, yolo_config_dir):
            self.model_path = model_path
            self.device = device
            self.conf = conf
            self.yolo_config_dir = yolo_config_dir
            self.load_calls = 0
            created.append(self)

        def load(self):
            self.load_calls += 1
            return self

    monkeypatch.setattr(gui_app_module, "SAM3Ultralytics", FakeBackend)
    window = SAM3MainWindow()

    backend_a = window._create_backend()
    backend_b = window._create_backend()
    assert backend_a is backend_b
    assert len(created) == 1
    assert backend_a.load_calls == 1

    image_path = tmp_path / "image.png"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    window._load_image_path(str(image_path))
    backend_c = window._create_backend()
    assert backend_c is backend_a
    assert len(created) == 1

    window.confidence_spin.setValue(0.35)
    backend_d = window._create_backend()
    assert backend_d is not backend_a
    assert len(created) == 2
    assert backend_d.load_calls == 1






