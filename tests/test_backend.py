import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sam3_ultralytics.backend import SAM3Ultralytics
from sam3_ultralytics.exceptions import ExportError, UnsupportedPromptError
from sam3_ultralytics.export import image_mask_filename, save_results, video_mask_filename
from sam3_ultralytics.model_loading import ModelLoader
from sam3_ultralytics.schemas import PredictionResult, SegmentationObject


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.cls = torch.tensor(cls, dtype=torch.float32)
        self.id = None if ids is None else torch.tensor(ids, dtype=torch.int32)


class FakeMasks:
    def __init__(self, masks):
        self.data = torch.tensor(masks, dtype=torch.bool)


class FakeResult:
    def __init__(self, path: str, masks, boxes, names, image, speed, shape):
        self.path = path
        self.masks = FakeMasks(masks)
        self.boxes = FakeBoxes(**boxes)
        self.names = names
        self.orig_img = image
        self.orig_shape = shape
        self.speed = speed


class FakePredictor:
    def __init__(self, results):
        self.results = results
        self.calls = []
        self.args = type("Args", (), {"compile": None, "imgsz": 1024})()
        self.model = type(
            "Model",
            (),
            {"sam_prompt_encoder": type("PromptEncoder", (), {"mask_input_size": (296, 296)})()},
        )()
        self.stride = 14

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if kwargs.get("stream"):
            return iter(self.results)
        return list(self.results)


class FakePreparedMaskPredictor(FakePredictor):
    def __init__(self, results):
        super().__init__(results)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.prepared_prompts = None

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        point_tensor = None
        label_tensor = None
        if points is not None:
            point_tensor = torch.tensor(points, dtype=self.torch_dtype, device=self.device)
            if point_tensor.ndim == 2:
                point_tensor = point_tensor[:, None, :]
        if labels is not None:
            label_tensor = torch.tensor(labels, dtype=torch.int32, device=self.device)
            if label_tensor.ndim == 1:
                label_tensor = label_tensor[:, None]
        return point_tensor, label_tensor, masks

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.prepared_prompts = self._prepare_prompts(
            (1036, 1036),
            (800, 800),
            points=kwargs.get("points"),
            labels=kwargs.get("labels"),
            masks=kwargs.get("masks"),
        )
        return list(self.results)


class SequentialPredictor(FakePredictor):
    def __init__(self, results):
        super().__init__(results)
        self._cursor = 0

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if kwargs.get("stream"):
            return iter(self.results)
        index = min(self._cursor, len(self.results) - 1)
        self._cursor += 1
        return [self.results[index]]


class FakeLoader:
    def __init__(self, semantic_image=None, interactive_image=None, semantic_video=None, interactive_video=None):
        self.semantic_image = semantic_image
        self.interactive_image = interactive_image
        self.semantic_video = semantic_video
        self.interactive_video = interactive_video

    def load(self):
        return self

    def get_semantic_image_predictor(self):
        return self.semantic_image

    def get_interactive_image_predictor(self):
        return self.interactive_image

    def get_semantic_video_predictor(self):
        return self.semantic_video

    def get_interactive_video_predictor(self):
        return self.interactive_video


class FakeExemplarAdapter:
    def predict_image(self, *, model_loader, source, payload):
        mask = np.ones((8, 8), dtype=bool)
        return PredictionResult(
            source=str(source),
            frame_index=None,
            mode="image",
            image_size=(8, 8),
            objects=[SegmentationObject(mask=mask, box=(0, 0, 7, 7), score=0.9, label="example", track_id=None, object_index=1)],
            prompt_metadata={"adapter": True},
            timings={"backend_total_ms": 1.0},
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )

    def track_video(self, *, model_loader, source, payload, progress_callback=None, cancel_callback=None):
        return []


@pytest.fixture()
def sample_image() -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    return image


@pytest.fixture()
def image_result(sample_image) -> FakeResult:
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 2:6, 2:6] = True
    return FakeResult(
        path="sample.png",
        masks=masks,
        boxes={"xyxy": [[2, 2, 6, 6]], "conf": [0.91], "cls": [0]},
        names=["person"],
        image=sample_image,
        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0},
        shape=(8, 8),
    )


def _make_video(path: Path, frames: int = 2) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 8))
    for _ in range(frames):
        writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    writer.release()
    return path


def test_model_loading(tmp_path):
    model_path = tmp_path / "sam3.pt"
    model_path.write_bytes(b"model")
    loader = ModelLoader(model_path, device="cpu", yolo_config_dir=tmp_path)
    loader.load()
    assert loader.model_path == model_path
    assert loader.device == "cpu"


def test_text_prompt_semantic_predictor_retries_on_cpu_cuda_launch_failure(sample_image):
    result = FakeResult(
        path="sample.png",
        masks=np.pad(np.ones((1, 4, 4), dtype=bool), ((0, 0), (2, 2), (2, 2))),
        boxes={"xyxy": [[2, 2, 5, 5]], "conf": [0.95], "cls": [0]},
        names=["person"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )

    class FailingSemanticPredictor:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("CUDA error: unspecified launch failure")

    class RecordingSemanticPredictor(FakePredictor):
        def __init__(self, results):
            super().__init__(results)
            self.device_overrides = []

    class FallbackLoader:
        def __init__(self):
            self.device = "cuda:0"
            self.gpu_predictor = FailingSemanticPredictor()
            self.cpu_predictor = RecordingSemanticPredictor([result])

        def get_semantic_image_predictor(self, device_override=None):
            if device_override == "cpu":
                self.cpu_predictor.device_overrides.append(device_override)
                return self.cpu_predictor
            return self.gpu_predictor

        def get_interactive_image_predictor(self):
            raise AssertionError("interactive predictor should not be used for text-only prompt")

    backend = SAM3Ultralytics(model_loader=FallbackLoader())
    prediction = backend.predict_image(sample_image, text_prompt="person")
    assert prediction.labels == ["person"]
    assert backend.model_loader.gpu_predictor.calls == 1
    assert backend.model_loader.cpu_predictor.device_overrides == ["cpu"]


def test_text_prompt_image_inference(image_result):
    predictor = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=predictor))
    result = backend.predict_image("sample.png", text_prompt="person")
    assert len(result.objects) == 1
    assert result.labels == ["person"]
    assert predictor.calls[0][1]["text"] == ["person"]


def test_save_results_merged_mask_includes_manual_mask_and_dilation(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    obj_mask = np.zeros((8, 8), dtype=bool)
    obj_mask[2:4, 2:4] = True
    result = PredictionResult(
        source="sample.png",
        frame_index=None,
        mode="image",
        image_size=(8, 8),
        objects=[SegmentationObject(mask=obj_mask, box=(2, 2, 3, 3), score=0.9, label="person", track_id=1, object_index=1)],
        image=image,
    )
    manual_mask = np.zeros((8, 8), dtype=np.float32)
    manual_mask[5:6, 5:6] = 1.0
    outputs = save_results(
        result,
        mask_dir=tmp_path,
        save_overlay=False,
        save_json=False,
        merged_mask_only=True,
        manual_masks_by_key={"sample.png": manual_mask},
        dilation_pixels=1,
    )
    merged_path = Path(outputs["merged_mask"])
    merged = cv2.imread(str(merged_path), cv2.IMREAD_GRAYSCALE)
    assert merged is not None
    assert merged[5, 5] == 255
    assert merged[4, 5] == 255


def test_save_results_merged_mask_accepts_manual_mask_path(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = PredictionResult(
        source="sample.png",
        frame_index=None,
        mode="image",
        image_size=(8, 8),
        objects=[],
        image=image,
    )
    manual_mask = np.zeros((8, 8), dtype=np.float32)
    manual_mask[5:6, 5:6] = 1.0
    manual_mask_path = tmp_path / "manual.npy"
    np.save(manual_mask_path, manual_mask)
    outputs = save_results(
        result,
        mask_dir=tmp_path / "out",
        save_overlay=False,
        save_json=False,
        merged_mask_only=True,
        manual_masks_by_key={"sample.png": str(manual_mask_path)},
    )
    merged = cv2.imread(outputs["masks"][0], cv2.IMREAD_GRAYSCALE)
    assert merged is not None
    assert merged[5, 5] == 255


def test_exemplar_prompt_image_inference(image_result):
    backend = SAM3Ultralytics(
        model_loader=FakeLoader(semantic_image=FakePredictor([image_result])),
        exemplar_adapter=FakeExemplarAdapter(),
    )
    result = backend.predict_image("sample.png", exemplar_image="ref.png")
    assert result.prompt_metadata["adapter"] is True
    assert result.labels == ["example"]


def test_mixed_prompt_flow_uses_semantic_predictor(image_result):
    predictor = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=predictor))
    result = backend.predict_image("sample.png", text_prompt="person", boxes=[(1, 1, 7, 7, 1)])
    assert result.prompt_metadata["box_count"] == 1
    assert predictor.calls[0][1]["bboxes"] == [[1.0, 1.0, 7.0, 7.0]]


def test_text_plus_points_refinement_uses_semantic_then_interactive(image_result):
    semantic = FakePredictor([image_result])
    interactive = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=semantic, interactive_image=interactive))
    result = backend.predict_image("sample.png", text_prompt="person", points=[(4, 4, 1)])
    assert semantic.calls
    assert interactive.calls
    assert interactive.calls[0][1]["points"] == [[4.0, 4.0]]
    assert interactive.calls[0][1]["bboxes"] == [[2.0, 2.0, 6.0, 6.0]]
    assert result.prompt_metadata["compatibility_mode"] == "semantic_text_then_interactive_refine"


def test_text_refinement_repeats_single_point_for_multiple_seed_boxes(sample_image):
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 1:4, 1:4] = True
    masks[1, 3:7, 3:7] = True
    multi_result = FakeResult(
        path="sample.png",
        masks=masks,
        boxes={"xyxy": [[1, 1, 4, 4], [3, 3, 7, 7]], "conf": [0.91, 0.89], "cls": [0, 0]},
        names=["person"],
        image=sample_image,
        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0},
        shape=(8, 8),
    )
    semantic = FakePredictor([multi_result])
    interactive = FakePredictor([multi_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=semantic, interactive_image=interactive))
    backend.predict_image("sample.png", text_prompt="person", points=[(4, 4, 1)])
    assert interactive.calls[0][1]["bboxes"] == [[1.0, 1.0, 4.0, 4.0], [3.0, 3.0, 7.0, 7.0]]
    assert interactive.calls[0][1]["points"] == [[4.0, 4.0], [4.0, 4.0]]
    assert interactive.calls[0][1]["labels"] == [1, 1]


def test_mask_prompt_image_inference_resizes_to_prompt_shape(image_result):
    predictor = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    result = backend.predict_image("sample.png", mask_input=np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8))
    assert predictor.calls[0][1]["masks"].shape == (296, 296)
    assert result.prompt_metadata["has_mask_input"] is True
    assert result.prompt_metadata["mask_input"]["kind"] == "probability"


def test_mask_prompt_image_inference_accepts_npy_mask_path(image_result, tmp_path):
    predictor = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8))
    result = backend.predict_image("sample.png", mask_input=mask_path)
    assert predictor.calls[0][1]["masks"].shape == (296, 296)
    assert result.prompt_metadata["has_mask_input"] is True


def test_mask_plus_text_refinement_uses_semantic_then_interactive(image_result):
    semantic = FakePredictor([image_result])
    interactive = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=semantic, interactive_image=interactive))
    result = backend.predict_image("sample.png", text_prompt="person", mask_input=np.ones((8, 8), dtype=np.uint8))
    assert semantic.calls
    assert interactive.calls
    assert interactive.calls[0][1]["masks"] is not None
    assert interactive.calls[0][1]["masks"].shape == (296, 296)
    assert interactive.calls[0][1]["bboxes"] == [[2.0, 2.0, 6.0, 6.0]]
    assert result.prompt_metadata["compatibility_mode"] == "semantic_text_then_interactive_refine"


def test_mask_prompt_internal_override_uses_low_res_tensors(image_result):
    predictor = FakePreparedMaskPredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    backend.predict_image("sample.png", mask_input=np.ones((8, 8), dtype=np.float32), points=[(1, 1, 1), (2, 2, 1)])
    assert predictor.prepared_prompts is not None
    prepared_masks = predictor.prepared_prompts[2]
    assert isinstance(prepared_masks, torch.Tensor)
    assert tuple(prepared_masks.shape) == (2, 1, 296, 296)


def test_video_tracking(image_result, tmp_path):
    video_path = _make_video(tmp_path / "video.mp4")
    frame2 = FakeResult(
        path=str(video_path),
        masks=np.zeros((1, 8, 8), dtype=bool),
        boxes={"xyxy": [[1, 1, 5, 5]], "conf": [0.88], "cls": [0], "ids": [12]},
        names=["person"],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        speed={"preprocess": 1.0},
        shape=(8, 8),
    )
    frame1 = FakeResult(
        path=str(video_path),
        masks=np.zeros((1, 8, 8), dtype=bool),
        boxes={"xyxy": [[1, 1, 4, 4]], "conf": [0.9], "cls": [0], "ids": [12]},
        names=["person"],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        speed={"preprocess": 1.0},
        shape=(8, 8),
    )
    predictor = FakePredictor([frame1, frame2])
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_video=predictor))
    results = backend.track_video(str(video_path), text_prompt="person")
    assert len(results) == 2
    assert results[0].frame_index == 0
    assert results[1].track_ids == [12]


def test_video_mask_initialization_uses_interactive_video(image_result, tmp_path):
    video_path = _make_video(tmp_path / "mask_video.mp4")
    frame1 = FakeResult(
        path=str(video_path),
        masks=np.zeros((1, 8, 8), dtype=bool),
        boxes={"xyxy": [[1, 1, 4, 4]], "conf": [0.9], "cls": [0], "ids": [5]},
        names=["person"],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        speed={"preprocess": 1.0},
        shape=(8, 8),
    )
    frame2 = FakeResult(
        path=str(video_path),
        masks=np.zeros((1, 8, 8), dtype=bool),
        boxes={"xyxy": [[2, 2, 5, 5]], "conf": [0.88], "cls": [0], "ids": [5]},
        names=["person"],
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        speed={"preprocess": 1.0},
        shape=(8, 8),
    )
    predictor = FakePredictor([frame1, frame2])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_video=predictor))
    results = backend.track_video(str(video_path), mask_input=np.ones((8, 8), dtype=np.uint8))
    assert len(results) == 2
    assert predictor.calls[0][1]["masks"] is not None


def test_predict_batch_expands_image_directory(tmp_path, image_result):
    for index in range(2):
        image_path = tmp_path / f"image_{index}.png"
        cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))
    predictor = FakePredictor([image_result])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    results = backend.predict_batch([tmp_path])
    assert len(results) == 2


def test_unsupported_exemplar_plus_mask(image_result):
    backend = SAM3Ultralytics(
        model_loader=FakeLoader(semantic_image=FakePredictor([image_result])),
        exemplar_adapter=FakeExemplarAdapter(),
    )
    with pytest.raises(UnsupportedPromptError):
        backend.predict_image("sample.png", exemplar_image="ref.png", mask_input=np.ones((8, 8), dtype=np.uint8))


def test_mask_export_existing_directory(tmp_path, image_result):
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=FakePredictor([image_result])))
    result = backend.predict_image("sample.png", text_prompt="person")
    export_dir = tmp_path / "masks"
    export_dir.mkdir()
    paths = backend.save_results(result, mask_dir=export_dir, save_overlay=False, save_json=False)
    assert Path(paths["masks"][0]).exists()


def test_mask_export_new_directory(tmp_path, image_result):
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=FakePredictor([image_result])))
    result = backend.predict_image("sample.png", text_prompt="person")
    export_dir = tmp_path / "new_masks"
    backend.save_results(result, mask_dir=export_dir, save_overlay=False, save_json=False)
    assert export_dir.exists()


def test_invalid_export_path(tmp_path, image_result):
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=FakePredictor([image_result])))
    result = backend.predict_image("sample.png", text_prompt="person")
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ExportError):
        backend.save_results(result, mask_dir=file_path)


def test_deterministic_mask_naming():
    assert image_mask_filename("sample", 1) == "sample_mask_001.png"
    assert image_mask_filename("sample", 1, track_id=12) == "sample_track_012.png"
    assert video_mask_filename(0, 1) == "frame_000001_obj_001.png"
    assert video_mask_filename(0, 1, track_id=12) == "frame_000001_track_012.png"


def test_video_export_structure(tmp_path):
    video_path = _make_video(tmp_path / "sample.mp4")
    results = [
        PredictionResult(
            source=str(video_path),
            frame_index=0,
            mode="video",
            image_size=(8, 8),
            objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=4, object_index=1)],
        ),
        PredictionResult(
            source=str(video_path),
            frame_index=1,
            mode="video",
            image_size=(8, 8),
            objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=4, object_index=1)],
        ),
    ]
    paths = save_results(results, output_dir=tmp_path / "export", mask_dir=tmp_path / "export" / "masks")
    assert Path(paths["annotated_video"]).exists()
    assert any("frame_000001_track_004.png" in path for path in paths["masks"])


def test_image_batch_export_structure(tmp_path):
    results = [
        PredictionResult(
            source="first.png",
            frame_index=None,
            mode="image",
            image_size=(8, 8),
            objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=None, object_index=1)],
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        PredictionResult(
            source="second.png",
            frame_index=None,
            mode="image",
            image_size=(8, 8),
            objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=None, object_index=1)],
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
    ]
    paths = save_results(results, mask_dir=tmp_path / "masks", output_dir=tmp_path / "out")
    assert len(paths["items"]) == 2
    assert len(paths["masks"]) == 2


def test_merged_mask_only_export_writes_single_mask(tmp_path, image_result):
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=FakePredictor([image_result])))
    result = backend.predict_image("sample.png", text_prompt="person")
    paths = backend.save_results(result, mask_dir=tmp_path / "merged", save_overlay=False, save_json=False, merged_mask_only=True)
    assert len(paths["masks"]) == 1
    assert paths["masks"][0].endswith("sample_merged_mask.png")


def test_merged_mask_only_export_includes_prompt_mask(tmp_path):
    prompt_mask = np.zeros((8, 8), dtype=np.uint8)
    prompt_mask[1:3, 1:3] = 1
    result = PredictionResult(
        source="prompt_only.png",
        frame_index=None,
        mode="image",
        image_size=(8, 8),
        objects=[],
        prompt_metadata={"has_mask_input": True},
        prompt_mask=prompt_mask,
    )
    paths = save_results(result, mask_dir=tmp_path / "merged_prompt", save_overlay=False, save_json=False, merged_mask_only=True)
    image = cv2.imread(paths["masks"][0], cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert int(image[1, 1]) == 255
    assert int(image[0, 0]) == 0


def test_inverted_mask_export_flips_written_pixels(tmp_path, image_result):
    backend = SAM3Ultralytics(model_loader=FakeLoader(semantic_image=FakePredictor([image_result])))
    result = backend.predict_image("sample.png", text_prompt="person")
    paths = backend.save_results(result, mask_dir=tmp_path / "invert", save_overlay=False, save_json=False, invert_mask=True)
    image = cv2.imread(paths["masks"][0], cv2.IMREAD_GRAYSCALE)
    assert image is not None
    assert int(image[0, 0]) == 255
    assert int(image[3, 3]) == 0


def test_predict_image_sequence_reuses_first_result_mask(tmp_path, image_result):
    class RecordingBackend(SAM3Ultralytics):
        def __init__(self):
            super().__init__(model_loader=FakeLoader())
            self.masks = []

        def predict_image(self, source, **kwargs):
            mask_input = kwargs.get("mask_input")
            self.masks.append(None if mask_input is None else np.asarray(mask_input, dtype=np.float32).copy())
            return PredictionResult(
                source=str(source),
                frame_index=None,
                mode="image",
                image_size=(8, 8),
                objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=None, object_index=1)],
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    backend = RecordingBackend()
    sources = [tmp_path / "a.png", tmp_path / "b.png"]
    for source in sources:
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
    initial_mask = np.zeros((8, 8), dtype=np.float32)
    initial_mask[0, 0] = 1.0
    backend.predict_image_sequence(sources, mask_input=initial_mask, reuse_first_mask=True)
    assert backend.masks[0][0, 0] == 1.0
    assert backend.masks[1].shape == (8, 8)
    assert float(backend.masks[1].max()) == 1.0


def test_predict_image_sequence_uses_per_source_masks(tmp_path):
    class RecordingBackend(SAM3Ultralytics):
        def __init__(self):
            super().__init__(model_loader=FakeLoader())
            self.masks = []

        def predict_image(self, source, **kwargs):
            mask_input = kwargs.get("mask_input")
            self.masks.append(None if mask_input is None else np.asarray(mask_input, dtype=np.float32).copy())
            return PredictionResult(
                source=str(source),
                frame_index=None,
                mode="image",
                image_size=(8, 8),
                objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=None, object_index=1)],
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    backend = RecordingBackend()
    sources = [tmp_path / "a.png", tmp_path / "b.png"]
    for source in sources:
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
    mask_inputs = {
        str(sources[0]): np.ones((8, 8), dtype=np.float32),
        str(sources[1]): np.eye(8, dtype=np.float32),
    }
    backend.predict_image_sequence(sources, mask_inputs=mask_inputs)
    assert float(backend.masks[0].sum()) == 64.0
    assert float(backend.masks[1].sum()) == 8.0


def test_track_image_sequence_assigns_stable_track_id(tmp_path, sample_image):
    frame1 = FakeResult(
        path="first.png",
        masks=np.pad(np.ones((1, 4, 4), dtype=bool), ((0, 0), (2, 2), (2, 2))),
        boxes={"xyxy": [[2, 2, 5, 5]], "conf": [0.95], "cls": [0]},
        names=["cart"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )
    frame2_mask = np.zeros((1, 8, 8), dtype=bool)
    frame2_mask[0, 3:7, 3:7] = True
    frame2 = FakeResult(
        path="second.png",
        masks=frame2_mask,
        boxes={"xyxy": [[3, 3, 6, 6]], "conf": [0.91], "cls": [0]},
        names=["cart"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )
    predictor = SequentialPredictor([frame1, frame2])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    sources = [tmp_path / "a.png", tmp_path / "b.png"]
    for source in sources:
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
    initial_mask = np.zeros((8, 8), dtype=np.float32)
    initial_mask[2:6, 2:6] = 1.0
    results = backend.track_image_sequence(sources, mask_input=initial_mask, mask_id=7, mask_label="cart")
    assert [item.objects[0].track_id for item in results] == [7, 7]
    assert results[0].prompt_metadata["mask_input"]["id"] == 7
    assert results[1].tracking_metadata["initial_mask_id"] == 7


def test_track_image_sequence_keeps_only_prompted_target_and_overrides_label(tmp_path, sample_image):
    frame1_masks = np.zeros((2, 8, 8), dtype=bool)
    frame1_masks[0, 0:2, 0:2] = True
    frame1_masks[1, 2:6, 2:6] = True
    frame1 = FakeResult(
        path="first.png",
        masks=frame1_masks,
        boxes={"xyxy": [[0, 0, 1, 1], [2, 2, 5, 5]], "conf": [0.4, 0.95], "cls": [0, 1]},
        names=["other", "cart"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )
    frame2_masks = np.zeros((2, 8, 8), dtype=bool)
    frame2_masks[0, 1:3, 1:3] = True
    frame2_masks[1, 3:7, 3:7] = True
    frame2 = FakeResult(
        path="second.png",
        masks=frame2_masks,
        boxes={"xyxy": [[1, 1, 2, 2], [3, 3, 6, 6]], "conf": [0.3, 0.9], "cls": [0, 1]},
        names=["other", "cart"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )
    predictor = SequentialPredictor([frame1, frame2])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    sources = [tmp_path / "a.png", tmp_path / "b.png"]
    for source in sources:
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
    initial_mask = np.zeros((8, 8), dtype=np.float32)
    initial_mask[2:6, 2:6] = 1.0
    results = backend.track_image_sequence(sources, mask_input=initial_mask, mask_id=1, mask_label="jeff")
    assert [len(item.objects) for item in results] == [1, 1]
    assert [item.objects[0].track_id for item in results] == [1, 1]
    assert [item.objects[0].label for item in results] == ["jeff", "jeff"]
    assert results[0].tracking_metadata["candidate_object_count"] == 2
    assert results[1].tracking_metadata["active_track_ids"] == [1]


def test_predict_image_sequence_first_frame_mask_keeps_only_prompted_target(tmp_path, sample_image):
    frame1_masks = np.zeros((2, 8, 8), dtype=bool)
    frame1_masks[0, 0:2, 0:2] = True
    frame1_masks[1, 2:6, 2:6] = True
    frame1 = FakeResult(
        path="first.png",
        masks=frame1_masks,
        boxes={"xyxy": [[0, 0, 1, 1], [2, 2, 5, 5]], "conf": [0.4, 0.95], "cls": [0, 1]},
        names=["other", "cart"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )
    frame2_masks = np.zeros((3, 8, 8), dtype=bool)
    frame2_masks[0, 1:3, 1:3] = True
    frame2_masks[1, 3:7, 3:7] = True
    frame2_masks[2, 1:3, 5:7] = True
    frame2 = FakeResult(
        path="second.png",
        masks=frame2_masks,
        boxes={"xyxy": [[1, 1, 2, 2], [3, 3, 6, 6], [5, 1, 6, 2]], "conf": [0.3, 0.9, 0.5], "cls": [0, 1, 0]},
        names=["other", "cart"],
        image=sample_image,
        speed={"inference": 1.0},
        shape=(8, 8),
    )
    predictor = SequentialPredictor([frame1, frame2])
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    sources = [tmp_path / "a.png", tmp_path / "b.png"]
    for source in sources:
        cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
    initial_mask = np.zeros((8, 8), dtype=np.float32)
    initial_mask[2:6, 2:6] = 1.0
    results = backend.predict_image_sequence(
        sources,
        mask_input=initial_mask,
        first_mask_initializer_only=True,
        mask_id=1,
        mask_label="jeff",
    )
    assert [len(item.objects) for item in results] == [1, 1]
    assert [item.objects[0].track_id for item in results] == [1, 1]
    assert [item.objects[0].label for item in results] == ["jeff", "jeff"]
    assert results[1].tracking_metadata["tracking_mode"] == "image_sequence_prompt_target_filter"
    assert results[1].tracking_metadata["active_track_ids"] == [1]


def test_track_video_with_mask_id_uses_compatibility_tracking(tmp_path, sample_image):
    video_path = _make_video(tmp_path / "tracked.mp4", frames=2)
    frame1_mask = np.zeros((1, 8, 8), dtype=bool)
    frame1_mask[0, 1:5, 1:5] = True
    frame2_mask = np.zeros((1, 8, 8), dtype=bool)
    frame2_mask[0, 2:6, 2:6] = True
    predictor = SequentialPredictor(
        [
            FakeResult(
                path=str(video_path),
                masks=frame1_mask,
                boxes={"xyxy": [[1, 1, 4, 4]], "conf": [0.93], "cls": [0]},
                names=["person"],
                image=sample_image,
                speed={"inference": 1.0},
                shape=(8, 8),
            ),
            FakeResult(
                path=str(video_path),
                masks=frame2_mask,
                boxes={"xyxy": [[2, 2, 5, 5]], "conf": [0.9], "cls": [0]},
                names=["person"],
                image=sample_image,
                speed={"inference": 1.0},
                shape=(8, 8),
            ),
        ]
    )
    backend = SAM3Ultralytics(model_loader=FakeLoader(interactive_image=predictor))
    initial_mask = np.zeros((8, 8), dtype=np.float32)
    initial_mask[1:5, 1:5] = 1.0
    results = backend.track_video(str(video_path), mask_input=initial_mask, mask_id=12, mask_label="person")
    assert len(results) == 2
    assert [item.objects[0].track_id for item in results] == [12, 12]
    assert results[0].tracking_metadata["tracking_mode"] == "video_mask_initialized_tracking"

def test_predict_video_frames_uses_first_mask_only(tmp_path):
    class RecordingBackend(SAM3Ultralytics):
        def __init__(self):
            super().__init__(model_loader=FakeLoader())
            self.masks = []

        def predict_image(self, source, **kwargs):
            mask_input = kwargs.get("mask_input")
            self.masks.append(None if mask_input is None else np.asarray(mask_input, dtype=np.float32).copy())
            return PredictionResult(
                source="video.mp4",
                frame_index=None,
                mode="image",
                image_size=(8, 8),
                objects=[SegmentationObject(mask=np.ones((8, 8), dtype=bool), box=(0, 0, 7, 7), score=0.9, label="person", track_id=None, object_index=1)],
                image=np.zeros((8, 8, 3), dtype=np.uint8),
            )

    backend = RecordingBackend()
    video_path = _make_video(tmp_path / "video.mp4", frames=2)
    mask = np.ones((8, 8), dtype=np.float32)
    backend.predict_video_frames(video_path, mask_input=mask, first_mask_initializer_only=True)
    assert backend.masks[0] is not None
    assert backend.masks[1] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))












