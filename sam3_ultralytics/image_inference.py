"""Image inference helpers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from .exceptions import UnsupportedPromptError
from .prompt_handling import prompt_metadata
from .schemas import BoxPrompt, PredictionResult, PromptPayload, SegmentationObject


def _label_from_names(names: Any, class_index: int | None) -> str | None:
    if class_index is None:
        return None
    if isinstance(names, dict):
        return names.get(class_index)
    if isinstance(names, list) and 0 <= class_index < len(names):
        return names[class_index]
    return None


def _is_numeric_label(label: str | None) -> bool:
    if label is None:
        return False
    text = str(label).strip()
    return bool(text) and text.isdigit()


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_mask = np.asarray(left) > 0
    right_mask = np.asarray(right) > 0
    if left_mask.shape != right_mask.shape:
        return 0.0
    intersection = np.logical_and(left_mask, right_mask).sum()
    union = np.logical_or(left_mask, right_mask).sum()
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _box_iou(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _propagate_semantic_labels(seed: PredictionResult, refined: PredictionResult, prompt_texts: list[str]) -> None:
    semantic_seeds = [obj for obj in seed.objects if obj.label and not _is_numeric_label(obj.label)]
    prompt_labels = [str(text).strip() for text in prompt_texts if str(text).strip()]
    for obj in refined.objects:
        current_label = str(obj.label).strip() if obj.label is not None else ""
        if current_label and not _is_numeric_label(current_label):
            continue
        best_label: str | None = None
        best_score = 0.0
        for seed_obj in semantic_seeds:
            score = _mask_iou(np.asarray(obj.mask), np.asarray(seed_obj.mask))
            if score <= 0.0:
                score = _box_iou(obj.box, seed_obj.box)
            if score > best_score:
                best_score = score
                best_label = str(seed_obj.label).strip()
        if best_label:
            obj.label = best_label
            continue
        if current_label.isdigit() and prompt_labels:
            index = int(current_label)
            if 0 <= index < len(prompt_labels):
                obj.label = prompt_labels[index]
                continue
        if len(prompt_labels) == 1:
            obj.label = prompt_labels[0]
            continue
        obj.label = "Unlabeled"


def normalize_ultralytics_result(result, payload: PromptPayload, *, mode: str, frame_index: int | None) -> PredictionResult:
    """Convert an Ultralytics result into the normalized schema."""
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)

    if masks is None or masks.data is None:
        mask_array = np.zeros((0, *getattr(result, "orig_shape", (0, 0))), dtype=bool)
    else:
        mask_array = masks.data.detach().cpu().numpy().astype(bool)

    box_array = boxes.xyxy.detach().cpu().numpy() if boxes is not None and boxes.xyxy is not None else np.zeros((0, 4))
    score_array = boxes.conf.detach().cpu().numpy() if boxes is not None and boxes.conf is not None else np.zeros((0,))
    class_array = boxes.cls.detach().cpu().numpy().astype(int) if boxes is not None and boxes.cls is not None else None
    id_array = boxes.id.detach().cpu().numpy().astype(int) if boxes is not None and boxes.id is not None else None

    objects: list[SegmentationObject] = []
    for index, mask in enumerate(mask_array):
        box = tuple(float(value) for value in box_array[index]) if index < len(box_array) else None
        cls_idx = int(class_array[index]) if class_array is not None and index < len(class_array) else None
        track_id = int(id_array[index]) if id_array is not None and index < len(id_array) else None
        score = float(score_array[index]) if index < len(score_array) else None
        label = _label_from_names(getattr(result, "names", None), cls_idx)
        objects.append(
            SegmentationObject(
                mask=mask,
                box=box,
                score=score,
                label=label,
                track_id=track_id,
                object_index=index + 1,
            )
        )

    timings = {key: float(value) for key, value in getattr(result, "speed", {}).items()}
    image = getattr(result, "orig_img", None)
    if image is not None:
        image = np.asarray(image).copy()

    prompt_mask = None
    if payload.mask_input is not None and not (mode == "video" and frame_index not in (None, 0)):
        prompt_mask = np.asarray(payload.mask_input).copy()

    return PredictionResult(
        source=str(getattr(result, "path", None) or ""),
        frame_index=frame_index,
        mode=mode,
        image_size=tuple(getattr(result, "orig_shape", (0, 0))),
        inference_image_size=tuple(getattr(result, "orig_shape", (0, 0))),
        objects=objects,
        prompt_metadata=prompt_metadata(payload),
        timings=timings,
        image=image,
        prompt_mask=prompt_mask,
    )


def _coerce_results(raw_results) -> list:
    if isinstance(raw_results, list):
        return raw_results
    return list(raw_results)


def _is_cuda_launch_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in [
            "cuda error",
            "device-side assert",
            "unspecified launch failure",
            "cudalaunchfailure",
            "cuda kernel errors might be asynchronously reported",
        ]
    )


def _semantic_predictor_call(predictor, source, payload: PromptPayload):
    return _coerce_results(
        predictor(
            source=source,
            text=payload.texts,
            bboxes=[list(box.xyxy) for box in payload.boxes] or None,
            labels=[box.label for box in payload.boxes] or None,
        )
    )


def _run_semantic_image_predictor(model_loader, source, payload: PromptPayload):
    predictor = model_loader.get_semantic_image_predictor()
    try:
        return _semantic_predictor_call(predictor, source, payload)
    except Exception as error:
        loader_device = str(getattr(model_loader, "device", ""))
        if not loader_device.startswith("cuda") or not _is_cuda_launch_failure(error):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cpu_predictor = None
        try:
            cpu_predictor = model_loader.get_semantic_image_predictor(device_override="cpu")
        except TypeError:
            raise error
        except Exception:
            raise error
        return _semantic_predictor_call(cpu_predictor, source, payload)


def _predictor_prompt_mask_shape(predictor) -> tuple[int, int]:
    model = getattr(predictor, "model", None)
    if model is None and hasattr(predictor, "setup_model"):
        predictor.setup_model(verbose=False)
        model = getattr(predictor, "model", None)

    prompt_encoder = getattr(model, "sam_prompt_encoder", None) or getattr(model, "prompt_encoder", None)
    mask_input_size = getattr(prompt_encoder, "mask_input_size", None)
    if mask_input_size is not None and len(mask_input_size) == 2:
        return (int(mask_input_size[0]), int(mask_input_size[1]))

    args = getattr(predictor, "args", None)
    imgsz = getattr(predictor, "imgsz", None) or getattr(args, "imgsz", 1024)
    if isinstance(imgsz, int):
        height = width = imgsz
    else:
        if len(imgsz) == 1:
            height = width = int(imgsz[0])
        else:
            height, width = int(imgsz[0]), int(imgsz[1])

    stride = int(getattr(predictor, "stride", 16))
    height = int(np.ceil(height / stride) * stride)
    width = int(np.ceil(width / stride) * stride)
    return ((height // stride) * 4, (width // stride) * 4)


def _resize_interactive_mask_stack(mask_input: np.ndarray, predictor) -> tuple[np.ndarray, bool]:
    array = np.asarray(mask_input, dtype=np.float32)
    squeeze_back = False
    if array.ndim == 2:
        array = array[None, ...]
        squeeze_back = True
    elif array.ndim != 3:
        raise UnsupportedPromptError("Mask prompts must be a 2D mask or an NxHxW mask stack.")

    target_h, target_w = _predictor_prompt_mask_shape(predictor)
    resized = np.stack(
        [cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for mask in array],
        axis=0,
    ).astype(np.float32, copy=False)
    return resized, squeeze_back


def _prepare_interactive_mask_input(mask_input: np.ndarray | None, predictor):
    if mask_input is None:
        return None
    resized, squeeze_back = _resize_interactive_mask_stack(mask_input, predictor)
    return resized[0] if squeeze_back else resized


def _prepare_interactive_mask_tensor(mask_input: np.ndarray | None, predictor, *, prompt_batch_size: int | None = None):
    if mask_input is None:
        return None

    resized, _ = _resize_interactive_mask_stack(mask_input, predictor)
    if prompt_batch_size is not None and prompt_batch_size > 0:
        if resized.shape[0] == 1 and prompt_batch_size > 1:
            resized = np.repeat(resized, prompt_batch_size, axis=0)
        elif resized.shape[0] not in {1, prompt_batch_size}:
            raise UnsupportedPromptError(
                f"Mask prompt count ({resized.shape[0]}) must be 1 or match the interactive prompt count ({prompt_batch_size})."
            )

    resized = resized[:, None, :, :]
    torch_dtype = getattr(predictor, "torch_dtype", torch.float32)
    device = getattr(predictor, "device", "cpu")
    return torch.tensor(resized, dtype=torch_dtype, device=device)


@contextmanager
def _override_interactive_mask_preparation(predictor, mask_input: np.ndarray | None):
    if mask_input is None or not hasattr(predictor, "_prepare_prompts"):
        yield
        return

    original_prepare_prompts = predictor._prepare_prompts

    def patched_prepare_prompts(dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        prepared = original_prepare_prompts(dst_shape, src_shape, bboxes, points, labels, None)
        if len(prepared) == 3:
            prepared_points, prepared_labels, _prepared_masks = prepared
            prompt_batch_size = int(prepared_points.shape[0]) if prepared_points is not None else None
            return (
                prepared_points,
                prepared_labels,
                _prepare_interactive_mask_tensor(mask_input, predictor, prompt_batch_size=prompt_batch_size),
            )
        if len(prepared) == 4:
            prepared_boxes, prepared_points, prepared_labels, _prepared_masks = prepared
            prompt_batch_size = int(prepared_points.shape[0]) if prepared_points is not None else None
            return (
                prepared_boxes,
                prepared_points,
                prepared_labels,
                _prepare_interactive_mask_tensor(mask_input, predictor, prompt_batch_size=prompt_batch_size),
            )
        raise RuntimeError(f"Unexpected prompt preparation payload of length {len(prepared)}.")

    predictor._prepare_prompts = patched_prepare_prompts
    try:
        yield
    finally:
        predictor._prepare_prompts = original_prepare_prompts


def _align_interactive_prompt_batches(boxes: list[list[float]] | None, points: list[list[float]] | None, labels: list[int] | None):
    if boxes is None or points is None:
        return boxes, points, labels

    box_count = len(boxes)
    point_count = len(points)
    if box_count == 0 or point_count == 0 or box_count == point_count:
        return boxes, points, labels

    if box_count > 1 and point_count == 1:
        points = [list(points[0]) for _ in range(box_count)]
        if labels is not None and len(labels) == 1:
            labels = [labels[0] for _ in range(box_count)]
        return boxes, points, labels

    if point_count > 1 and box_count == 1:
        boxes = [list(boxes[0]) for _ in range(point_count)]
        return boxes, points, labels

    raise UnsupportedPromptError(
        f"Cannot align {box_count} boxes with {point_count} interactive point prompts in a single SAM 3 request."
    )


def _run_interactive_image_predictor(model_loader, source, payload: PromptPayload):
    predictor = model_loader.get_interactive_image_predictor()
    bboxes = [list(box.xyxy) for box in payload.boxes] or None
    points = [[point.x, point.y] for point in payload.points] or None
    point_labels = [point.label for point in payload.points] or None
    bboxes, points, point_labels = _align_interactive_prompt_batches(bboxes, points, point_labels)

    prepared_masks = None
    mask_context = nullcontext()
    if payload.mask_input is not None and hasattr(predictor, "_prepare_prompts"):
        mask_context = _override_interactive_mask_preparation(predictor, payload.mask_input)
        prepared_masks = payload.mask_input
    else:
        prepared_masks = _prepare_interactive_mask_input(payload.mask_input, predictor)

    with mask_context:
        return _coerce_results(
            predictor(
                source=source,
                bboxes=bboxes,
                points=points,
                labels=point_labels,
                masks=prepared_masks,
            )
        )


def _dedupe_boxes(boxes: list[BoxPrompt]) -> list[BoxPrompt]:
    deduped: list[BoxPrompt] = []
    seen: set[tuple[float, float, float, float, int]] = set()
    for box in boxes:
        key = (
            round(box.x1, 3),
            round(box.y1, 3),
            round(box.x2, 3),
            round(box.y2, 3),
            int(box.label),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def _boxes_from_result(result: PredictionResult) -> list[BoxPrompt]:
    boxes: list[BoxPrompt] = []
    for obj in result.objects:
        if obj.box is None:
            continue
        boxes.append(BoxPrompt(*obj.box, 1))
    return boxes


def run_image_prediction(model_loader, source, payload: PromptPayload, exemplar_adapter=None) -> PredictionResult:
    """Run image inference and normalize the result."""
    start = perf_counter()
    if payload.has_exemplar:
        if exemplar_adapter is None:
            raise UnsupportedPromptError(
                "Exemplar prompting requires a dedicated compatibility adapter because this Ultralytics SAM 3 build "
                "does not expose a stable public exemplar-prompt API."
            )
        result = exemplar_adapter.predict_image(model_loader=model_loader, source=source, payload=payload)
        result.timings.setdefault("backend_total_ms", (perf_counter() - start) * 1000.0)
        return result

    semantic_only = payload.has_text and not payload.has_mask_input and not payload.has_points
    if semantic_only:
        raw_results = _run_semantic_image_predictor(model_loader, source, payload)
        normalized = normalize_ultralytics_result(raw_results[0], payload, mode="image", frame_index=None)
    elif payload.has_text:
        seed_payload = PromptPayload(texts=list(payload.texts), boxes=list(payload.boxes))
        semantic_seed = normalize_ultralytics_result(
            _run_semantic_image_predictor(model_loader, source, seed_payload)[0],
            seed_payload,
            mode="image",
            frame_index=None,
        )
        refined_payload = PromptPayload(
            texts=list(payload.texts),
            points=list(payload.points),
            boxes=_dedupe_boxes(list(payload.boxes) + _boxes_from_result(semantic_seed)),
            mask_input=payload.mask_input,
            mask_metadata=dict(payload.mask_metadata),
        )
        raw_results = _run_interactive_image_predictor(model_loader, source, refined_payload)
        normalized = normalize_ultralytics_result(raw_results[0], payload, mode="image", frame_index=None)
        _propagate_semantic_labels(semantic_seed, normalized, payload.texts)
        normalized.prompt_metadata["compatibility_mode"] = "semantic_text_then_interactive_refine"
        normalized.prompt_metadata["semantic_seed_object_count"] = len(semantic_seed.objects)
        normalized.prompt_metadata["refinement_box_count"] = len(refined_payload.boxes)
    else:
        raw_results = _run_interactive_image_predictor(model_loader, source, payload)
        normalized = normalize_ultralytics_result(raw_results[0], payload, mode="image", frame_index=None)

    normalized.timings["backend_total_ms"] = (perf_counter() - start) * 1000.0
    if not normalized.source:
        normalized.source = str(source) if isinstance(source, (str, Path)) else None
    return normalized


