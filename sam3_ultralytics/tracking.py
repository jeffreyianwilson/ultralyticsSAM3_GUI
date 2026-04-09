"""Tracking-oriented helpers."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .exceptions import InferenceCancelledError
from .inference_scaling import apply_inference_transform, normalize_inference_scale, prepare_inference_source
from .image_inference import run_image_prediction
from .io_utils import expand_sources, read_video_frame, video_frame_count
from .prompt_handling import build_prompt_payload, validate_prompt_payload
from .schemas import PredictionResult, SegmentationObject
from .video_inference import run_video_prediction


def _binary_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[0] <= 4 and array.shape[1:] != array.shape[:2]:
            array = np.max(array, axis=0)
        else:
            array = np.max(array, axis=-1)
    return (array > 0).astype(bool, copy=False)


def _mask_iou(left: np.ndarray | None, right: np.ndarray | None) -> float:
    left_mask = _binary_mask(left)
    right_mask = _binary_mask(right)
    if left_mask is None or right_mask is None:
        return 0.0
    if left_mask.shape != right_mask.shape:
        return 0.0
    intersection = np.logical_and(left_mask, right_mask).sum()
    union = np.logical_or(left_mask, right_mask).sum()
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _mask_box(mask: np.ndarray | None) -> tuple[float, float, float, float] | None:
    binary = _binary_mask(mask)
    if binary is None or not binary.any():
        return None
    ys, xs = np.where(binary)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _resolve_text(prompt_value: Any, fallback_label: str | None) -> str | None:
    if prompt_value is None:
        return fallback_label
    if isinstance(prompt_value, str):
        text = prompt_value.strip()
        return text or fallback_label
    values = [str(item).strip() for item in prompt_value if str(item).strip()]
    if values:
        return ", ".join(values)
    return fallback_label


def _pick_tracked_object(result: PredictionResult, reference_mask: np.ndarray | None) -> tuple[SegmentationObject | None, float, int | None]:
    if not result.objects:
        return None, 0.0, None
    if reference_mask is None:
        return result.objects[0], 0.0, 0
    best_index = 0
    best_score = -1.0
    for index, obj in enumerate(result.objects):
        score = _mask_iou(obj.mask, reference_mask)
        if score > best_score:
            best_score = score
            best_index = index
    return result.objects[best_index], max(best_score, 0.0), best_index


def _fallback_object(mask: np.ndarray | None, *, label: str | None, track_id: int, object_index: int = 1) -> SegmentationObject | None:
    binary = _binary_mask(mask)
    if binary is None or not binary.any():
        return None
    return SegmentationObject(
        mask=binary,
        box=_mask_box(binary),
        score=None,
        label=label,
        track_id=track_id,
        object_index=object_index,
    )


def _assign_tracking_identity(
    result: PredictionResult,
    *,
    reference_mask: np.ndarray | None,
    track_id: int,
    mask_label: str | None,
    fallback_allowed: bool,
) -> tuple[np.ndarray | None, float, bool]:
    candidate_count = len(result.objects)
    tracked_obj, tracked_iou, tracked_index = _pick_tracked_object(result, reference_mask)
    used_fallback = False

    if tracked_obj is None and fallback_allowed:
        tracked_obj = _fallback_object(reference_mask, label=mask_label, track_id=track_id)
        if tracked_obj is not None:
            result.objects = [tracked_obj]
            tracked_index = 0
            used_fallback = True

    if tracked_obj is None:
        result.tracking_metadata.update(
            {
                "candidate_object_count": candidate_count,
                "tracked_mask_iou": 0.0,
                "tracked_object_index": None,
                "tracked_object_found": False,
                "used_fallback_mask": False,
                "active_track_ids": [],
            }
        )
        return reference_mask, 0.0, False

    tracked_obj.track_id = track_id
    tracked_obj.object_index = 1
    if mask_label:
        tracked_obj.label = mask_label
    result.objects = [tracked_obj]
    result.tracking_metadata.update(
        {
            "candidate_object_count": candidate_count,
            "tracked_mask_iou": float(tracked_iou),
            "tracked_object_index": None if tracked_index is None else tracked_index + 1,
            "tracked_object_found": True,
            "used_fallback_mask": used_fallback,
            "active_track_ids": [track_id],
        }
    )
    tracked_mask = _binary_mask(tracked_obj.mask)
    if tracked_mask is not None:
        tracked_mask = np.array(tracked_mask, dtype=np.bool_, copy=True)
    return tracked_mask, float(tracked_iou), used_fallback


def _resolve_per_frame_value(mapping: dict | None, key, fallback):
    if mapping is None:
        return fallback
    return mapping.get(key, fallback)


def _tracking_mask_metadata(
    *,
    track_id: int | None,
    mask_label: str | None,
    frame_key,
    frame_index: int,
    is_video: bool,
    is_initial_frame: bool,
    has_override_mask: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "track_id": track_id,
        "class": mask_label,
        "prompt_target_id": track_id,
        "tracking_mode": "video_mask_initialized_tracking" if is_video else "image_sequence_mask_tracking",
        "frame_index": frame_index,
        "is_initial_tracking_frame": is_initial_frame,
        "frame_key": frame_key,
        "has_mask_correction": has_override_mask,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def track_image_sequence(
    model_loader,
    sources,
    *,
    text_prompt=None,
    exemplar_image=None,
    exemplar_box=None,
    points=None,
    boxes=None,
    mask_input=None,
    mask_inputs: dict[str, np.ndarray] | None = None,
    points_by_source: dict[str, Any] | None = None,
    boxes_by_source: dict[str, Any] | None = None,
    text_prompts_by_source: dict[str, Any] | None = None,
    mask_id: int | None = None,
    mask_label: str | None = None,
    inference_scale: float = 1.0,
    exemplar_adapter=None,
    progress_callback=None,
    cancel_callback=None,
) -> list[PredictionResult]:
    """Track a mask-initialized object across an image sequence."""
    expanded_sources = expand_sources(sources)
    tracked_mask = _binary_mask(mask_input)
    if tracked_mask is not None:
        tracked_mask = np.array(tracked_mask, dtype=np.bool_, copy=True)
    resolved_track_id = int(mask_id or 1)
    resolved_mask_label = _resolve_text(text_prompt, mask_label)
    results: list[PredictionResult] = []
    total = len(expanded_sources)
    start = perf_counter()
    resolved_scale = normalize_inference_scale(inference_scale)

    for index, source in enumerate(expanded_sources):
        if cancel_callback is not None and cancel_callback():
            raise InferenceCancelledError("Image sequence tracking was cancelled.")

        source_key = str(source)
        frame_points = _resolve_per_frame_value(points_by_source, source_key, points)
        frame_boxes = _resolve_per_frame_value(boxes_by_source, source_key, boxes)
        frame_text_prompt = _resolve_text(_resolve_per_frame_value(text_prompts_by_source, source_key, text_prompt), None)
        override_mask = _resolve_per_frame_value(mask_inputs, source_key, None)
        current_mask = tracked_mask if override_mask is None else _binary_mask(override_mask)
        if current_mask is not None and current_mask is not tracked_mask:
            current_mask = np.array(current_mask, dtype=np.bool_, copy=True)
        inference_source, scaled_points, scaled_boxes, scaled_mask_input, transform = prepare_inference_source(
            source,
            points=frame_points,
            boxes=frame_boxes,
            mask_input=current_mask,
            inference_scale=resolved_scale,
        )
        payload = build_prompt_payload(
            text_prompt=frame_text_prompt,
            points=scaled_points,
            boxes=scaled_boxes,
            mask_input=scaled_mask_input,
            exemplar_image=exemplar_image,
            exemplar_box=exemplar_box,
            mask_metadata=_tracking_mask_metadata(
                track_id=resolved_track_id,
                mask_label=resolved_mask_label,
                frame_key=source_key,
                frame_index=index,
                is_video=False,
                is_initial_frame=index == 0,
                has_override_mask=override_mask is not None,
            ),
        )
        validate_prompt_payload(payload, is_video=False)
        result = run_image_prediction(model_loader, inference_source, payload, exemplar_adapter=exemplar_adapter)
        result = apply_inference_transform(result, transform, source=source)
        result.source = str(source)
        result.tracking_metadata.update(
            {
                "tracking_mode": "image_sequence_mask_tracking",
                "initial_mask_id": resolved_track_id,
                "initial_mask_class": resolved_mask_label,
                "frame_index": index,
                "source_key": source_key,
            }
        )
        tracked_mask, tracked_iou, _used_fallback = _assign_tracking_identity(
            result,
            reference_mask=current_mask,
            track_id=resolved_track_id,
            mask_label=resolved_mask_label,
            fallback_allowed=current_mask is not None,
        )
        if tracked_mask is not None:
            result.prompt_mask = np.array(tracked_mask, dtype=np.bool_, copy=True)
        result.prompt_metadata.setdefault("mask_input", {})
        if isinstance(result.prompt_metadata["mask_input"], dict):
            result.prompt_metadata["mask_input"]["id"] = resolved_track_id
            if resolved_mask_label:
                result.prompt_metadata["mask_input"]["class"] = resolved_mask_label
        result.tracking_metadata["tracked_mask_iou"] = float(tracked_iou)
        results.append(result)
        if progress_callback is not None:
            progress_callback(index + 1, total, f"Tracked {Path(source).name if isinstance(source, (str, Path)) else 'image'}")

    elapsed_ms = (perf_counter() - start) * 1000.0
    for item in results:
        item.timings.setdefault("backend_total_ms", elapsed_ms / max(len(results), 1))
    return results


def track_video_frames(
    model_loader,
    source,
    *,
    frame_indices: list[int] | None = None,
    text_prompt=None,
    exemplar_image=None,
    exemplar_box=None,
    points=None,
    boxes=None,
    mask_input=None,
    mask_inputs_by_frame: dict[int, np.ndarray] | None = None,
    points_by_frame: dict[int, Any] | None = None,
    boxes_by_frame: dict[int, Any] | None = None,
    text_prompts_by_frame: dict[int, Any] | None = None,
    mask_id: int | None = None,
    mask_label: str | None = None,
    inference_scale: float = 1.0,
    exemplar_adapter=None,
    progress_callback=None,
    cancel_callback=None,
    item_start_callback=None,
    item_result_callback=None,
) -> list[PredictionResult]:
    """Track a mask-initialized object across selected video frames."""
    if frame_indices is None:
        total_frames = video_frame_count(source)
        frame_indices = list(range(total_frames or 0))
    tracked_mask = _binary_mask(mask_input)
    if tracked_mask is not None:
        tracked_mask = np.array(tracked_mask, dtype=np.bool_, copy=True)
    resolved_track_id = int(mask_id or 1)
    resolved_mask_label = _resolve_text(text_prompt, mask_label)
    results: list[PredictionResult] = []
    total = len(frame_indices)
    start = perf_counter()
    resolved_scale = normalize_inference_scale(inference_scale)

    for position, frame_index in enumerate(frame_indices):
        if cancel_callback is not None and cancel_callback():
            raise InferenceCancelledError("Video tracking was cancelled.")

        if item_start_callback is not None:
            item_start_callback(position, total, f"frame:{frame_index}")
        frame = read_video_frame(source, frame_index)
        frame_points = _resolve_per_frame_value(points_by_frame, frame_index, points)
        frame_boxes = _resolve_per_frame_value(boxes_by_frame, frame_index, boxes)
        frame_text_prompt = _resolve_text(_resolve_per_frame_value(text_prompts_by_frame, frame_index, text_prompt), None)
        override_mask = _resolve_per_frame_value(mask_inputs_by_frame, frame_index, None)
        current_mask = tracked_mask if override_mask is None else _binary_mask(override_mask)
        if current_mask is not None and current_mask is not tracked_mask:
            current_mask = np.array(current_mask, dtype=np.bool_, copy=True)
        inference_source, scaled_points, scaled_boxes, scaled_mask_input, transform = prepare_inference_source(
            frame,
            points=frame_points,
            boxes=frame_boxes,
            mask_input=current_mask,
            inference_scale=resolved_scale,
        )
        payload = build_prompt_payload(
            text_prompt=frame_text_prompt,
            points=scaled_points,
            boxes=scaled_boxes,
            mask_input=scaled_mask_input,
            exemplar_image=exemplar_image,
            exemplar_box=exemplar_box,
            mask_metadata=_tracking_mask_metadata(
                track_id=resolved_track_id,
                mask_label=resolved_mask_label,
                frame_key=str(frame_index),
                frame_index=frame_index,
                is_video=True,
                is_initial_frame=position == 0,
                has_override_mask=override_mask is not None,
            ),
        )
        validate_prompt_payload(payload, is_video=True)
        result = run_image_prediction(model_loader, inference_source, payload, exemplar_adapter=exemplar_adapter)
        result = apply_inference_transform(result, transform, source=frame)
        result.source = str(source)
        result.mode = "video"
        result.frame_index = frame_index
        result.image = frame
        result.tracking_metadata.update(
            {
                "tracking_mode": "video_mask_initialized_tracking",
                "initial_mask_id": resolved_track_id,
                "initial_mask_class": resolved_mask_label,
                "frame_index": frame_index,
                "source_key": str(source),
            }
        )
        tracked_mask, tracked_iou, _used_fallback = _assign_tracking_identity(
            result,
            reference_mask=current_mask,
            track_id=resolved_track_id,
            mask_label=resolved_mask_label,
            fallback_allowed=current_mask is not None,
        )
        if tracked_mask is not None:
            result.prompt_mask = np.array(tracked_mask, dtype=np.bool_, copy=True)
        result.prompt_metadata.setdefault("mask_input", {})
        if isinstance(result.prompt_metadata["mask_input"], dict):
            result.prompt_metadata["mask_input"]["id"] = resolved_track_id
            if resolved_mask_label:
                result.prompt_metadata["mask_input"]["class"] = resolved_mask_label
        result.tracking_metadata["tracked_mask_iou"] = float(tracked_iou)
        results.append(result)
        if item_result_callback is not None:
            item_result_callback(position, total, result, f"frame:{frame_index}")
        if progress_callback is not None:
            progress_callback(position + 1, total, f"Tracked frame {frame_index + 1}")

    elapsed_ms = (perf_counter() - start) * 1000.0
    for item in results:
        item.timings.setdefault("backend_total_ms", elapsed_ms / max(len(results), 1))
    return results


def track_video_sequence(*args, **kwargs):
    """Use the native Ultralytics video tracking flow when compatibility tracking is not required."""
    return run_video_prediction(*args, **kwargs)

