"""Helpers for downscaled inference and restoring result coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_utils import to_bgr_image


@dataclass(slots=True)
class InferenceTransform:
    """Coordinate transform between original source size and inference size."""

    original_size: tuple[int, int]
    inference_size: tuple[int, int]
    scale_x: float
    scale_y: float
    original_image: np.ndarray | None = None


def normalize_inference_scale(value: float | None) -> float:
    """Clamp inference scaling to the supported user range."""
    if value is None:
        return 1.0
    return max(0.1, min(1.0, float(value)))


def _resize_mask(mask_input, target_size: tuple[int, int]):
    array = np.asarray(mask_input, dtype=np.float32)
    target_h, target_w = target_size
    if array.ndim == 2:
        return cv2.resize(array, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
    if array.ndim == 3:
        return np.stack(
            [cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for mask in array],
            axis=0,
        ).astype(np.float32, copy=False)
    raise ValueError("Inference mask scaling expects a 2D mask or NxHxW mask stack.")


def _scale_points(points, *, scale_x: float, scale_y: float):
    if not points:
        return points
    return [(float(x) * scale_x, float(y) * scale_y, int(label)) for x, y, label in points]


def _scale_boxes(boxes, *, scale_x: float, scale_y: float):
    if not boxes:
        return boxes
    return [
        (
            float(x1) * scale_x,
            float(y1) * scale_y,
            float(x2) * scale_x,
            float(y2) * scale_y,
            int(label),
        )
        for x1, y1, x2, y2, label in boxes
    ]


def prepare_inference_source(
    source,
    *,
    points=None,
    boxes=None,
    mask_input=None,
    inference_scale: float = 1.0,
):
    """Return a potentially downscaled inference source and transformed prompts."""
    scale = normalize_inference_scale(inference_scale)
    if scale >= 0.999:
        return source, points, boxes, mask_input, None

    original_image = to_bgr_image(source)
    original_h, original_w = original_image.shape[:2]
    target_h = max(1, int(round(original_h * scale)))
    target_w = max(1, int(round(original_w * scale)))
    resized_image = cv2.resize(original_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    scale_y = target_h / max(original_h, 1)
    scale_x = target_w / max(original_w, 1)

    transform = InferenceTransform(
        original_size=(original_h, original_w),
        inference_size=(target_h, target_w),
        scale_x=scale_x,
        scale_y=scale_y,
        original_image=None if isinstance(source, (str, Path)) else original_image,
    )
    return (
        resized_image,
        _scale_points(points, scale_x=scale_x, scale_y=scale_y),
        _scale_boxes(boxes, scale_x=scale_x, scale_y=scale_y),
        None if mask_input is None else _resize_mask(mask_input, (target_h, target_w)),
        transform,
    )


def apply_inference_transform(result, transform: InferenceTransform | None, *, source=None):
    """Project a downscaled inference result back into original-image coordinates."""
    if result is None:
        return None

    if transform is None:
        if result.inference_image_size is None:
            result.inference_image_size = result.image_size
        if isinstance(source, (str, Path)):
            result.source = str(source)
        return result

    for obj in result.objects:
        if obj.box is not None:
            x1, y1, x2, y2 = obj.box
            obj.box = (
                float(x1) / max(transform.scale_x, 1e-8),
                float(y1) / max(transform.scale_y, 1e-8),
                float(x2) / max(transform.scale_x, 1e-8),
                float(y2) / max(transform.scale_y, 1e-8),
            )

    result.image_size = transform.original_size
    result.inference_image_size = transform.inference_size
    if isinstance(source, (str, Path)):
        result.source = str(source)
        result.image = None
    elif transform.original_image is not None:
        result.image = np.asarray(transform.original_image)
    result.prompt_metadata["inference_scale"] = float(transform.scale_x)
    result.prompt_metadata["inference_image_size"] = list(transform.inference_size)
    return result
