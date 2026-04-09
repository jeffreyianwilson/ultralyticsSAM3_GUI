"""Visualization helpers for normalized results."""

from __future__ import annotations

import cv2
import numpy as np

from .schemas import PredictionResult


def color_for_index(index: int) -> tuple[int, int, int]:
    """Return a stable BGR color for an object index."""
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (30, 144, 255),
        (255, 215, 0),
        (138, 43, 226),
        (255, 140, 0),
        (32, 178, 170),
        (220, 20, 60),
    ]
    return palette[index % len(palette)]


def render_overlay(
    image: np.ndarray,
    result: PredictionResult,
    *,
    opacity: float = 0.45,
    show_labels: bool = True,
    show_masks: bool = True,
    show_track_ids: bool = True,
    visible_track_ids: set[int] | None = None,
    visible_labels: set[str] | None = None,
    objects: list | None = None,
) -> np.ndarray:
    """Render a compact overlay for a normalized result."""
    canvas = image.copy()
    if objects is None:
        objects = []
        for obj in result.objects:
            if visible_track_ids is not None and obj.track_id not in visible_track_ids:
                continue
            if visible_labels is not None and obj.label not in visible_labels:
                continue
            objects.append(obj)
    if show_masks:
        tint = np.zeros_like(canvas)
        for obj in objects:
            color = color_for_index(obj.object_index)
            mask = _normalize_union_mask(obj.mask, tuple(canvas.shape[:2]))
            tint[mask] = color
        canvas = cv2.addWeighted(tint, opacity, canvas, 1.0 - opacity, 0.0)

    for obj in objects:
        color = color_for_index(obj.object_index)
        if obj.box is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in obj.box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            parts: list[str] = []
            if show_labels and obj.label:
                parts.append(obj.label)
            if obj.score is not None:
                parts.append(f"{obj.score:.2f}")
            if show_track_ids and obj.track_id is not None:
                parts.append(f"id={obj.track_id}")
            if parts:
                label = " ".join(parts)
                cv2.putText(
                    canvas,
                    label,
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return canvas


def _normalize_union_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[0] <= 4 and array.shape[1:] != array.shape[:2]:
            union = np.any(array, axis=0)
        else:
            union = np.any(array, axis=-1)
    else:
        union = array if array.dtype == np.bool_ else (array > 0)
    if union.dtype != np.bool_:
        union = union > 0
    if shape is not None and tuple(union.shape[:2]) != tuple(shape):
        resized = cv2.resize(union.astype(np.uint8, copy=False), (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
        return resized > 0
    return union


def merged_mask(
    result: PredictionResult,
    *,
    extra_masks: list[np.ndarray] | None = None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Return a merged binary mask for a result, including prompt and extra masks when present."""
    reference_shape: tuple[int, int] | None = shape
    if reference_shape is None and result.image_size is not None:
        reference_shape = tuple(int(value) for value in result.image_size)
    if reference_shape is None and result.objects:
        reference_shape = tuple(result.objects[0].mask.shape[:2])
    elif reference_shape is None and result.prompt_mask is not None:
        reference_shape = tuple(np.asarray(result.prompt_mask).shape[:2])
    elif reference_shape is None and extra_masks:
        reference_shape = tuple(np.asarray(extra_masks[0]).shape[:2])

    if reference_shape is None:
        return None

    union = np.zeros(reference_shape, dtype=np.bool_)
    for obj in result.objects:
        union |= _normalize_union_mask(obj.mask, reference_shape)
    if result.prompt_mask is not None:
        union |= _normalize_union_mask(result.prompt_mask, reference_shape)
    for mask in extra_masks or []:
        union |= _normalize_union_mask(mask, reference_shape)
    return union.astype(np.uint8, copy=False) * 255
