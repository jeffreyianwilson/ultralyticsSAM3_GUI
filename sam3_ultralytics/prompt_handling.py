"""Prompt normalization and validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .exceptions import UnsupportedPromptError
from .io_utils import normalize_mask_input
from .schemas import BoxPrompt, MaskSource, PointPrompt, PromptPayload


def _normalize_texts(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [part.strip() for part in value.split(",")]
    else:
        candidates = [str(part).strip() for part in value]
    return [candidate for candidate in candidates if candidate]


def _normalize_points(value: Sequence[PointPrompt | Sequence[float]] | None) -> list[PointPrompt]:
    if value is None:
        return []
    points: list[PointPrompt] = []
    for item in value:
        if isinstance(item, PointPrompt):
            points.append(item)
            continue
        raw = list(item)
        if len(raw) not in {2, 3}:
            raise UnsupportedPromptError("Point prompts must contain x, y, and optional label.")
        label = int(raw[2]) if len(raw) == 3 else 1
        points.append(PointPrompt(float(raw[0]), float(raw[1]), label))
    return points


def _normalize_boxes(value: Sequence[BoxPrompt | Sequence[float]] | None) -> list[BoxPrompt]:
    if value is None:
        return []
    boxes: list[BoxPrompt] = []
    for item in value:
        if isinstance(item, BoxPrompt):
            boxes.append(item)
            continue
        raw = list(item)
        if len(raw) not in {4, 5}:
            raise UnsupportedPromptError("Box prompts must contain x1, y1, x2, y2, and optional label.")
        label = int(raw[4]) if len(raw) == 5 else 1
        boxes.append(BoxPrompt(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]), label))
    return boxes


def build_prompt_payload(
    text_prompt: str | Sequence[str] | None = None,
    points: Sequence[PointPrompt | Sequence[float]] | None = None,
    boxes: Sequence[BoxPrompt | Sequence[float]] | None = None,
    mask_input: MaskSource | None = None,
    exemplar_image=None,
    exemplar_box: BoxPrompt | Sequence[float] | None = None,
    mask_metadata: dict[str, Any] | None = None,
) -> PromptPayload:
    """Build a normalized prompt payload."""
    normalized_exemplar_box = None
    if exemplar_box is not None:
        normalized_exemplar_box = _normalize_boxes([exemplar_box])[0]
    normalized_mask, normalized_mask_metadata = normalize_mask_input(mask_input)
    merged_mask_metadata = dict(normalized_mask_metadata)
    if mask_metadata:
        merged_mask_metadata.update({key: value for key, value in mask_metadata.items() if value is not None})
    return PromptPayload(
        texts=_normalize_texts(text_prompt),
        points=_normalize_points(points),
        boxes=_normalize_boxes(boxes),
        mask_input=normalized_mask,
        mask_metadata=merged_mask_metadata,
        exemplar_image=exemplar_image,
        exemplar_box=normalized_exemplar_box,
    )


def validate_prompt_payload(payload: PromptPayload, *, is_video: bool = False) -> None:
    """Validate prompt combinations against supported runtime behavior."""
    if payload.has_exemplar and payload.has_mask_input:
        raise UnsupportedPromptError("Exemplar prompting and mask prompting cannot be combined in the current adapter boundary.")


def prompt_metadata(payload: PromptPayload) -> dict[str, object]:
    """Return a JSON-friendly description of the prompt state."""
    return {
        "texts": list(payload.texts),
        "point_count": len(payload.points),
        "box_count": len(payload.boxes),
        "has_mask_input": payload.has_mask_input,
        "mask_input": dict(payload.mask_metadata) if payload.mask_metadata else None,
        "has_exemplar": payload.has_exemplar,
    }
