"""Typed schemas shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ImageSource = str | Path | np.ndarray | Image.Image
MaskSource = str | Path | np.ndarray | Image.Image


@dataclass(slots=True)
class PointPrompt:
    """A single point prompt."""

    x: float
    y: float
    label: int = 1


@dataclass(slots=True)
class BoxPrompt:
    """A single box prompt in xyxy pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: int = 1

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as an xyxy tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class PromptPayload:
    """Normalized prompt payload."""

    texts: list[str] = field(default_factory=list)
    points: list[PointPrompt] = field(default_factory=list)
    boxes: list[BoxPrompt] = field(default_factory=list)
    mask_input: np.ndarray | None = None
    mask_metadata: dict[str, Any] = field(default_factory=dict)
    exemplar_image: ImageSource | None = None
    exemplar_box: BoxPrompt | None = None

    @property
    def has_text(self) -> bool:
        """Whether text prompts are present."""
        return bool(self.texts)

    @property
    def has_points(self) -> bool:
        """Whether point prompts are present."""
        return bool(self.points)

    @property
    def has_boxes(self) -> bool:
        """Whether box prompts are present."""
        return bool(self.boxes)

    @property
    def has_exemplar(self) -> bool:
        """Whether an exemplar prompt is present."""
        return self.exemplar_image is not None

    @property
    def has_mask_input(self) -> bool:
        """Whether a mask prompt is present."""
        return self.mask_input is not None

    @property
    def is_empty(self) -> bool:
        """Whether no prompt has been supplied."""
        return not any(
            [
                self.has_text,
                self.has_points,
                self.has_boxes,
                self.has_exemplar,
                self.has_mask_input,
            ]
        )


@dataclass(slots=True)
class SegmentationObject:
    """A normalized segmentation object."""

    mask: np.ndarray = field(repr=False)
    box: tuple[float, float, float, float] | None
    score: float | None
    label: str | None
    track_id: int | None
    object_index: int

    def to_dict(self, mask_path: str | None = None) -> dict[str, Any]:
        """Serialize the object without embedding mask pixels."""
        payload: dict[str, Any] = {
            "index": self.object_index,
            "box": list(self.box) if self.box is not None else None,
            "score": self.score,
            "label": self.label,
            "track_id": self.track_id,
        }
        if mask_path is not None:
            payload["mask_path"] = mask_path
        return payload


@dataclass(slots=True)
class PredictionResult:
    """Normalized inference result independent from Ultralytics internals."""

    source: str | None
    frame_index: int | None
    mode: str
    image_size: tuple[int, int] | None
    objects: list[SegmentationObject] = field(default_factory=list)
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    tracking_metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    image: np.ndarray | None = field(default=None, repr=False)
    prompt_mask: np.ndarray | None = field(default=None, repr=False)

    def __len__(self) -> int:
        """Return the object count."""
        return len(self.objects)

    @property
    def masks(self) -> list[np.ndarray]:
        """Return normalized masks."""
        return [item.mask for item in self.objects]

    @property
    def boxes(self) -> list[tuple[float, float, float, float] | None]:
        """Return normalized boxes."""
        return [item.box for item in self.objects]

    @property
    def scores(self) -> list[float | None]:
        """Return normalized scores."""
        return [item.score for item in self.objects]

    @property
    def labels(self) -> list[str | None]:
        """Return normalized labels."""
        return [item.label for item in self.objects]

    @property
    def track_ids(self) -> list[int | None]:
        """Return normalized track ids."""
        return [item.track_id for item in self.objects]

    def to_dict(self, mask_paths: list[str] | None = None) -> dict[str, Any]:
        """Serialize the result without embedding mask pixels."""
        if mask_paths is None:
            mask_paths = [None] * len(self.objects)
        return {
            "source": self.source,
            "frame_index": self.frame_index,
            "mode": self.mode,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "object_count": len(self.objects),
            "objects": [
                item.to_dict(mask_path=mask_paths[index]) for index, item in enumerate(self.objects)
            ],
            "prompt_metadata": self.prompt_metadata,
            "tracking_metadata": self.tracking_metadata,
            "timings": self.timings,
            "has_prompt_mask": self.prompt_mask is not None,
        }
