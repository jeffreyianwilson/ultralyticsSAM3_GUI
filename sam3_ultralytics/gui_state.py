"""Session state for the PySide6 GUI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schemas import PredictionResult


@dataclass(slots=True)
class GUIState:
    """Mutable GUI state stored on the main window."""

    source_path: str | None = None
    source_kind: str = "image"
    source_items: list[str] = field(default_factory=list)
    source_frame_count: int | None = None
    export_dir: str | None = None
    cache_dir: str | None = None
    compact_cache_enabled: bool = True
    inference_scale_enabled: bool = False
    inference_scale: float = 1.0
    mask_path: str | None = None
    mask_input: np.ndarray | None = None
    mask_source: str | None = None
    mask_id: int | None = None
    next_mask_id: int = 1
    mask_inputs_by_key: dict[str, str] = field(default_factory=dict)
    mask_paths_by_key: dict[str, str] = field(default_factory=dict)
    mask_sources_by_key: dict[str, str] = field(default_factory=dict)
    mask_ids_by_key: dict[str, int] = field(default_factory=dict)
    mask_class: str | None = None
    mask_classes_by_key: dict[str, str] = field(default_factory=dict)
    manual_mask_input: np.ndarray | None = None
    manual_masks_by_key: dict[str, str] = field(default_factory=dict)
    points: list[tuple[float, float, int]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float, int]] = field(default_factory=list)
    points_by_key: dict[str, list[tuple[float, float, int]]] = field(default_factory=dict)
    boxes_by_key: dict[str, list[tuple[float, float, float, float, int]]] = field(default_factory=dict)
    results: PredictionResult | list[PredictionResult | None] | None = None
    current_frame_index: int = 0
    playing: bool = False
