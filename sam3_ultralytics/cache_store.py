"""Disk-backed cache helpers for GUI masks and inference results."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import PredictionResult, SegmentationObject


class DiskMaskArray:
    """Lazy disk-backed mask wrapper."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def _memmap(self):
        return np.load(self.path, mmap_mode="r")

    def __array__(self, dtype=None):
        array = np.asarray(self._memmap())
        if dtype is not None:
            return array.astype(dtype)
        return array

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        return np.asarray(self).astype(dtype, order=order, casting=casting, subok=subok, copy=copy)

    def copy(self):
        return np.asarray(self).copy()

    @property
    def shape(self):
        return self._memmap().shape

    @property
    def dtype(self):
        return self._memmap().dtype

    def __repr__(self) -> str:
        return f"DiskMaskArray(path={self.path!r})"


@dataclass(slots=True)
class CacheStore:
    """Manage a writable cache directory for GUI state."""

    root: Path

    @classmethod
    def create(cls, root: str | Path) -> "CacheStore":
        store = cls(Path(root))
        store.ensure_ready()
        return store

    def ensure_ready(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_dir.mkdir(parents=True, exist_ok=True)
        return self.root

    @property
    def mask_dir(self) -> Path:
        return self.root / "masks"

    @property
    def result_dir(self) -> Path:
        return self.root / "results"

    @property
    def yolo_dir(self) -> Path:
        return self.root / "yolo"

    def set_root(self, root: str | Path) -> None:
        self.root = Path(root)
        self.ensure_ready()

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.ensure_ready()

    def _safe_token(self, *parts: object) -> str:
        joined = "||".join("" if part is None else str(part) for part in parts)
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
        return digest

    def write_mask(self, namespace: str, key: str, array: np.ndarray) -> str:
        target_dir = self.mask_dir / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{self._safe_token(key)}.npy"
        np.save(target, np.asarray(array, dtype=np.float32))
        return str(target)

    def write_result(self, namespace: str, key: str, result: PredictionResult) -> PredictionResult:
        target_dir = self.result_dir / namespace / self._safe_token(key)
        target_dir.mkdir(parents=True, exist_ok=True)
        cached_objects: list[SegmentationObject] = []
        for obj in result.objects:
            mask_path = target_dir / f"object_{obj.object_index:03d}.npy"
            np.save(mask_path, np.asarray(obj.mask, dtype=np.float32))
            cached_objects.append(
                SegmentationObject(
                    mask=DiskMaskArray(mask_path),
                    box=obj.box,
                    score=obj.score,
                    label=obj.label,
                    track_id=obj.track_id,
                    object_index=obj.object_index,
                )
            )
        prompt_mask = None
        if result.prompt_mask is not None:
            prompt_mask_path = target_dir / "prompt_mask.npy"
            np.save(prompt_mask_path, np.asarray(result.prompt_mask, dtype=np.float32))
            prompt_mask = DiskMaskArray(prompt_mask_path)
        image = None
        if result.image is not None:
            if result.mode == "video" and result.source and Path(str(result.source)).exists():
                image = None
            elif result.source and Path(str(result.source)).exists():
                image = None
            else:
                image = np.asarray(result.image).copy()
        return PredictionResult(
            source=result.source,
            frame_index=result.frame_index,
            mode=result.mode,
            image_size=result.image_size,
            objects=cached_objects,
            prompt_metadata=dict(result.prompt_metadata),
            tracking_metadata=dict(result.tracking_metadata),
            timings=dict(result.timings),
            image=image,
            prompt_mask=prompt_mask,
        )
