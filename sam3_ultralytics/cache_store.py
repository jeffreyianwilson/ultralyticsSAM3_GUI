"""Disk-backed cache helpers for GUI masks and inference results."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import PredictionResult, SegmentationObject

ARCHIVE_VERSION = 2
_ARCHIVE_CACHE_LIMIT = 12
_ARCHIVE_RUNTIME_CACHE: OrderedDict[str, "_ArchiveRecord"] = OrderedDict()


def _cache_mask_array(array: np.ndarray) -> np.ndarray:
    """Normalize cached masks to compact binary arrays."""
    mask = np.asarray(array)
    if mask.dtype == np.bool_:
        return mask
    if np.issubdtype(mask.dtype, np.floating):
        return np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0) > 0.5
    return mask > 0


def clear_archive_runtime_cache() -> None:
    """Drop any in-process archive payloads cached for rendering."""
    _ARCHIVE_RUNTIME_CACHE.clear()


def _invalidate_archive_runtime_cache(path: str | Path) -> None:
    """Drop a specific archive record so overwrite writes are visible immediately."""
    _ARCHIVE_RUNTIME_CACHE.pop(str(path), None)


def _read_json_array(value: np.ndarray) -> dict[str, Any]:
    raw = value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _shape_payload(shape: tuple[int, int] | None) -> list[int] | None:
    if shape is None:
        return None
    return [int(shape[0]), int(shape[1])]


def _pack_mask_collection(masks: list[np.ndarray]) -> dict[str, np.ndarray]:
    count = len(masks)
    origins = np.zeros((count, 2), dtype=np.int32)
    shapes = np.zeros((count, 2), dtype=np.int32)
    full_shapes = np.zeros((count, 2), dtype=np.int32)
    bit_lengths = np.zeros((count,), dtype=np.int64)
    offsets = np.zeros((count + 1,), dtype=np.int64)
    chunks: list[np.ndarray] = []
    cursor = 0

    for index, raw_mask in enumerate(masks):
        mask = _cache_mask_array(raw_mask)
        if mask.ndim != 2:
            raise ValueError("Compact cache archives only support 2D binary masks.")

        full_shapes[index] = np.asarray(mask.shape[:2], dtype=np.int32)
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        y0 = int(ys.min())
        x0 = int(xs.min())
        y1 = int(ys.max()) + 1
        x1 = int(xs.max()) + 1
        crop = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.uint8)
        packed = np.packbits(crop.reshape(-1), bitorder="little")

        origins[index] = np.asarray([y0, x0], dtype=np.int32)
        shapes[index] = np.asarray(crop.shape[:2], dtype=np.int32)
        bit_lengths[index] = int(crop.size)
        cursor += int(packed.size)
        offsets[index + 1] = cursor
        chunks.append(packed.astype(np.uint8, copy=False))

    data = np.concatenate(chunks).astype(np.uint8, copy=False) if chunks else np.empty((0,), dtype=np.uint8)
    return {
        "origins": origins,
        "shapes": shapes,
        "full_shapes": full_shapes,
        "bit_lengths": bit_lengths,
        "offsets": offsets,
        "data": data,
    }


def _archive_payload(prefix: str, masks: list[np.ndarray]) -> dict[str, np.ndarray]:
    packed = _pack_mask_collection(masks)
    return {
        f"{prefix}_origins": packed["origins"],
        f"{prefix}_shapes": packed["shapes"],
        f"{prefix}_full_shapes": packed["full_shapes"],
        f"{prefix}_bit_lengths": packed["bit_lengths"],
        f"{prefix}_offsets": packed["offsets"],
        f"{prefix}_data": packed["data"],
    }


def _load_archive_record(path: str | Path) -> "_ArchiveRecord":
    key = str(path)
    cached = _ARCHIVE_RUNTIME_CACHE.get(key)
    if cached is not None:
        _ARCHIVE_RUNTIME_CACHE.move_to_end(key)
        return cached

    with np.load(key, allow_pickle=False) as archive:
        metadata = _read_json_array(archive["metadata_json"])
        arrays = {name: np.asarray(archive[name]) for name in archive.files if name != "metadata_json"}
    record = _ArchiveRecord(metadata=metadata, arrays=arrays)
    _ARCHIVE_RUNTIME_CACHE[key] = record
    while len(_ARCHIVE_RUNTIME_CACHE) > _ARCHIVE_CACHE_LIMIT:
        _ARCHIVE_RUNTIME_CACHE.popitem(last=False)
    return record


@dataclass(slots=True)
class _ArchiveRecord:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    decoded_masks: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    decoded_stacks: dict[str, np.ndarray] = field(default_factory=dict)

    def count(self, prefix: str) -> int:
        return int(self.arrays.get(f"{prefix}_bit_lengths", np.empty((0,), dtype=np.int64)).shape[0])

    def mask(self, prefix: str, index: int) -> np.ndarray:
        key = (prefix, int(index))
        if key in self.decoded_masks:
            return self.decoded_masks[key]

        full_shape = tuple(int(value) for value in self.arrays[f"{prefix}_full_shapes"][index].tolist())
        mask = np.zeros(full_shape, dtype=bool)
        bit_length = int(self.arrays[f"{prefix}_bit_lengths"][index])
        crop_shape = tuple(int(value) for value in self.arrays[f"{prefix}_shapes"][index].tolist())
        if bit_length > 0 and crop_shape[0] > 0 and crop_shape[1] > 0:
            offsets = self.arrays[f"{prefix}_offsets"]
            start = int(offsets[index])
            end = int(offsets[index + 1])
            packed = self.arrays[f"{prefix}_data"][start:end]
            crop = np.unpackbits(packed, count=bit_length, bitorder="little").astype(bool, copy=False)
            crop = crop.reshape(crop_shape)
            origin_y, origin_x = (int(value) for value in self.arrays[f"{prefix}_origins"][index].tolist())
            mask[origin_y : origin_y + crop_shape[0], origin_x : origin_x + crop_shape[1]] = crop

        self.decoded_masks[key] = mask
        return mask

    def stack(self, prefix: str) -> np.ndarray:
        if prefix in self.decoded_stacks:
            return self.decoded_stacks[prefix]
        count = self.count(prefix)
        if count == 0:
            stack = np.zeros((0, 0, 0), dtype=bool)
        elif count == 1:
            stack = self.mask(prefix, 0)
        else:
            stack = np.stack([self.mask(prefix, index) for index in range(count)], axis=0)
        self.decoded_stacks[prefix] = stack
        return stack


def _archive_mask_array(mask_ref: str | Path, *, prefix: str = "mask", index: int | None = None) -> np.ndarray:
    record = _load_archive_record(mask_ref)
    if index is None:
        return record.stack(prefix)
    return record.mask(prefix, index)


def load_cached_mask(mask_ref: object) -> np.ndarray | None:
    """Resolve cached mask references from either legacy or compact cache paths."""
    if mask_ref is None:
        return None
    if isinstance(mask_ref, (DiskMaskArray, ArchiveMaskArray)):
        return np.asarray(mask_ref)
    if isinstance(mask_ref, np.ndarray):
        return np.asarray(mask_ref)
    if isinstance(mask_ref, (str, Path)):
        path = Path(mask_ref)
        if not path.exists():
            raise FileNotFoundError(f"Cached mask does not exist: {path}")
        if path.suffix.lower() == ".npy":
            return np.load(path)
        if path.suffix.lower() == ".npz":
            record = _load_archive_record(path)
            if record.metadata.get("type") != "mask":
                raise ValueError(f"Unsupported cached archive type for mask loading: {path}")
            return record.stack("mask")
    return np.asarray(mask_ref)


def load_cached_result(result_ref: str | Path) -> PredictionResult:
    """Resolve cached result references from compact archive paths."""
    path = Path(result_ref)
    if not path.exists():
        raise FileNotFoundError(f"Cached result does not exist: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported cached result path: {path}")
    record = _load_archive_record(path)
    if record.metadata.get("type") != "result":
        raise ValueError(f"Unsupported cached archive type for result loading: {path}")
    metadata = record.metadata
    objects: list[SegmentationObject] = []
    for index, item in enumerate(metadata.get("objects", [])):
        box = item.get("box")
        objects.append(
            SegmentationObject(
                mask=ArchiveMaskArray(path, "object", index=index),
                box=None if box is None else tuple(float(value) for value in box),
                score=None if item.get("score") is None else float(item.get("score")),
                label=item.get("label"),
                track_id=None if item.get("track_id") is None else int(item.get("track_id")),
                object_index=int(item.get("index", index + 1)),
            )
        )
    prompt_mask = None
    if int(metadata.get("prompt_mask_count", 0)) > 0:
        prompt_mask = ArchiveMaskArray(path, "prompt")
    image_size = metadata.get("image_size")
    inference_image_size = metadata.get("inference_image_size")
    return PredictionResult(
        source=metadata.get("source"),
        frame_index=metadata.get("frame_index"),
        mode=str(metadata.get("mode") or "image"),
        image_size=None if image_size is None else (int(image_size[0]), int(image_size[1])),
        inference_image_size=None if inference_image_size is None else (int(inference_image_size[0]), int(inference_image_size[1])),
        objects=objects,
        prompt_metadata=dict(metadata.get("prompt_metadata") or {}),
        tracking_metadata=dict(metadata.get("tracking_metadata") or {}),
        timings={str(key): float(value) for key, value in dict(metadata.get("timings") or {}).items()},
        image=None,
        prompt_mask=prompt_mask,
    )


class DiskMaskArray:
    """Lazy disk-backed mask wrapper for legacy .npy masks."""

    def __init__(self, path: str | Path, index: int | None = None) -> None:
        self.path = str(path)
        self.index = index

    def _memmap(self):
        array = np.load(self.path, mmap_mode="r")
        if self.index is None:
            return array
        return array[self.index]

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
        if self.index is None:
            return f"DiskMaskArray(path={self.path!r})"
        return f"DiskMaskArray(path={self.path!r}, index={self.index!r})"


class ArchiveMaskArray:
    """Lazy compact archive-backed mask wrapper."""

    def __init__(self, path: str | Path, prefix: str, index: int | None = None) -> None:
        self.path = str(path)
        self.prefix = prefix
        self.index = index

    def __array__(self, dtype=None):
        array = _archive_mask_array(self.path, prefix=self.prefix, index=self.index)
        if dtype is not None:
            return array.astype(dtype)
        return array

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        return np.asarray(self).astype(dtype, order=order, casting=casting, subok=subok, copy=copy)

    def copy(self):
        return np.asarray(self).copy()

    @property
    def shape(self):
        return np.asarray(self).shape

    @property
    def dtype(self):
        return np.asarray(self).dtype

    def __repr__(self) -> str:
        return f"ArchiveMaskArray(path={self.path!r}, prefix={self.prefix!r}, index={self.index!r})"


@dataclass(slots=True)
class CacheStore:
    """Manage a writable cache directory for GUI state."""

    root: Path
    @classmethod
    def create(cls, root: str | Path) -> "CacheStore":
        store = cls(Path(root))
        store.ensure_ready()
        return store

    @property
    def data_root(self) -> Path:
        return self.root / f"v{ARCHIVE_VERSION}"

    def ensure_ready(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_dir.mkdir(parents=True, exist_ok=True)
        return self.root

    @property
    def mask_dir(self) -> Path:
        return self.data_root / "masks"

    @property
    def result_dir(self) -> Path:
        return self.data_root / "results"

    @property
    def yolo_dir(self) -> Path:
        return self.data_root / "yolo"

    def set_root(self, root: str | Path) -> None:
        self.root = Path(root)
        clear_archive_runtime_cache()
        self.ensure_ready()

    def clear(self) -> None:
        if self.data_root.exists():
            shutil.rmtree(self.data_root)
        clear_archive_runtime_cache()
        self.ensure_ready()

    def _safe_token(self, *parts: object) -> str:
        joined = "||".join("" if part is None else str(part) for part in parts)
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
        return digest

    def _write_mask_archive(self, target: Path, array: np.ndarray) -> str:
        _invalidate_archive_runtime_cache(target)
        masks = np.asarray(array)
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [masks[index] for index in range(masks.shape[0])]
        else:
            raise ValueError("Compact cached masks must be 2D or NxHxW.")

        payload = {
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "version": ARCHIVE_VERSION,
                        "type": "mask",
                        "mask_count": len(mask_list),
                    },
                    separators=(",", ":"),
                )
            ),
            **_archive_payload("mask", mask_list),
        }
        np.savez_compressed(target, **payload)
        _invalidate_archive_runtime_cache(target)
        return str(target)

    def _write_result_archive(self, target: Path, result: PredictionResult) -> PredictionResult:
        _invalidate_archive_runtime_cache(target)
        object_masks = [np.asarray(obj.mask) for obj in result.objects]
        prompt_masks = None if result.prompt_mask is None else np.asarray(result.prompt_mask)
        metadata = {
            "version": ARCHIVE_VERSION,
            "type": "result",
            "source": result.source,
            "frame_index": result.frame_index,
            "mode": result.mode,
            "image_size": _shape_payload(result.image_size),
            "inference_image_size": _shape_payload(result.inference_image_size),
            "prompt_metadata": dict(result.prompt_metadata),
            "tracking_metadata": dict(result.tracking_metadata),
            "timings": dict(result.timings),
            "objects": [
                {
                    "index": int(obj.object_index),
                    "box": None if obj.box is None else [float(value) for value in obj.box],
                    "score": None if obj.score is None else float(obj.score),
                    "label": obj.label,
                    "track_id": None if obj.track_id is None else int(obj.track_id),
                }
                for obj in result.objects
            ],
            "prompt_mask_count": 0 if prompt_masks is None else (1 if prompt_masks.ndim == 2 else int(prompt_masks.shape[0])),
        }
        payload: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(json.dumps(metadata, separators=(",", ":"))),
            **_archive_payload("object", object_masks),
        }
        if prompt_masks is not None:
            if prompt_masks.ndim == 2:
                prompt_list = [prompt_masks]
            elif prompt_masks.ndim == 3:
                prompt_list = [prompt_masks[index] for index in range(prompt_masks.shape[0])]
            else:
                raise ValueError("Prompt masks must be 2D or NxHxW for compact caching.")
            payload.update(_archive_payload("prompt", prompt_list))
        np.savez_compressed(target, **payload)
        _invalidate_archive_runtime_cache(target)

        cached_objects: list[SegmentationObject] = []
        for index, obj in enumerate(result.objects):
            cached_objects.append(
                SegmentationObject(
                    mask=ArchiveMaskArray(target, "object", index=index),
                    box=obj.box,
                    score=obj.score,
                    label=obj.label,
                    track_id=obj.track_id,
                    object_index=obj.object_index,
                )
            )

        prompt_mask = None
        if result.prompt_mask is not None:
            prompt_mask = ArchiveMaskArray(target, "prompt")

        image = None
        if result.image is not None:
            # Keep source images only when the GUI cannot cheaply rehydrate them
            # from disk. This preserves preview correctness while pushing large
            # mask payloads and most frame data out of process memory.
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
            inference_image_size=result.inference_image_size,
            objects=cached_objects,
            prompt_metadata=dict(result.prompt_metadata),
            tracking_metadata=dict(result.tracking_metadata),
            timings=dict(result.timings),
            image=image,
            prompt_mask=prompt_mask,
        )

    def write_mask(self, namespace: str, key: str, array: np.ndarray) -> str:
        target_dir = self.mask_dir / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        token = self._safe_token(key)
        target = target_dir / f"{token}.npz"
        return self._write_mask_archive(target, array)

    def write_result(self, namespace: str, key: str, result: PredictionResult) -> PredictionResult:
        target_dir = self.result_dir / namespace
        target_dir.mkdir(parents=True, exist_ok=True)
        token = self._safe_token(key)
        target = target_dir / f"{token}.npz"
        return self._write_result_archive(target, result)
