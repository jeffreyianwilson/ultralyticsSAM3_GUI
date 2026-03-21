"""I/O helpers for images, videos, masks, and export directories."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .exceptions import ExportError, InvalidSourceError
from .schemas import ImageSource, MaskSource

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def is_video_path(source: object) -> bool:
    """Return whether the given source appears to be a video path."""
    if not isinstance(source, (str, Path)):
        return False
    return Path(source).suffix.lower() in VIDEO_EXTENSIONS


def is_image_path(source: object) -> bool:
    """Return whether the given source appears to be an image path."""
    if not isinstance(source, (str, Path)):
        return False
    return Path(source).suffix.lower() in IMAGE_EXTENSIONS


def source_label(source: ImageSource | str | Path) -> str:
    """Return a stable label for a source."""
    if isinstance(source, Path):
        return str(source)
    if isinstance(source, str):
        return source
    if isinstance(source, Image.Image):
        return "pil_image"
    if isinstance(source, np.ndarray):
        return "numpy_image"
    return "source"


def source_stem(source: ImageSource | str | Path) -> str:
    """Return a stable stem used for export names."""
    if isinstance(source, (str, Path)):
        return Path(source).stem or "source"
    return source_label(source)


def ensure_directory(path: str | Path | None, *, create: bool = True) -> Path | None:
    """Validate a directory path."""
    if path is None:
        return None
    resolved = Path(path)
    if create:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExportError(f"Could not create directory: {resolved}") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise ExportError(f"Directory does not exist: {resolved}")
    return resolved


def ensure_writable_directory(path: str | Path, *, create: bool = True) -> Path:
    """Ensure a directory exists and is writable."""
    resolved = ensure_directory(path, create=create)
    if resolved is None:
        raise ExportError("A writable directory path is required.")
    probe = resolved / ".sam3_ultralytics_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ExportError(f"Directory is not writable: {resolved}") from exc
    return resolved


def to_bgr_image(source: ImageSource) -> np.ndarray:
    """Convert a supported image source to a BGR uint8 image."""
    if isinstance(source, np.ndarray):
        image = np.asarray(source)
        if image.ndim == 2:
            image = cv2.cvtColor(_normalize_gray_image(image), cv2.COLOR_GRAY2BGR)
        elif image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise InvalidSourceError("NumPy image sources must have shape HxW, HxWx3, or HxWx4.")
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return np.asarray(image).copy()
    if isinstance(source, Image.Image):
        rgb = np.asarray(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path = Path(source)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidSourceError(f"Could not read image: {path}")
    return image


def _normalize_gray_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
        return array.astype(np.uint8)
    return np.clip(array, 0, 255).astype(np.uint8)


def _mask_array_from_source(mask_input: MaskSource) -> tuple[np.ndarray, str]:
    if isinstance(mask_input, Image.Image):
        return np.asarray(mask_input.convert("L")), "pil_image"
    if isinstance(mask_input, (str, Path)):
        path = Path(mask_input)
        if not path.exists():
            raise InvalidSourceError(f"Mask source does not exist: {path}")
        if path.suffix.lower() == ".npy":
            return np.load(path), str(path)
        if path.suffix.lower() == ".npz":
            with np.load(path) as archive:
                if "metadata_json" in archive.files:
                    from .cache_store import load_cached_mask

                    return np.asarray(load_cached_mask(path)), str(path)
                first_key = next(iter(archive.files), None)
                if first_key is None:
                    raise InvalidSourceError(f"Mask archive is empty: {path}")
                return np.asarray(archive[first_key]), str(path)
        with Image.open(path) as image:
            return np.asarray(image.convert("L")), str(path)
    array = np.asarray(mask_input)
    return array, "numpy_array"


def normalize_mask_input(mask_input: MaskSource | None) -> tuple[np.ndarray | None, dict[str, object]]:
    """Normalize a mask source into a model-friendly float32 array in [0, 1]."""
    if mask_input is None:
        return None, {}

    array, source = _mask_array_from_source(mask_input)
    original_dtype = str(array.dtype)

    if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] == 4:
            array = cv2.cvtColor(_normalize_gray_image(array), cv2.COLOR_BGRA2GRAY)
        else:
            array = cv2.cvtColor(_normalize_gray_image(array), cv2.COLOR_BGR2GRAY)

    if array.ndim == 2:
        mask_stack = array[None, ...]
    elif array.ndim == 3:
        mask_stack = array
    else:
        raise InvalidSourceError("Mask inputs must be HxW, HxWxC image-like data, or NxHxW mask stacks.")

    mask_stack = np.asarray(mask_stack)
    if mask_stack.dtype == np.bool_:
        normalized = mask_stack.astype(np.float32)
    else:
        normalized = mask_stack.astype(np.float32)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        max_value = float(normalized.max()) if normalized.size else 0.0
        min_value = float(normalized.min()) if normalized.size else 0.0
        if max_value > 1.0:
            if max_value <= 255.0 and min_value >= 0.0:
                normalized = normalized / 255.0
            else:
                scale = max(max_value - min_value, 1.0)
                normalized = (normalized - min_value) / scale
        normalized = np.clip(normalized, 0.0, 1.0)

    collapsed = normalized[0] if normalized.shape[0] == 1 else normalized
    values = np.unique((normalized > 0.5).astype(np.uint8)) if normalized.size else np.array([], dtype=np.uint8)
    kind = "binary" if set(values.tolist()).issubset({0, 1}) and np.allclose(normalized, normalized.round()) else "probability"
    metadata = {
        "source": source,
        "shape": list(collapsed.shape),
        "original_dtype": original_dtype,
        "normalized_dtype": "float32",
        "mask_count": int(normalized.shape[0]),
        "kind": kind,
        "min": float(normalized.min()) if normalized.size else 0.0,
        "max": float(normalized.max()) if normalized.size else 0.0,
    }
    return collapsed.astype(np.float32, copy=False), metadata


def preview_mask(mask_input: np.ndarray | None) -> np.ndarray | None:
    """Collapse a normalized mask prompt into a single preview mask."""
    if mask_input is None:
        return None
    array = np.asarray(mask_input, dtype=np.float32)
    if array.ndim == 2:
        return np.clip(array, 0.0, 1.0)
    if array.ndim == 3:
        return np.clip(array.max(axis=0), 0.0, 1.0)
    raise InvalidSourceError("Preview masks must be a 2D mask or 3D stack of masks.")


def list_image_directory(path: str | Path) -> list[Path]:
    """Return a stable, sorted list of supported image files in a directory."""
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        raise InvalidSourceError(f"Directory does not exist: {directory}")
    files = sorted(item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise InvalidSourceError(f"No supported images found in directory: {directory}")
    return files


def expand_sources(sources) -> list:
    """Expand directory sources into image paths while leaving image arrays and videos untouched."""
    expanded: list = []
    for source in sources:
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            expanded.extend(list_image_directory(source))
        else:
            expanded.append(source)
    return expanded


def read_video_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read a single frame from a video path."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise InvalidSourceError(f"Could not read frame {frame_index} from video: {video_path}")
    return frame


def video_frame_count(video_path: str | Path) -> int | None:
    """Return the frame count for a video when available."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    return frames or None


def configure_yolo_environment(config_dir: str | Path | None) -> Path:
    """Configure a writable YOLO settings directory."""
    target = Path(config_dir or Path.cwd())
    target.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(target)
    return target
