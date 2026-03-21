"""Result export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .exceptions import ExportError
from .io_utils import ensure_writable_directory, normalize_mask_input, read_video_frame, source_stem
from .schemas import PredictionResult
from .visualization import merged_mask, render_overlay


def image_mask_filename(source_name: str, object_index: int, track_id: int | None = None) -> str:
    """Return a deterministic image mask filename."""
    if track_id is not None:
        return f"{source_name}_track_{track_id:03d}.png"
    return f"{source_name}_mask_{object_index:03d}.png"


def image_merged_mask_filename(source_name: str) -> str:
    """Return a deterministic merged image mask filename."""
    return f"{source_name}_merged_mask.png"


def video_mask_filename(frame_index: int, object_index: int, track_id: int | None = None) -> str:
    """Return a deterministic video mask filename."""
    if track_id is not None:
        return f"frame_{frame_index + 1:06d}_track_{track_id:03d}.png"
    return f"frame_{frame_index + 1:06d}_obj_{object_index:03d}.png"


def video_merged_mask_filename(frame_index: int) -> str:
    """Return a deterministic merged video mask filename."""
    return f"frame_{frame_index + 1:06d}_merged_mask.png"


def _write_png(path: Path, image: np.ndarray, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ExportError(f"Failed to write image: {path}")


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dilate_mask_image(image: np.ndarray, dilation_pixels: int) -> np.ndarray:
    if dilation_pixels <= 0:
        return image
    binary = (np.asarray(image) > 0).astype(np.uint8)
    kernel_size = max(1, dilation_pixels * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return (dilated * 255).astype(np.uint8)


def _mask_to_png(mask: np.ndarray, *, invert_mask: bool, dilation_pixels: int = 0) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype == np.bool_:
        image = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        image = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        max_value = float(image.max()) if image.size else 0.0
        if max_value <= 1.0:
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(array, 0, 255).astype(np.uint8)
    if dilation_pixels > 0:
        image = _dilate_mask_image(image, dilation_pixels)
    if invert_mask:
        image = 255 - image
    return image


def _result_manual_mask(result: PredictionResult, manual_masks_by_key: dict[str, object] | None) -> np.ndarray | None:
    if not manual_masks_by_key:
        return None
    if result.frame_index is not None and result.source:
        key = f"{result.source}::frame:{int(result.frame_index)}"
        if key in manual_masks_by_key:
            return normalize_mask_input(manual_masks_by_key[key])[0]
    if result.source and result.source in manual_masks_by_key:
        return normalize_mask_input(manual_masks_by_key[result.source])[0]
    return None


def _export_image_masks(
    result: PredictionResult,
    *,
    base_name: str,
    mask_dir: Path | None,
    overwrite: bool,
    save_merged_mask: bool,
    merged_mask_only: bool,
    invert_mask: bool,
    manual_mask: np.ndarray | None,
    dilation_pixels: int,
) -> tuple[list[str], dict[str, Any]]:
    export_paths: dict[str, Any] = {"masks": []}
    mask_paths: list[str] = []
    if mask_dir is None:
        return mask_paths, export_paths

    if merged_mask_only:
        merged = merged_mask(result, extra_masks=[manual_mask] if manual_mask is not None else None)
        if merged is not None:
            merged_path = mask_dir / image_merged_mask_filename(base_name)
            _write_png(merged_path, _mask_to_png(merged, invert_mask=invert_mask, dilation_pixels=dilation_pixels), overwrite=overwrite)
            merged_path_str = str(merged_path)
            export_paths["masks"].append(merged_path_str)
            export_paths["merged_mask"] = merged_path_str
            mask_paths = [merged_path_str] * len(result.objects)
        return mask_paths, export_paths

    for obj in result.objects:
        mask_path = mask_dir / image_mask_filename(base_name, obj.object_index, obj.track_id)
        _write_png(mask_path, _mask_to_png(obj.mask, invert_mask=invert_mask, dilation_pixels=dilation_pixels), overwrite=overwrite)
        mask_path_str = str(mask_path)
        mask_paths.append(mask_path_str)
        export_paths["masks"].append(mask_path_str)

    if save_merged_mask:
        merged = merged_mask(result, extra_masks=[manual_mask] if manual_mask is not None else None)
        if merged is not None:
            merged_path = mask_dir / image_merged_mask_filename(base_name)
            _write_png(merged_path, _mask_to_png(merged, invert_mask=invert_mask, dilation_pixels=dilation_pixels), overwrite=overwrite)
            export_paths["merged_mask"] = str(merged_path)

    return mask_paths, export_paths


def _export_image_result(
    result: PredictionResult,
    *,
    output_dir: Path | None,
    mask_dir: Path | None,
    overwrite: bool,
    save_overlay: bool,
    save_json: bool,
    save_merged_mask: bool,
    save_cutout: bool,
    merged_mask_only: bool,
    invert_mask: bool,
    manual_masks_by_key: dict[str, object] | None,
    dilation_pixels: int,
    progress_callback=None,
    progress_index: int = 1,
    progress_total: int = 1,
) -> dict[str, Any]:
    base_name = source_stem(result.source or "image")
    output_dir = output_dir or mask_dir
    manual_mask = _result_manual_mask(result, manual_masks_by_key)
    mask_paths, export_paths = _export_image_masks(
        result,
        base_name=base_name,
        mask_dir=mask_dir,
        overwrite=overwrite,
        save_merged_mask=save_merged_mask,
        merged_mask_only=merged_mask_only,
        invert_mask=invert_mask,
        manual_mask=manual_mask,
        dilation_pixels=dilation_pixels,
    )

    if output_dir is not None and result.image is not None:
        if save_overlay:
            overlay_path = output_dir / f"{base_name}_overlay.png"
            overlay = render_overlay(result.image, result)
            _write_png(overlay_path, overlay, overwrite=overwrite)
            export_paths["overlay"] = str(overlay_path)
        if save_cutout:
            union = merged_mask(result, extra_masks=[manual_mask] if manual_mask is not None else None)
            if union is not None:
                rgba = cv2.cvtColor(result.image, cv2.COLOR_BGR2BGRA)
                rgba[:, :, 3] = _mask_to_png(union, invert_mask=False, dilation_pixels=dilation_pixels).astype(np.uint8)
                cutout_path = output_dir / f"{base_name}_cutout.png"
                _write_png(cutout_path, rgba, overwrite=overwrite)
                export_paths["cutout"] = str(cutout_path)

    if output_dir is not None and save_json:
        json_path = output_dir / f"{base_name}_results.json"
        _write_json(json_path, result.to_dict(mask_paths=mask_paths), overwrite=overwrite)
        export_paths["json"] = str(json_path)

    if progress_callback is not None:
        progress_callback(progress_index, progress_total, f"Exported {base_name}")
    return export_paths


def _export_image_batch_results(
    results: list[PredictionResult],
    *,
    output_dir: Path | None,
    mask_dir: Path | None,
    overwrite: bool,
    save_overlay: bool,
    save_json: bool,
    save_merged_mask: bool,
    save_cutout: bool,
    merged_mask_only: bool,
    invert_mask: bool,
    manual_masks_by_key: dict[str, object] | None,
    dilation_pixels: int,
    progress_callback=None,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"items": [], "masks": []}
    total = len(results)
    for index, result in enumerate(results, start=1):
        item_paths = _export_image_result(
            result,
            output_dir=output_dir,
            mask_dir=mask_dir,
            overwrite=overwrite,
            save_overlay=save_overlay,
            save_json=save_json,
            save_merged_mask=save_merged_mask,
            save_cutout=save_cutout,
            merged_mask_only=merged_mask_only,
            invert_mask=invert_mask,
            manual_masks_by_key=manual_masks_by_key,
            dilation_pixels=dilation_pixels,
            progress_callback=progress_callback,
            progress_index=index,
            progress_total=total,
        )
        aggregate["items"].append(item_paths)
        for key, value in item_paths.items():
            if key == "masks":
                aggregate["masks"].extend(value)
                continue
            if isinstance(value, list):
                aggregate.setdefault(key, []).extend(value)
            else:
                aggregate.setdefault(key, []).append(value)
    return aggregate


def _export_video_results(
    results: list[PredictionResult],
    *,
    output_dir: Path | None,
    mask_dir: Path | None,
    annotated_video_path: str | Path | None,
    overwrite: bool,
    save_overlay: bool,
    save_json: bool,
    merged_mask_only: bool,
    invert_mask: bool,
    manual_masks_by_key: dict[str, object] | None,
    dilation_pixels: int,
    progress_callback=None,
) -> dict[str, Any]:
    if not results:
        return {}
    source = results[0].source
    source_name = source_stem(source or "video")
    base_dir = output_dir or mask_dir
    if base_dir is None and annotated_video_path is None:
        raise ExportError("Video export requires output_dir, mask_dir, or annotated_video_path.")

    export_paths: dict[str, Any] = {"frames": [], "masks": [], "json": []}
    frames_dir = base_dir / "frames" if base_dir is not None else None
    masks_dir = mask_dir or (base_dir / "masks" if base_dir is not None else None)
    json_dir = base_dir / "json" if base_dir is not None else None

    if annotated_video_path is not None:
        annotated_path = Path(annotated_video_path)
    elif base_dir is not None:
        annotated_path = base_dir / "annotated_video.mp4"
    else:
        annotated_path = None

    writer = None
    if annotated_path is not None:
        first_frame = read_video_frame(source, 0)
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            24.0,
            (width, height),
        )
        export_paths["annotated_video"] = str(annotated_path)

    try:
        total = len(results)
        for index, result in enumerate(results, start=1):
            assert result.frame_index is not None
            frame = read_video_frame(source, result.frame_index)
            mask_paths: list[str] = []
            if masks_dir is not None:
                manual_mask = _result_manual_mask(result, manual_masks_by_key)
                if merged_mask_only:
                    merged = merged_mask(result, extra_masks=[manual_mask] if manual_mask is not None else None)
                    if merged is not None:
                        mask_path = masks_dir / video_merged_mask_filename(result.frame_index)
                        _write_png(mask_path, _mask_to_png(merged, invert_mask=invert_mask, dilation_pixels=dilation_pixels), overwrite=overwrite)
                        mask_path_str = str(mask_path)
                        mask_paths = [mask_path_str] * len(result.objects)
                        export_paths["masks"].append(mask_path_str)
                else:
                    for obj in result.objects:
                        mask_path = masks_dir / video_mask_filename(result.frame_index, obj.object_index, obj.track_id)
                        _write_png(mask_path, _mask_to_png(obj.mask, invert_mask=invert_mask, dilation_pixels=dilation_pixels), overwrite=overwrite)
                        mask_path_str = str(mask_path)
                        mask_paths.append(mask_path_str)
                        export_paths["masks"].append(mask_path_str)
            if frames_dir is not None and save_overlay:
                overlay = render_overlay(frame, result)
                frame_path = frames_dir / f"frame_{result.frame_index + 1:06d}_overlay.png"
                _write_png(frame_path, overlay, overwrite=overwrite)
                export_paths["frames"].append(str(frame_path))
            if writer is not None:
                writer.write(render_overlay(frame, result))
            if json_dir is not None and save_json:
                json_path = json_dir / f"frame_{result.frame_index + 1:06d}.json"
                _write_json(json_path, result.to_dict(mask_paths=mask_paths), overwrite=overwrite)
                export_paths["json"].append(str(json_path))
            if progress_callback is not None:
                progress_callback(index, total, f"Exported frame {result.frame_index + 1}")
    finally:
        if writer is not None:
            writer.release()

    if base_dir is not None and save_json:
        summary_path = base_dir / f"{source_name}_results.json"
        _write_json(summary_path, [item.to_dict() for item in results], overwrite=overwrite)
        export_paths["summary_json"] = str(summary_path)
    return export_paths


def save_results(
    results: PredictionResult | list[PredictionResult],
    *,
    output_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    annotated_video_path: str | Path | None = None,
    overwrite: bool = True,
    save_overlay: bool = True,
    save_json: bool = True,
    save_merged_mask: bool = True,
    save_cutout: bool = False,
    merged_mask_only: bool = False,
    invert_mask: bool = False,
    manual_masks_by_key: dict[str, object] | None = None,
    dilation_pixels: int = 0,
    progress_callback=None,
) -> dict[str, Any]:
    """Export normalized results to PNG, JSON, and video outputs."""
    resolved_output_dir = ensure_writable_directory(output_dir, create=True) if output_dir is not None else None
    resolved_mask_dir = ensure_writable_directory(mask_dir, create=True) if mask_dir is not None else None

    if isinstance(results, PredictionResult):
        return _export_image_result(
            results,
            output_dir=resolved_output_dir,
            mask_dir=resolved_mask_dir,
            overwrite=overwrite,
            save_overlay=save_overlay,
            save_json=save_json,
            save_merged_mask=save_merged_mask,
            save_cutout=save_cutout,
            merged_mask_only=merged_mask_only,
            invert_mask=invert_mask,
            manual_masks_by_key=manual_masks_by_key,
            dilation_pixels=dilation_pixels,
            progress_callback=progress_callback,
        )
    if not results:
        return {}
    if all(item.mode == "image" for item in results):
        return _export_image_batch_results(
            results,
            output_dir=resolved_output_dir,
            mask_dir=resolved_mask_dir,
            overwrite=overwrite,
            save_overlay=save_overlay,
            save_json=save_json,
            save_merged_mask=save_merged_mask,
            save_cutout=save_cutout,
            merged_mask_only=merged_mask_only,
            invert_mask=invert_mask,
            manual_masks_by_key=manual_masks_by_key,
            dilation_pixels=dilation_pixels,
            progress_callback=progress_callback,
        )
    return _export_video_results(
        results,
        output_dir=resolved_output_dir,
        mask_dir=resolved_mask_dir,
        annotated_video_path=annotated_video_path,
        overwrite=overwrite,
        save_overlay=save_overlay,
        save_json=save_json,
        merged_mask_only=merged_mask_only,
        invert_mask=invert_mask,
        manual_masks_by_key=manual_masks_by_key,
        dilation_pixels=dilation_pixels,
        progress_callback=progress_callback,
    )


