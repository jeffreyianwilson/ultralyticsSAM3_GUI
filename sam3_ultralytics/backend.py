"""High-level backend API for sam3_ultralytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .exceptions import InferenceCancelledError, ModelNotLoadedError
from .export import save_results as export_results
from .inference_scaling import apply_inference_transform, normalize_inference_scale, prepare_inference_source
from .image_inference import run_image_prediction
from .io_utils import expand_sources, is_video_path, read_video_frame, video_frame_count
from .model_loading import ModelLoader
from .prompt_handling import build_prompt_payload, validate_prompt_payload
from .schemas import PredictionResult
from .tracking import _assign_tracking_identity
from .tracking import track_image_sequence as compatibility_track_image_sequence
from .tracking import track_video_frames as compatibility_track_video_frames
from .tracking import track_video_sequence


class ExemplarAdapterProtocol(Protocol):
    """Protocol for optional exemplar-prompt adapters."""

    def predict_image(self, *, model_loader, source, payload): ...

    def track_video(self, *, model_loader, source, payload, progress_callback=None, cancel_callback=None): ...


class SAM3Ultralytics:
    """API-first SAM 3 wrapper around Ultralytics predictors."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str = "auto",
        half: bool | None = None,
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
        yolo_config_dir: str | Path | None = None,
        model_loader: ModelLoader | None = None,
        exemplar_adapter: ExemplarAdapterProtocol | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.yolo_config_dir = yolo_config_dir
        self.exemplar_adapter = exemplar_adapter
        self._model_loader = model_loader

    @property
    def model_loader(self) -> ModelLoader:
        """Return the active model loader."""
        if self._model_loader is None:
            raise ModelNotLoadedError("Call load() with a valid SAM 3 checkpoint before running inference.")
        return self._model_loader

    def load(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        half: bool | None = None,
    ) -> "SAM3Ultralytics":
        """Load or reload the backend checkpoint and runtime settings."""
        if model_path is not None:
            self.model_path = Path(model_path)
        if self.model_path is None:
            raise ModelNotLoadedError("A SAM 3 checkpoint path is required.")
        if device is not None:
            self.device = device
        if half is not None:
            self.half = half

        # ModelLoader owns the Ultralytics predictor lifecycle. Rebuilding it is
        # the expensive step, so the GUI tries hard to reuse this backend.
        self._model_loader = ModelLoader(
            self.model_path,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            yolo_config_dir=self.yolo_config_dir,
        ).load()
        return self

    @staticmethod
    def _first_object_mask(result: PredictionResult) -> np.ndarray | None:
        if not result.objects:
            return None
        return np.asarray(result.objects[0].mask, dtype=np.float32).copy()

    @staticmethod
    def _mask_metadata(mask_id: int | None = None, mask_label: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = dict(extra or {})
        if mask_id is not None:
            metadata["id"] = int(mask_id)
        if mask_label:
            metadata["class"] = str(mask_label)
        return metadata

    def predict_image(
        self,
        source,
        *,
        text_prompt=None,
        exemplar_image=None,
        exemplar_box=None,
        points=None,
        boxes=None,
        mask_input=None,
        mask_id: int | None = None,
        mask_label: str | None = None,
        mask_metadata: dict[str, Any] | None = None,
        inference_scale: float = 1.0,
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        cancel_callback=None,
    ) -> PredictionResult:
        """Run image segmentation against a single image-like input."""
        if cancel_callback is not None and cancel_callback():
            raise InferenceCancelledError("Image inference was cancelled.")
        resolved_scale = normalize_inference_scale(inference_scale)
        inference_source, scaled_points, scaled_boxes, scaled_mask_input, transform = prepare_inference_source(
            source,
            points=points,
            boxes=boxes,
            mask_input=mask_input,
            inference_scale=resolved_scale,
        )
        payload = build_prompt_payload(
            text_prompt=text_prompt,
            points=scaled_points,
            boxes=scaled_boxes,
            mask_input=scaled_mask_input,
            exemplar_image=exemplar_image,
            exemplar_box=exemplar_box,
            mask_metadata=self._mask_metadata(mask_id, mask_label, mask_metadata),
        )
        validate_prompt_payload(payload, is_video=False)
        result = run_image_prediction(self.model_loader, inference_source, payload, exemplar_adapter=self.exemplar_adapter)
        if cancel_callback is not None and cancel_callback():
            raise InferenceCancelledError("Image inference was cancelled.")
        result = apply_inference_transform(result, transform, source=source)
        if export_mask_dir is not None or output_dir is not None:
            export_results(
                result,
                output_dir=output_dir,
                mask_dir=export_mask_dir,
                overwrite=overwrite,
                save_overlay=output_dir is not None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                preserve_source_filenames=preserve_source_filenames,
            )
        return result

    def track_image_sequence(
        self,
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
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
    ) -> list[PredictionResult]:
        """Track a mask-initialized object across an image sequence."""
        results = compatibility_track_image_sequence(
            self.model_loader,
            sources,
            text_prompt=text_prompt,
            exemplar_image=exemplar_image,
            exemplar_box=exemplar_box,
            points=points,
            boxes=boxes,
            mask_input=mask_input,
            mask_inputs=mask_inputs,
            points_by_source=points_by_source,
            boxes_by_source=boxes_by_source,
            text_prompts_by_source=text_prompts_by_source,
            mask_id=mask_id,
            mask_label=mask_label,
            inference_scale=inference_scale,
            exemplar_adapter=self.exemplar_adapter,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        if export_mask_dir is not None or output_dir is not None:
            export_results(
                results,
                output_dir=output_dir,
                mask_dir=export_mask_dir,
                overwrite=overwrite,
                save_overlay=output_dir is not None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                preserve_source_filenames=preserve_source_filenames,
            )
        return results

    def predict_image_sequence(
        self,
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
        reuse_first_mask: bool = False,
        first_mask_initializer_only: bool = False,
        mask_id: int | None = None,
        mask_label: str | None = None,
        inference_scale: float = 1.0,
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
        item_start_callback=None,
        item_result_callback=None,
    ) -> list[PredictionResult]:
        """Run image prediction across a sequence of image-like inputs."""
        expanded_sources = expand_sources(sources)
        results: list[PredictionResult] = []
        first_result_mask: np.ndarray | None = None
        target_reference_mask = None if mask_input is None else np.asarray(mask_input, dtype=np.float32).copy()
        resolved_track_id = int(mask_id or 1) if (mask_input is not None or mask_id is not None or mask_label is not None) else None
        resolved_mask_label = str(mask_label).strip() or None if mask_label is not None else None
        total = len(expanded_sources)
        for index, source in enumerate(expanded_sources, start=1):
            if cancel_callback is not None and cancel_callback():
                raise InferenceCancelledError("Image sequence inference was cancelled.")

            source_key = str(source)
            if item_start_callback is not None:
                item_start_callback(index - 1, total, str(source_key))
            per_source_mask = None if mask_inputs is None else mask_inputs.get(source_key)
            current_points = points if points_by_source is None else points_by_source.get(source_key, points)
            current_boxes = boxes if boxes_by_source is None else boxes_by_source.get(source_key, boxes)
            current_text_prompt = text_prompt if text_prompts_by_source is None else text_prompts_by_source.get(source_key, text_prompt)

            if index == 1:
                current_mask = per_source_mask if per_source_mask is not None else mask_input
            elif per_source_mask is not None:
                current_mask = per_source_mask
            elif reuse_first_mask and first_result_mask is not None:
                current_mask = first_result_mask
            elif first_mask_initializer_only:
                current_mask = None
            else:
                current_mask = mask_input

            result = self.predict_image(
                source,
                text_prompt=current_text_prompt,
                exemplar_image=exemplar_image,
                exemplar_box=exemplar_box,
                points=current_points,
                boxes=current_boxes,
                mask_input=current_mask,
                mask_id=mask_id,
                mask_label=mask_label,
                inference_scale=inference_scale,
                preserve_source_filenames=preserve_source_filenames,
            )
            reference_mask = None
            if current_mask is not None:
                reference_mask = np.asarray(current_mask, dtype=np.float32).copy()
            elif target_reference_mask is not None:
                reference_mask = np.asarray(target_reference_mask, dtype=np.float32).copy()

            should_filter_target = (
                resolved_track_id is not None
                and reference_mask is not None
                and (reuse_first_mask or first_mask_initializer_only or mask_input is not None or mask_id is not None)
            )
            if should_filter_target:
                target_reference_mask, tracked_iou, _used_fallback = _assign_tracking_identity(
                    result,
                    reference_mask=reference_mask,
                    track_id=resolved_track_id,
                    mask_label=resolved_mask_label,
                    fallback_allowed=True,
                )
                if target_reference_mask is not None:
                    result.prompt_mask = np.asarray(target_reference_mask, dtype=np.float32).copy()
                result.tracking_metadata.update(
                    {
                        "tracking_mode": "image_sequence_prompt_target_filter",
                        "initial_mask_id": resolved_track_id,
                        "initial_mask_class": resolved_mask_label,
                        "frame_index": index - 1,
                        "source_key": source_key,
                        "tracked_mask_iou": float(tracked_iou),
                    }
                )
                result.prompt_metadata.setdefault("mask_input", {})
                if isinstance(result.prompt_metadata["mask_input"], dict):
                    result.prompt_metadata["mask_input"]["id"] = resolved_track_id
                    if resolved_mask_label:
                        result.prompt_metadata["mask_input"]["class"] = resolved_mask_label
            if (reuse_first_mask or first_mask_initializer_only) and first_result_mask is None:
                first_result_mask = self._first_object_mask(result)
            results.append(result)
            if item_result_callback is not None:
                item_result_callback(index - 1, total, result, str(source_key))
            if progress_callback is not None:
                progress_callback(index, total, f"Processed {Path(source).name if isinstance(source, (str, Path)) else 'image'}")

        if export_mask_dir is not None or output_dir is not None:
            export_results(
                results,
                output_dir=output_dir,
                mask_dir=export_mask_dir,
                overwrite=overwrite,
                save_overlay=output_dir is not None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                preserve_source_filenames=preserve_source_filenames,
            )
        return results

    def predict_video(
        self,
        source,
        *,
        text_prompt=None,
        exemplar_image=None,
        exemplar_box=None,
        points=None,
        boxes=None,
        mask_input=None,
        inference_scale: float = 1.0,
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        annotated_video_path: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
    ) -> list[PredictionResult]:
        """Run prompt-initialized video prediction."""
        return self.track_video(
            source,
            text_prompt=text_prompt,
            exemplar_image=exemplar_image,
            exemplar_box=exemplar_box,
            points=points,
            boxes=boxes,
            mask_input=mask_input,
            inference_scale=inference_scale,
            export_mask_dir=export_mask_dir,
            output_dir=output_dir,
            annotated_video_path=annotated_video_path,
            overwrite=overwrite,
            merged_mask_only=merged_mask_only,
            invert_mask=invert_mask,
            preserve_source_filenames=preserve_source_filenames,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    def predict_video_frames(
        self,
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
        reuse_first_mask: bool = False,
        first_mask_initializer_only: bool = False,
        mask_id: int | None = None,
        mask_label: str | None = None,
        inference_scale: float = 1.0,
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        annotated_video_path: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
        item_start_callback=None,
        item_result_callback=None,
    ) -> list[PredictionResult]:
        """Run image-style segmentation on selected video frames."""
        if frame_indices is None:
            total_frames = video_frame_count(source)
            if total_frames is None:
                raise ModelNotLoadedError("Could not determine video frame count for sequence prediction.")
            frame_indices = list(range(total_frames))
        results: list[PredictionResult] = []
        first_result_mask: np.ndarray | None = None
        total = len(frame_indices)
        for index, frame_index in enumerate(frame_indices, start=1):
            if cancel_callback is not None and cancel_callback():
                raise InferenceCancelledError("Video frame inference was cancelled.")
            if item_start_callback is not None:
                item_start_callback(index - 1, total, f"frame:{frame_index}")
            frame = read_video_frame(source, frame_index)

            per_frame_mask = None if mask_inputs_by_frame is None else mask_inputs_by_frame.get(frame_index)
            if index == 1:
                current_mask = per_frame_mask if per_frame_mask is not None else mask_input
            elif reuse_first_mask and first_result_mask is not None:
                current_mask = first_result_mask
            elif per_frame_mask is not None:
                current_mask = per_frame_mask
            elif first_mask_initializer_only:
                current_mask = None
            else:
                current_mask = mask_input

            result = self.predict_image(
                frame,
                text_prompt=text_prompt,
                exemplar_image=exemplar_image,
                exemplar_box=exemplar_box,
                points=points,
                boxes=boxes,
                mask_input=current_mask,
                mask_id=mask_id,
                mask_label=mask_label,
                inference_scale=inference_scale,
                preserve_source_filenames=preserve_source_filenames,
            )
            result.source = str(source)
            result.mode = "video"
            result.frame_index = frame_index
            result.image = frame
            if (reuse_first_mask or first_mask_initializer_only) and first_result_mask is None:
                first_result_mask = self._first_object_mask(result)
            results.append(result)
            if item_result_callback is not None:
                item_result_callback(index - 1, total, result, f"frame:{frame_index}")
            if progress_callback is not None:
                progress_callback(index, total, f"Processed frame {frame_index + 1}")

        if export_mask_dir is not None or output_dir is not None or annotated_video_path is not None:
            export_results(
                results,
                output_dir=output_dir,
                mask_dir=export_mask_dir,
                annotated_video_path=annotated_video_path,
                overwrite=overwrite,
                save_overlay=output_dir is not None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                preserve_source_filenames=preserve_source_filenames,
            )
        return results

    def track_video_frames(
        self,
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
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
    ) -> list[PredictionResult]:
        """Track a mask-initialized object across selected video frames."""
        results = compatibility_track_video_frames(
            self.model_loader,
            source,
            frame_indices=frame_indices,
            text_prompt=text_prompt,
            exemplar_image=exemplar_image,
            exemplar_box=exemplar_box,
            points=points,
            boxes=boxes,
            mask_input=mask_input,
            mask_inputs_by_frame=mask_inputs_by_frame,
            points_by_frame=points_by_frame,
            boxes_by_frame=boxes_by_frame,
            text_prompts_by_frame=text_prompts_by_frame,
            mask_id=mask_id,
            mask_label=mask_label,
            inference_scale=inference_scale,
            exemplar_adapter=self.exemplar_adapter,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        if export_mask_dir is not None or output_dir is not None:
            export_results(
                results,
                output_dir=output_dir,
                mask_dir=export_mask_dir,
                overwrite=overwrite,
                save_overlay=output_dir is not None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                preserve_source_filenames=preserve_source_filenames,
            )
        return results

    def track_video(
        self,
        source,
        *,
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
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        annotated_video_path: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
        item_start_callback=None,
        item_result_callback=None,
    ) -> list[PredictionResult]:
        """Run video tracking with frame-wise normalized results."""
        compatibility_tracking = any(
            [
                mask_id is not None,
                mask_label is not None,
                mask_inputs_by_frame,
                points_by_frame,
                boxes_by_frame,
                text_prompts_by_frame,
                normalize_inference_scale(inference_scale) < 0.999,
            ]
        )
        if compatibility_tracking:
            results = compatibility_track_video_frames(
                self.model_loader,
                source,
                text_prompt=text_prompt,
                exemplar_image=exemplar_image,
                exemplar_box=exemplar_box,
                points=points,
                boxes=boxes,
                mask_input=mask_input,
                mask_inputs_by_frame=mask_inputs_by_frame,
                points_by_frame=points_by_frame,
                boxes_by_frame=boxes_by_frame,
                text_prompts_by_frame=text_prompts_by_frame,
                mask_id=mask_id,
                mask_label=mask_label,
                inference_scale=inference_scale,
                exemplar_adapter=self.exemplar_adapter,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                item_start_callback=item_start_callback,
                item_result_callback=item_result_callback,
            )
        else:
            payload = build_prompt_payload(
                text_prompt=text_prompt,
                points=points,
                boxes=boxes,
                mask_input=mask_input,
                exemplar_image=exemplar_image,
                exemplar_box=exemplar_box,
                mask_metadata=self._mask_metadata(mask_id, mask_label),
            )
            validate_prompt_payload(payload, is_video=True)
            results = track_video_sequence(
                self.model_loader,
                source,
                payload,
                exemplar_adapter=self.exemplar_adapter,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                item_result_callback=item_result_callback,
            )

        if export_mask_dir is not None or output_dir is not None or annotated_video_path is not None:
            export_results(
                results,
                output_dir=output_dir,
                mask_dir=export_mask_dir,
                annotated_video_path=annotated_video_path,
                overwrite=overwrite,
                save_overlay=output_dir is not None,
                merged_mask_only=merged_mask_only,
                invert_mask=invert_mask,
                preserve_source_filenames=preserve_source_filenames,
            )
        return results

    def predict_batch(
        self,
        sources,
        *,
        text_prompt=None,
        points=None,
        boxes=None,
        mask_input=None,
        inference_scale: float = 1.0,
        export_mask_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        overwrite: bool = True,
        merged_mask_only: bool = False,
        invert_mask: bool = False,
        preserve_source_filenames: bool = False,
        progress_callback=None,
        cancel_callback=None,
    ) -> list[PredictionResult] | list[list[PredictionResult]]:
        """Run batch inference across a mix of images, videos, and image directories."""
        expanded_sources = expand_sources(sources)
        outputs = []
        total = len(expanded_sources)
        for index, source in enumerate(expanded_sources, start=1):
            if cancel_callback is not None and cancel_callback():
                raise InferenceCancelledError("Batch inference was cancelled.")
            if is_video_path(source):
                result = self.predict_video(
                    source,
                    text_prompt=text_prompt,
                    points=points,
                    boxes=boxes,
                    mask_input=mask_input,
                    inference_scale=inference_scale,
                    export_mask_dir=export_mask_dir,
                    output_dir=output_dir,
                    overwrite=overwrite,
                    merged_mask_only=merged_mask_only,
                    invert_mask=invert_mask,
                    preserve_source_filenames=preserve_source_filenames,
                )
            else:
                result = self.predict_image(
                    source,
                    text_prompt=text_prompt,
                    points=points,
                    boxes=boxes,
                    mask_input=mask_input,
                    inference_scale=inference_scale,
                    export_mask_dir=export_mask_dir,
                    output_dir=output_dir,
                    overwrite=overwrite,
                    merged_mask_only=merged_mask_only,
                    invert_mask=invert_mask,
                    preserve_source_filenames=preserve_source_filenames,
                )
            outputs.append(result)
            if progress_callback is not None:
                progress_callback(index, total, f"Processed {Path(source).name if isinstance(source, (str, Path)) else 'source'}")
        return outputs

    def save_results(
        self,
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
        preserve_source_filenames: bool = False,
        progress_callback=None,
    ):
        """Export normalized results."""
        return export_results(
            results,
            output_dir=output_dir,
            mask_dir=mask_dir,
            annotated_video_path=annotated_video_path,
            overwrite=overwrite,
            save_overlay=save_overlay,
            save_json=save_json,
            save_merged_mask=save_merged_mask,
            save_cutout=save_cutout,
            merged_mask_only=merged_mask_only,
            invert_mask=invert_mask,
            manual_masks_by_key=manual_masks_by_key,
            dilation_pixels=dilation_pixels,
            preserve_source_filenames=preserve_source_filenames,
            progress_callback=progress_callback,
        )
