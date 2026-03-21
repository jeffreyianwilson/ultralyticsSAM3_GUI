"""Video inference helpers."""

from __future__ import annotations

from time import perf_counter

from .exceptions import InferenceCancelledError, UnsupportedPromptError
from .image_inference import normalize_ultralytics_result, run_image_prediction
from .io_utils import read_video_frame, video_frame_count
from .schemas import BoxPrompt, PredictionResult, PromptPayload


def _dedupe_boxes(boxes: list[BoxPrompt]) -> list[BoxPrompt]:
    deduped: list[BoxPrompt] = []
    seen: set[tuple[float, float, float, float, int]] = set()
    for box in boxes:
        key = (
            round(box.x1, 3),
            round(box.y1, 3),
            round(box.x2, 3),
            round(box.y2, 3),
            int(box.label),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def _boxes_from_result(result: PredictionResult) -> list[BoxPrompt]:
    boxes: list[BoxPrompt] = []
    for obj in result.objects:
        if obj.box is None:
            continue
        boxes.append(BoxPrompt(*obj.box, 1))
    return boxes


def _consume_stream(
    stream,
    *,
    source,
    payload: PromptPayload,
    total_frames,
    progress_callback=None,
    cancel_callback=None,
    item_result_callback=None,
):
    results: list[PredictionResult] = []
    for frame_index, raw_result in enumerate(stream):
        if cancel_callback is not None and cancel_callback():
            raise InferenceCancelledError("Video inference was cancelled.")
        normalized = normalize_ultralytics_result(raw_result, payload, mode="video", frame_index=frame_index)
        normalized.source = str(source)
        results.append(normalized)
        if item_result_callback is not None:
            item_result_callback(frame_index, total_frames, normalized, f"frame:{frame_index}")
        if progress_callback is not None:
            progress_callback(frame_index + 1, total_frames, f"Processed frame {frame_index + 1}")
    return results


def _run_semantic_stream(
    model_loader,
    *,
    source,
    payload: PromptPayload,
    total_frames,
    progress_callback=None,
    cancel_callback=None,
    item_result_callback=None,
):
    predictor = model_loader.get_semantic_video_predictor()
    stream = predictor(
        source=source,
        text=payload.texts or None,
        bboxes=[list(box.xyxy) for box in payload.boxes] or None,
        labels=[box.label for box in payload.boxes] or None,
        stream=True,
    )
    return _consume_stream(
        stream,
        source=source,
        payload=payload,
        total_frames=total_frames,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        item_result_callback=item_result_callback,
    )


def _run_interactive_stream(
    model_loader,
    *,
    source,
    payload: PromptPayload,
    total_frames,
    progress_callback=None,
    cancel_callback=None,
    item_result_callback=None,
):
    predictor = model_loader.get_interactive_video_predictor()
    stream = predictor(
        source=source,
        bboxes=[list(box.xyxy) for box in payload.boxes] or None,
        points=[[point.x, point.y] for point in payload.points] or None,
        labels=[point.label for point in payload.points] or None,
        masks=payload.mask_input,
        stream=True,
    )
    return _consume_stream(
        stream,
        source=source,
        payload=payload,
        total_frames=total_frames,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        item_result_callback=item_result_callback,
    )


def run_video_prediction(
    model_loader,
    source,
    payload: PromptPayload,
    *,
    exemplar_adapter=None,
    progress_callback=None,
    cancel_callback=None,
    item_result_callback=None,
) -> list[PredictionResult]:
    """Run video prediction and tracking with frame-wise normalized outputs."""
    if payload.has_exemplar:
        if exemplar_adapter is None:
            raise UnsupportedPromptError(
                "Exemplar prompting requires a dedicated compatibility adapter because this Ultralytics SAM 3 build "
                "does not expose a stable public exemplar-prompt API."
            )
        return exemplar_adapter.track_video(
            model_loader=model_loader,
            source=source,
            payload=payload,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )

    start = perf_counter()
    total_frames = video_frame_count(source)
    direct_semantic = payload.has_text and not payload.has_mask_input and not payload.has_points

    if direct_semantic:
        results = _run_semantic_stream(
            model_loader,
            source=source,
            payload=payload,
            total_frames=total_frames,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            item_result_callback=item_result_callback,
        )
    else:
        interactive_payload = payload
        if payload.has_text:
            first_frame = read_video_frame(source, 0)
            seed_payload = PromptPayload(texts=list(payload.texts), boxes=list(payload.boxes))
            semantic_seed = run_image_prediction(model_loader, first_frame, seed_payload)
            interactive_payload = PromptPayload(
                texts=list(payload.texts),
                points=list(payload.points),
                boxes=_dedupe_boxes(list(payload.boxes) + _boxes_from_result(semantic_seed)),
                mask_input=payload.mask_input,
                mask_metadata=dict(payload.mask_metadata),
            )
        results = _run_interactive_stream(
            model_loader,
            source=source,
            payload=interactive_payload,
            total_frames=total_frames,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            item_result_callback=item_result_callback,
        )
        if payload.has_text:
            for item in results:
                item.prompt_metadata["compatibility_mode"] = "semantic_first_frame_then_interactive_track"
                item.prompt_metadata["refinement_box_count"] = len(interactive_payload.boxes)

    elapsed_ms = (perf_counter() - start) * 1000.0
    for item in results:
        item.timings.setdefault("backend_total_ms", elapsed_ms / max(len(results), 1))
    return results
