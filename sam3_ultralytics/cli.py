"""Command-line interface for sam3_ultralytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backend import SAM3Ultralytics
from .io_utils import list_image_directory


def _parse_point(raw: str) -> tuple[float, float, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("Point prompts must be X,Y or X,Y,LABEL.")
    label = int(parts[2]) if len(parts) == 3 else 1
    return (float(parts[0]), float(parts[1]), label)


def _parse_box(raw: str) -> tuple[float, float, float, float, int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError("Box prompts must be X1,Y1,X2,Y2 or X1,Y1,X2,Y2,LABEL.")
    label = int(parts[4]) if len(parts) == 5 else 1
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), label)


def _serialize_batch(results):
    serialized = []
    for item in results:
        if hasattr(item, "to_dict"):
            serialized.append(item.to_dict())
        else:
            serialized.append([frame.to_dict() for frame in item])
    return serialized


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to the SAM 3 checkpoint.")
    parser.add_argument("--device", default="auto", help="Device string such as auto, cuda:0, or cpu.")
    parser.add_argument("--text", action="append", default=[], help="Repeatable text prompt.")
    parser.add_argument("--point", action="append", type=_parse_point, default=[], help="Point prompt X,Y,LABEL.")
    parser.add_argument("--box", action="append", type=_parse_box, default=[], help="Box prompt X1,Y1,X2,Y2,LABEL.")
    parser.add_argument("--mask", help="Mask image path used as an initial mask prompt.")
    parser.add_argument("--output-dir", help="Directory for overlay and JSON exports.")
    parser.add_argument("--mask-dir", help="Directory for raw mask exports.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing export files.")
    parser.add_argument("--merged-masks-only", action="store_true", help="Export a single merged mask per source instead of per-object masks.")
    parser.add_argument("--invert-mask", action="store_true", help="Invert exported mask PNGs.")
    parser.add_argument("--reuse-first-mask", action="store_true", help="Reuse the first predicted mask for subsequent items in image or frame sequences.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(prog="sam3-ultralytics", description="SAM 3 wrapper over Ultralytics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Run image prediction.")
    _add_common_args(image_parser)
    image_parser.add_argument("source", help="Image path or directory path.")
    image_parser.add_argument("--all-items", action="store_true", help="Process all images when source is a directory.")

    video_parser = subparsers.add_parser("video-track", help="Run video tracking or per-frame prediction.")
    _add_common_args(video_parser)
    video_parser.add_argument("source", help="Video path.")
    video_parser.add_argument("--annotated-video", help="Annotated output MP4 path.")
    video_parser.add_argument("--current-frame", type=int, help="Run image-style prediction on a single frame index.")
    video_parser.add_argument("--all-frames", action="store_true", help="Run image-style prediction across all frames instead of tracker mode.")

    batch_parser = subparsers.add_parser("batch", help="Run mixed batch prediction.")
    _add_common_args(batch_parser)
    batch_parser.add_argument("sources", nargs="+", help="Image, directory, and video sources.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    backend = SAM3Ultralytics(args.model, device=args.device)
    backend.load()

    payload = {
        "text_prompt": args.text or None,
        "points": args.point or None,
        "boxes": args.box or None,
        "mask_input": args.mask,
    }
    export_kwargs = {
        "output_dir": args.output_dir,
        "export_mask_dir": args.mask_dir,
        "overwrite": args.overwrite,
        "merged_mask_only": args.merged_masks_only,
        "invert_mask": args.invert_mask,
    }

    if args.command == "image":
        source_path = Path(args.source)
        if source_path.exists() and source_path.is_dir():
            if args.all_items or args.reuse_first_mask:
                results = backend.predict_image_sequence(
                    [args.source],
                    reuse_first_mask=args.reuse_first_mask,
                    **export_kwargs,
                    **payload,
                )
                print(json.dumps(_serialize_batch(results), indent=2))
                return 0
            result = backend.predict_image(
                str(list_image_directory(source_path)[0]),
                **export_kwargs,
                **payload,
            )
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        result = backend.predict_image(
            args.source,
            **export_kwargs,
            **payload,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "video-track":
        if args.current_frame is not None:
            results = backend.predict_video_frames(
                args.source,
                frame_indices=[args.current_frame],
                annotated_video_path=args.annotated_video,
                reuse_first_mask=args.reuse_first_mask,
                **export_kwargs,
                **payload,
            )
            print(json.dumps([item.to_dict() for item in results], indent=2))
            return 0
        if args.all_frames or args.reuse_first_mask:
            results = backend.predict_video_frames(
                args.source,
                annotated_video_path=args.annotated_video,
                reuse_first_mask=args.reuse_first_mask,
                **export_kwargs,
                **payload,
            )
        else:
            results = backend.track_video(
                args.source,
                annotated_video_path=args.annotated_video,
                **export_kwargs,
                **payload,
            )
        print(json.dumps([item.to_dict() for item in results], indent=2))
        return 0

    results = backend.predict_batch(
        args.sources,
        **export_kwargs,
        **payload,
    )
    print(json.dumps(_serialize_batch(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
