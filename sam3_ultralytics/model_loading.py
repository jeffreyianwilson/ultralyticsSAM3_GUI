"""Ultralytics SAM 3 model and predictor loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .exceptions import InvalidSourceError, ModelNotLoadedError
from .io_utils import configure_yolo_environment


class ModelLoader:
    """Lazy loader for the Ultralytics SAM 3 predictors used by the backend."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        half: bool | None = None,
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.7,
        yolo_config_dir: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = self.resolve_device(device)
        self.half = bool(half) if half is not None else False
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.yolo_config_dir = configure_yolo_environment(yolo_config_dir)

        self._semantic_image = None
        self._interactive_image = None
        self._semantic_video = None
        self._interactive_video = None

    @staticmethod
    def resolve_device(device: str | None) -> str:
        """Resolve a device string to an actual backend device."""
        if device is None or device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and torch.cuda.is_available():
            return "cuda:0"
        return device

    def load(self) -> "ModelLoader":
        """Validate the checkpoint and prepare the runtime environment."""
        if not self.model_path.exists():
            raise InvalidSourceError(f"Model checkpoint does not exist: {self.model_path}")
        configure_yolo_environment(self.yolo_config_dir)
        return self

    def _predictor_overrides(self, device: str | None = None) -> dict[str, Any]:
        resolved_device = self.resolve_device(device or self.device)
        return {
            "model": str(self.model_path),
            "device": resolved_device,
            "half": self.half and resolved_device.startswith("cuda"),
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "save": False,
            "show": False,
            "verbose": False,
            "exist_ok": True,
            "retina_masks": True,
        }

    @staticmethod
    def _set_predictor_runtime_defaults(predictor):
        args = getattr(predictor, "args", None)
        if args is not None and hasattr(args, "compile"):
            args.compile = None
        return predictor

    def get_semantic_image_predictor(self, device_override: str | None = None):
        """Return the image semantic predictor."""
        if device_override is not None:
            from ultralytics.models.sam.predict import SAM3SemanticPredictor

            return self._set_predictor_runtime_defaults(SAM3SemanticPredictor(overrides=self._predictor_overrides(device_override)))
        if self._semantic_image is None:
            from ultralytics.models.sam.predict import SAM3SemanticPredictor

            self._semantic_image = self._set_predictor_runtime_defaults(SAM3SemanticPredictor(overrides=self._predictor_overrides()))
        return self._semantic_image

    def get_interactive_image_predictor(self):
        """Return the image interactive predictor."""
        if self._interactive_image is None:
            from ultralytics.models.sam.predict import SAM3Predictor

            self._interactive_image = self._set_predictor_runtime_defaults(SAM3Predictor(overrides=self._predictor_overrides()))
        return self._interactive_image

    def get_semantic_video_predictor(self):
        """Return the video semantic predictor."""
        if self._semantic_video is None:
            from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor

            self._semantic_video = self._set_predictor_runtime_defaults(SAM3VideoSemanticPredictor(overrides=self._predictor_overrides()))
        return self._semantic_video

    def get_interactive_video_predictor(self):
        """Return the video interactive predictor."""
        if self._interactive_video is None:
            from ultralytics.models.sam.predict import SAM3VideoPredictor

            self._interactive_video = self._set_predictor_runtime_defaults(SAM3VideoPredictor(overrides=self._predictor_overrides()))
        return self._interactive_video

    def ensure_ready(self) -> None:
        """Raise when the loader has not been validated yet."""
        if not self.model_path.exists():
            raise ModelNotLoadedError("A valid SAM 3 checkpoint must be loaded before inference.")
