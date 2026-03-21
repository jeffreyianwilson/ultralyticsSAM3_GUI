"""Package exceptions for sam3_ultralytics."""

from __future__ import annotations


class SAM3UltralyticsError(Exception):
    """Base package error."""


class ModelNotLoadedError(SAM3UltralyticsError):
    """Raised when inference is requested before a model is ready."""


class InvalidSourceError(SAM3UltralyticsError):
    """Raised when an input source is invalid or unsupported."""


class UnsupportedPromptError(SAM3UltralyticsError):
    """Raised when a prompt combination is unsupported by the backend."""


class ExportError(SAM3UltralyticsError):
    """Raised when result export fails."""


class InferenceCancelledError(SAM3UltralyticsError):
    """Raised when an inference job is cancelled."""
