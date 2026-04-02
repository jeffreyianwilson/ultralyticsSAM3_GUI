"""Project save/load helpers for GUI sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gui_state import ViewFilterState

PROJECT_VERSION = 1
PROJECT_SUFFIX = ".sam3proj.json"


def encode_path(path: str | None, *, base_dir: str | Path) -> dict[str, Any] | None:
    """Serialize a filesystem path relative to the project file when practical."""
    if not path:
        return None
    raw = Path(path)
    base = Path(base_dir)
    try:
        value = str(raw.resolve().relative_to(base.resolve()))
        return {"path": value, "relative": True}
    except Exception:
        return {"path": str(raw), "relative": False}


def decode_path(payload: dict[str, Any] | None, *, base_dir: str | Path) -> str | None:
    """Restore a serialized filesystem path."""
    if not payload:
        return None
    value = str(payload.get("path") or "").strip()
    if not value:
        return None
    if bool(payload.get("relative")):
        return str((Path(base_dir) / value).resolve())
    return value


def encode_view_filter_state(state: ViewFilterState) -> dict[str, Any]:
    """Serialize a frame-scoped filter state into JSON-safe data."""
    return {
        "frame_key": state.frame_key,
        "class_options": list(state.class_options),
        "id_options": [int(value) for value in state.id_options],
        "instance_options": [[label, key] for label, key in state.instance_options],
        "selected_classes": sorted(state.selected_classes),
        "selected_ids": sorted(int(value) for value in state.selected_ids),
        "selected_instances": sorted(state.selected_instances),
        "all_classes_selected": bool(state.all_classes_selected),
        "all_ids_selected": bool(state.all_ids_selected),
        "all_instances_selected": bool(state.all_instances_selected),
    }


def decode_view_filter_state(payload: dict[str, Any] | None) -> ViewFilterState:
    """Restore a serialized frame-scoped filter state."""
    if not payload:
        return ViewFilterState()
    return ViewFilterState(
        frame_key=payload.get("frame_key"),
        class_options=[str(value) for value in payload.get("class_options", [])],
        id_options=[int(value) for value in payload.get("id_options", [])],
        instance_options=[
            (str(item[0]), str(item[1]))
            for item in payload.get("instance_options", [])
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ],
        selected_classes={str(value) for value in payload.get("selected_classes", [])},
        selected_ids={int(value) for value in payload.get("selected_ids", [])},
        selected_instances={str(value) for value in payload.get("selected_instances", [])},
        all_classes_selected=bool(payload.get("all_classes_selected", True)),
        all_ids_selected=bool(payload.get("all_ids_selected", True)),
        all_instances_selected=bool(payload.get("all_instances_selected", True)),
    )


def project_cache_dir(project_path: str | Path) -> Path:
    """Return the sidecar cache directory for a project file."""
    path = Path(project_path)
    name = path.name
    if name.endswith(PROJECT_SUFFIX):
        stem = name[: -len(PROJECT_SUFFIX)]
    else:
        stem = path.stem
    return path.with_name(f"{stem}.sam3_cache")


def load_project_document(path: str | Path) -> dict[str, Any]:
    """Read and decode a project JSON document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("version", 0)) != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {payload.get('version')}")
    return payload


def save_project_document(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a normalized project JSON document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    normalized["version"] = PROJECT_VERSION
    with target.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2, sort_keys=False)
