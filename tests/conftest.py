import os
import shutil
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

import _pytest.pathlib as pytest_pathlib
import _pytest.tmpdir as pytest_tmpdir


_orig_cleanup_dead_symlinks = pytest_pathlib.cleanup_dead_symlinks


def _safe_cleanup_dead_symlinks(root):
    try:
        _orig_cleanup_dead_symlinks(root)
    except PermissionError:
        return


pytest_pathlib.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks
pytest_tmpdir.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


@pytest.fixture
def tmp_path():
    root = Path(tempfile.gettempdir()) / "sam3-ultralytics-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
