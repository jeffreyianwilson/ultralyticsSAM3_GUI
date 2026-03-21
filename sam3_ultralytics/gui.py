"""GUI module entrypoint."""

from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .gui_app import SAM3MainWindow


def main() -> int:
    """Launch the desktop GUI."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = SAM3MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
