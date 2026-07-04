"""PyInstaller entry point for the standalone stockscan executable.

Kept outside the package so the frozen app imports ``stockscan`` the same way
a pip install does. ``freeze_support`` is required on Windows in case any
dependency spawns worker processes in a frozen context.
"""

import multiprocessing

from stockscan.cli import app

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app()
