"""Enable ``python -m stockscan`` (and give PyInstaller a module entry)."""

from .cli import app

if __name__ == "__main__":
    app()
