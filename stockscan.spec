# PyInstaller spec: build a single-file stockscan executable.
#
#   pip install -e ".[build]"     (or: uv pip install -e ".[build]")
#   pyinstaller stockscan.spec
#
# Output lands in dist/stockscan (dist/stockscan.exe on Windows). The build is
# per-platform: run it on Windows to get a Windows .exe, on Linux/macOS for a
# native binary — PyInstaller does not cross-compile.

from PyInstaller.utils.hooks import collect_submodules

# uvicorn resolves its loop/protocol/logging classes from strings at runtime,
# so its submodules never appear as static imports.
hiddenimports = sorted(
    set(collect_submodules("uvicorn"))
    | {"dotenv", "stockscan"}
)

a = Analysis(
    ["packaging/pyinstaller_entry.py"],
    pathex=["src"],
    binaries=[],
    datas=[
        # dashboard front end, served by web/server.py via Path(__file__)
        ("src/stockscan/web/static", "stockscan/web/static"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # heavyweight libraries pulled in transitively but never used
        "matplotlib", "PIL", "tkinter", "IPython", "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name="stockscan",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
