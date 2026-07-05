"""PyInstaller entry point for the standalone stockscan executable.

Kept outside the package so the frozen app imports ``stockscan`` the same way
a pip install does.

Double-click behavior: launching the exe from Explorer used to print a
quickstart that *mentioned* the dashboard URL — but no server was running, so
the address went nowhere. Now a double-click (or STOCKSCAN_AUTOLAUNCH=1)
launches the dashboard directly: pick a watchlist (watchlist.txt or
watchlists/default.txt next to the exe or in the working directory, else a
built-in default), bind the first free port from 8501, open the browser, and
run ``serve``. The console window stays open — it IS the server — and any
crash holds the window so the error stays readable.
"""

import multiprocessing
import os
import sys
import traceback
from pathlib import Path

DEFAULT_SYMBOLS = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,NFLX,SPY,QQQ"

BANNER = """\
──────────────────────────────────────────────────────────────────
  ORE Signal Terminal — starting the dashboard

  watchlist : {watchlist}
  dashboard : {url}   (opening in your browser…)

  Keep this window open — it is the scanner. Close it (or press
  Ctrl+C) to stop.

  Tip: put a watchlist.txt (one symbol per line) next to the exe
  to scan your own list, or run from a terminal for full control:
      stockscan serve watchlist.txt --timeframe 15m
      stockscan --help
──────────────────────────────────────────────────────────────────
"""


def _owns_console() -> bool:
    """True when the console belongs to this app alone, i.e. the window dies
    with the process (Explorer double-click).

    A PyInstaller onefile build runs as TWO attached processes — the
    bootloader (our parent) and this child — so "count == 1" never holds for
    it. Instead: the console is ours alone iff every attached process is this
    process or its parent. A launching shell (cmd/PowerShell) stays attached
    to the console, fails that test, and keeps normal terminal behavior.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        pids = (ctypes.c_uint * 16)()
        count = ctypes.windll.kernel32.GetConsoleProcessList(pids, 16)
        if count == 0 or count > 16:
            return False
        ours = {os.getpid(), os.getppid()}
        return all(pid in ours for pid in pids[:count])
    except Exception:  # noqa: BLE001 — heuristic only, never fatal
        return False


def _find_watchlist() -> str:
    exe_dir = Path(sys.executable).parent
    for base in (Path.cwd(), exe_dir):
        for candidate in (base / "watchlist.txt", base / "watchlists" / "default.txt"):
            if candidate.is_file():
                return str(candidate)
    return DEFAULT_SYMBOLS


def _free_port(start: int = 8501) -> int:
    import socket

    for port in range(start, start + 25):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _autolaunch_dashboard() -> None:
    """Rewrite argv into a `serve` invocation and open the browser once the
    server has had a moment to bind."""
    import threading
    import webbrowser

    watchlist = _find_watchlist()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    shown = watchlist if watchlist != DEFAULT_SYMBOLS else "built-in default (large caps + SPY/QQQ)"
    print(BANNER.format(watchlist=shown, url=url), flush=True)

    def open_browser() -> None:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — the URL is printed either way
            pass

    threading.Timer(3.0, open_browser).start()
    sys.argv = [sys.argv[0], "serve", watchlist, "--port", str(port)]


def main() -> int:
    multiprocessing.freeze_support()
    hold_window = _owns_console()
    force_launch = os.environ.get("STOCKSCAN_AUTOLAUNCH") == "1"

    if len(sys.argv) == 1 and (hold_window or force_launch):
        _autolaunch_dashboard()

    from stockscan.cli import app

    try:
        app()
    except SystemExit as exc:
        return int(exc.code or 0)
    except KeyboardInterrupt:
        return 130
    except Exception:  # noqa: BLE001 — keep the traceback visible on screen
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    code = main()
    if _owns_console():
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass
    sys.exit(code)
