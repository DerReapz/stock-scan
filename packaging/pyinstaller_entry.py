"""PyInstaller entry point for the standalone stockscan executable.

Kept outside the package so the frozen app imports ``stockscan`` the same way
a pip install does.

Windows double-click handling: launching a console executable from Explorer
spawns a console owned solely by this process, which Windows destroys the
instant the process exits — help text or a traceback flashes for a moment and
vanishes. When we detect that case, show a quickstart if no command was given
and hold the window open until Enter is pressed, including after crashes so
errors stay readable.
"""

import multiprocessing
import os
import sys
import traceback

QUICKSTART = """\
stockscan is a command-line tool — open PowerShell or cmd in this folder and
run a command, e.g.:

  stockscan scan AAPL,MSFT,NVDA --timeframe 15m
  stockscan scan watchlist.txt --filter "mb_state >= 2 and ggr_state >= 1"
  stockscan serve watchlist.txt          (dashboard on http://127.0.0.1:8501)
  stockscan scan AAPL --pine my_indicator.pine
  stockscan providers                    (data feeds + API-key status)
  stockscan --help                       (all options)

A watchlist is a text file with one symbol per line. API keys for the live
feeds go in a .env file next to where you run the command (see the project's
.env.example).
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


def main() -> int:
    multiprocessing.freeze_support()
    hold_window = _owns_console()

    if hold_window and len(sys.argv) == 1:
        print(QUICKSTART)
        return 0

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
