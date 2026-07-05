"""Result rendering: rich terminal table + JSON/CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from .scan import ScanResult

# Columns shown in the terminal table (full set goes to JSON/CSV).
DEFAULT_COLUMNS = [
    "last_price",
    "day_chg",
    "composite",
    "signal",
    "mb_score",
    "mb_state",
    "mb_rvol",
    "mb_star",
    "ggr_state",
    "ggr_bars_since_flip",
    "sf_regime",
    "sf_inst_bias",
    "sf_accum",
    "sf_dist",
    "sf_confluence",
]

# Context columns kept out of the terminal view (still in JSON/CSV/filters).
HIDDEN_COLUMNS = {"last_bar", "pct_chg", "above_sma50", "new_high", "ggr_width_atr", "sf_inst"}

_STATE_STYLES = {2: "bold green", 1: "green", 0: "dim", -1: "red", -2: "bold red",
                 3: "bold green", -3: "bold red"}


def render_table(result: ScanResult, console: Console | None = None) -> None:
    console = console or Console()
    meta = result.meta
    if meta.get("is_delayed"):
        delay_min = int(meta.get("delay_seconds", 0)) // 60
        console.print(
            f"[yellow]⚠ data delayed ~{delay_min} min ({meta.get('provider')})[/yellow]"
        )
    title = (
        f"scan · {meta.get('timeframe')} · {meta.get('provider')} · "
        f"{meta.get('symbols_scanned')}/{meta.get('symbols_requested')} symbols"
    )
    table = Table(title=title, header_style="bold")
    table.add_column("symbol", style="bold cyan")

    df = result.rows
    columns = [c for c in DEFAULT_COLUMNS if c in df.columns]
    extra = [
        c
        for c in df.columns
        if c not in columns and c not in HIDDEN_COLUMNS and not _is_builtin(c)
    ]
    columns += extra
    for col in columns:
        table.add_column(col, justify="right")

    for symbol, row in df.iterrows():
        cells = [str(symbol)]
        for col in columns:
            cells.append(_format_cell(col, row[col]))
        table.add_row(*cells)
    console.print(table)

    if result.errors:
        console.print("[dim]skipped:[/dim]")
        for symbol, err in sorted(result.errors.items()):
            console.print(f"  [dim]{symbol}: {err}[/dim]")


def _is_builtin(col: str) -> bool:
    return col.startswith(("mb_", "ggr_", "sf_"))


def _format_cell(col: str, value) -> str:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return "[dim]—[/dim]"
    if isinstance(value, bool):
        return "[green]✓[/green]" if value else "[dim]·[/dim]"
    if col.endswith(("state", "regime", "flip", "star", "confluence")):
        iv = int(value)
        style = _STATE_STYLES.get(iv, "white")
        sign = "+" if iv > 0 else ""
        return f"[{style}]{sign}{iv}[/{style}]"
    if isinstance(value, float):
        if col in ("pct_chg", "day_chg"):
            style = "green" if value >= 0 else "red"
            return f"[{style}]{value:+.2f}%[/{style}]"
        return f"{value:,.2f}"
    if col == "signal":
        style = {"Strong Buy": "bold green", "Buy": "green", "Watch": "yellow",
                 "Neutral": "dim", "Avoid": "red"}.get(str(value), "white")
        return f"[{style}]{value}[/{style}]"
    return str(value)


def export_json(result: ScanResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_json_dict(), indent=2, default=str))


def export_csv(result: ScanResult, path: Path) -> None:
    result.rows.to_csv(path)
