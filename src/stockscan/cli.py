"""stockscan CLI: scan, serve, providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import AppConfig
from .models import Timeframe
from .scan import ScanRequest, build_engines, load_watchlist, run_scan

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Stock scanner: Iron Momentum, Silver Flow, and Gold Regime engines "
    "over a live (paid) or delayed (free) market-data feed. Bring your own "
    "PineScript indicators with --pine.",
)
console = Console()

WatchlistArg = Annotated[
    str, typer.Argument(help="Watchlist file (one symbol per line) or comma-separated symbols")
]
ConfigOpt = Annotated[Path | None, typer.Option("--config", help="TOML config file")]
PineOpt = Annotated[
    list[Path] | None,
    typer.Option("--pine", help="PineScript indicator file to run as an extra engine (repeatable)"),
]
PineInputOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--pine-input",
        help='Override a Pine input, e.g. --pine-input "Sensitivity=1.5" (applies to all --pine scripts)',
    ),
]


def _build_request(
    watchlist: str,
    cfg: AppConfig,
    timeframe: str | None,
    provider: str | None,
    lookback: int | None,
    extended: bool | None,
    filter_expr: str | None,
    sort: str | None,
    limit: int | None,
) -> ScanRequest:
    return ScanRequest(
        symbols=load_watchlist(watchlist),
        timeframe=Timeframe.parse(timeframe or cfg.scan.timeframe),
        provider=provider or cfg.scan.provider,
        lookback=lookback or cfg.scan.lookback,
        include_extended=cfg.scan.include_extended if extended is None else extended,
        filter_expr=cfg.scan.filter if filter_expr is None else filter_expr,
        sort_by=sort or cfg.scan.sort,
        descending=cfg.scan.descending,
        limit=cfg.scan.limit if limit is None else limit,
    )


def _engines_with_pine(
    cfg: AppConfig, pine: list[Path] | None, pine_inputs: list[str] | None
):
    overrides = {}
    for item in pine_inputs or []:
        if "=" not in item:
            raise typer.BadParameter(f"--pine-input must be name=value, got {item!r}")
        name, value = item.split("=", 1)
        overrides[name.strip()] = _coerce(value.strip())
    if overrides and pine:
        cfg = _with_overrides(cfg, [str(p) for p in pine], overrides)
    return build_engines(cfg, pine_scripts=[str(p) for p in pine or []])


def _with_overrides(cfg: AppConfig, scripts: list[str], overrides: dict) -> AppConfig:
    from dataclasses import replace

    merged = dict(cfg.pine_inputs)
    for script in scripts:
        merged[script] = {**merged.get(script, {}), **overrides}
    return replace(cfg, pine_inputs=merged)


def _coerce(text: str):
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


@app.command()
def scan(
    watchlist: WatchlistArg,
    timeframe: Annotated[str | None, typer.Option("--timeframe", "-t")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    lookback: Annotated[int | None, typer.Option("--lookback")] = None,
    extended: Annotated[
        bool | None, typer.Option("--extended/--rth-only", help="Include pre/post-market bars")
    ] = None,
    filter_expr: Annotated[
        str | None, typer.Option("--filter", "-f", help='e.g. "mb_state >= 2 and ggr_state >= 1"')
    ] = None,
    sort: Annotated[str | None, typer.Option("--sort", "-s")] = None,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
    json_path: Annotated[Path | None, typer.Option("--json")] = None,
    csv_path: Annotated[Path | None, typer.Option("--csv")] = None,
    pine: PineOpt = None,
    pine_input: PineInputOpt = None,
    config: ConfigOpt = None,
):
    """Run a scan and print a ranked results table."""
    from . import output

    cfg = AppConfig.load(config)
    req = _build_request(
        watchlist, cfg, timeframe, provider, lookback, extended, filter_expr, sort, limit
    )
    engines = _engines_with_pine(cfg, pine, pine_input)
    result = run_scan(req, cfg, engines=engines)
    output.render_table(result, console)
    if json_path:
        output.export_json(result, json_path)
        console.print(f"[dim]wrote {json_path}[/dim]")
    if csv_path:
        output.export_csv(result, csv_path)
        console.print(f"[dim]wrote {csv_path}[/dim]")
    if len(result.rows) == 0 and result.errors:
        raise typer.Exit(code=1)


@app.command()
def serve(
    watchlist: WatchlistArg,
    port: Annotated[int, typer.Option("--port")] = 8501,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    interval: Annotated[int, typer.Option("--interval", help="Rescan interval, seconds")] = 60,
    timeframe: Annotated[str | None, typer.Option("--timeframe", "-t")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    lookback: Annotated[int | None, typer.Option("--lookback")] = None,
    extended: Annotated[bool | None, typer.Option("--extended/--rth-only")] = None,
    filter_expr: Annotated[str | None, typer.Option("--filter", "-f")] = None,
    sort: Annotated[str | None, typer.Option("--sort", "-s")] = None,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
    pine: PineOpt = None,
    pine_input: PineInputOpt = None,
    config: ConfigOpt = None,
):
    """Serve the auto-refreshing scan dashboard."""
    import uvicorn

    from .web.server import create_app

    cfg = AppConfig.load(config)
    req = _build_request(
        watchlist, cfg, timeframe, provider, lookback, extended, filter_expr, sort, limit
    )
    engines = _engines_with_pine(cfg, pine, pine_input)
    web_app = create_app(req, cfg, engines=engines, interval=interval)
    console.print(f"dashboard → [bold]http://{host}:{port}[/bold] (rescan every {interval}s)")
    uvicorn.run(web_app, host=host, port=port, log_level="warning")


@app.command()
def providers(config: ConfigOpt = None):
    """List data providers, capabilities, and whether keys are configured."""
    cfg = AppConfig.load(config)
    table = Table(header_style="bold")
    for col in ("provider", "feed class", "delay", "batch", "keys"):
        table.add_column(col)
    alpaca_keys = bool(cfg.alpaca_key_id and cfg.alpaca_secret)
    polygon_keys = bool(cfg.polygon_api_key)
    rows = [
        ("yfinance", "free (delayed)", "~15 min", "yes", "[green]none needed[/green]"),
        (
            "alpaca",
            f"feed={cfg.alpaca_feed} ({'paid SIP' if cfg.alpaca_feed == 'sip' else 'free IEX'})",
            "realtime (IEX is thinner than SIP)",
            "yes",
            "[green]configured[/green]" if alpaca_keys else "[red]ALPACA_KEY_ID / ALPACA_SECRET missing[/red]",
        ),
        (
            "polygon",
            f"tier={cfg.polygon_tier}",
            "15 min (free) / realtime (paid)",
            "no (5 req/min on free)",
            "[green]configured[/green]" if polygon_keys else "[red]POLYGON_API_KEY missing[/red]",
        ),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)
    if not os.path.exists(".env") and not (alpaca_keys or polygon_keys):
        console.print("[dim]tip: copy .env.example to .env to configure API keys[/dim]")


if __name__ == "__main__":
    app()
