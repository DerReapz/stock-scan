"""Scan orchestration: watchlist → bars → engines → results frame."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .engines import (
    GoldRegimeEngine,
    IronMomentumEngine,
    SilverFlowEngine,
)
from .engines.base import Engine
from .filters import apply_filter
from .models import Timeframe
from .providers import get_provider


@dataclass
class ScanRequest:
    symbols: list[str]
    timeframe: Timeframe
    provider: str = "yfinance"
    lookback: int = 300
    include_extended: bool = True
    filter_expr: str = ""
    sort_by: str = "mb_score"
    descending: bool = True
    limit: int = 0


@dataclass
class ScanResult:
    rows: pd.DataFrame
    errors: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        rows = self.rows.reset_index(names="symbol") if len(self.rows) else pd.DataFrame()
        records = rows.to_dict(orient="records") if len(rows) else []
        for record in records:
            for key, value in record.items():
                if isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif pd.api.types.is_scalar(value) and pd.isna(value):
                    record[key] = None
        return {"rows": records, "errors": self.errors, "meta": self.meta}


def build_engines(cfg: AppConfig, pine_scripts: list[str] | None = None) -> list[Engine]:
    engines: list[Engine] = [
        IronMomentumEngine(cfg.iron_momentum),
        GoldRegimeEngine(cfg.gold_regime),
        SilverFlowEngine(cfg.silver_flow),
    ]
    scripts = list(cfg.pine_scripts)
    for script in pine_scripts or []:
        if script not in scripts:
            scripts.append(script)
    for script in scripts:
        from .pine.engine import PineEngine

        overrides = cfg.pine_inputs.get(script, {})
        engines.append(PineEngine.from_file(Path(script), input_overrides=overrides))
    return engines


def load_watchlist(source: str) -> list[str]:
    """A path to a watchlist file, or a comma-separated symbol list."""
    path = Path(source)
    if path.is_file():
        symbols = []
        for line in path.read_text().splitlines():
            entry = line.split("#", 1)[0].strip().upper()
            if entry:
                symbols.append(entry)
        return symbols
    looks_like_path = any(ch in source for ch in "/\\") or source.lower().endswith(".txt")
    if not looks_like_path and source.strip():
        return [s.strip().upper() for s in source.split(",") if s.strip()]
    raise FileNotFoundError(f"watchlist file not found: {source}")


# Signal tiers over the 0-100 composite score (equal-weight blend of the
# three engines, each normalized to 0-100).
SIGNAL_TIERS = (
    (80.0, "Strong Buy"),
    (68.0, "Buy"),
    (50.0, "Watch"),
    (45.0, "Neutral"),
)


def _with_composite(df: pd.DataFrame) -> pd.DataFrame:
    needed = ("mb_score", "ggr_state", "sf_inst_bias")
    if not all(col in df.columns for col in needed):
        return df
    iron = (50.0 + pd.to_numeric(df["mb_score"], errors="coerce")).clip(0, 100)
    gold = (pd.to_numeric(df["ggr_state"], errors="coerce") + 3.0) / 6.0 * 100.0
    silver = ((pd.to_numeric(df["sf_inst_bias"], errors="coerce") + 1.0) / 2.0 * 100.0).clip(
        0, 100
    )
    composite = ((iron + gold + silver) / 3.0).round(1)
    df = df.copy()
    df["composite"] = composite

    def tier(score: float) -> str | None:
        if pd.isna(score):
            return None
        for threshold, label in SIGNAL_TIERS:
            if score >= threshold:
                return label
        return "Avoid"

    df["signal"] = composite.map(tier)
    return df


def _day_change(bars: pd.DataFrame) -> float:
    """Percent change of the last close vs the previous trading day's close
    (falls back to bar-over-bar when there is no prior session in frame)."""
    closes = bars["close"]
    if len(closes) < 2:
        return 0.0
    dates = bars.index.tz_convert("America/New_York").date
    last_date = dates[-1]
    prior = closes[dates < last_date]
    reference = prior.iloc[-1] if len(prior) else closes.iloc[-2]
    return float((closes.iloc[-1] / reference - 1.0) * 100.0)


def run_scan(
    req: ScanRequest, cfg: AppConfig, *, engines: list[Engine] | None = None
) -> ScanResult:
    engines = engines if engines is not None else build_engines(cfg)
    warmup = max(engine.warmup_bars() for engine in engines)
    lookback = max(req.lookback, warmup + 30)

    provider = get_provider(req.provider, cfg)
    frames, errors = provider.get_bars(
        req.symbols, req.timeframe, lookback, include_extended=req.include_extended
    )

    def compute_row(item: tuple[str, pd.DataFrame]) -> tuple[str, dict[str, Any] | None, str]:
        symbol, bars = item
        if len(bars) < warmup:
            return symbol, None, f"only {len(bars)} bars, need >= {warmup} for warmup"
        closes = bars["close"]
        sma50 = closes.iloc[-50:].mean() if len(closes) >= 50 else closes.mean()
        row: dict[str, Any] = {
            "last_price": float(closes.iloc[-1]),
            "pct_chg": float((closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0)
            if len(bars) > 1
            else 0.0,
            "day_chg": _day_change(bars),
            "above_sma50": bool(closes.iloc[-1] > sma50),
            "new_high": bool(closes.iloc[-1] >= closes.max()),
            "last_bar": bars.index[-1],
        }
        try:
            for engine in engines:
                row.update(engine.compute(bars).snapshot)
        except Exception as exc:  # noqa: BLE001 — reported per symbol
            return symbol, None, f"{type(exc).__name__}: {exc}"
        return symbol, row, ""

    rows: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for symbol, row, err in pool.map(compute_row, frames.items()):
            if row is None:
                errors[symbol] = err
            else:
                rows[symbol] = row

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "symbol"
    if len(df):
        df = _with_composite(df)
        if req.filter_expr:
            df = apply_filter(df, req.filter_expr)
        sort_col = req.sort_by if req.sort_by in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=not req.descending, na_position="last")
        if req.limit > 0:
            df = df.head(req.limit)

    caps = provider.capabilities
    meta = {
        "provider": caps.name,
        "is_delayed": caps.is_delayed,
        "delay_seconds": caps.delay_seconds,
        "note": caps.note,
        "timeframe": req.timeframe.value,
        "as_of": datetime.now(UTC).isoformat(),
        "symbols_requested": len(req.symbols),
        "symbols_scanned": len(rows),
        "filter": req.filter_expr,
    }
    return ScanResult(rows=df, errors=errors, meta=meta)
