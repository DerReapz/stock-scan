"""Dashboard server: background rescan loop + JSON API + static front end.

The provider is only hit by the background loop; browser polling reads the
cached result, so free-tier rate limits are respected no matter how many tabs
are open. The optional ?filter= is re-applied server-side to the cached frame.

Endpoints powering the ORE Signal Terminal front end:
  /api/scan      cached results (+ ?filter=), meta, errors
  /api/overview  market aggregates: engine hero cards, breadth, top signals
  /api/tape      index tape quotes (daily bars, provider-dependent symbols)
  /api/library   saved scans + starred tickers (JSON file persistence)
  /api/settings  provider/feed status and scan configuration
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import AppConfig
from ..engines.base import Engine
from ..filters import FilterError, apply_filter
from ..models import Timeframe
from ..scan import ScanRequest, ScanResult, run_scan

STATIC_DIR = Path(__file__).parent / "static"
LIBRARY_FILE = "stockscan_library.json"

# Index tape: yfinance understands ^index symbols; other feeds get the ETFs.
TAPE_YF = [("^GSPC", "SPX"), ("^IXIC", "NASDAQ"), ("^DJI", "DOW"), ("^RUT", "RUSSELL"), ("^VIX", "VIX")]
TAPE_ETF = [("SPY", "SPY"), ("QQQ", "QQQ"), ("DIA", "DIA"), ("IWM", "IWM")]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):  # numpy scalar → Python native
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class ScanCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: ScanResult | None = None
        self._error: str | None = None
        self._tape: list[dict[str, Any]] = []
        self.history: deque[dict[str, Any]] = deque(maxlen=12)

    def store(self, result: ScanResult | None, error: str | None = None) -> None:
        with self._lock:
            if result is not None:
                self._result = result
                self.history.append(_history_point(result))
            self._error = error

    def store_tape(self, tape: list[dict[str, Any]]) -> None:
        with self._lock:
            self._tape = tape

    def snapshot(self) -> tuple[ScanResult | None, str | None]:
        with self._lock:
            return self._result, self._error

    def tape(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._tape)

    def payload(self, filter_expr: str | None) -> dict[str, Any]:
        result, error = self.snapshot()
        if result is None:
            return {"rows": [], "errors": {}, "meta": {}, "status": error or "warming up"}
        if filter_expr:
            try:
                rows = apply_filter(result.rows, filter_expr)
            except FilterError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            result = ScanResult(rows=rows, errors=result.errors, meta=result.meta)
        payload = result.to_json_dict()
        payload["status"] = error or "ok"
        return payload


def _history_point(result: ScanResult) -> dict[str, Any]:
    df = result.rows
    point: dict[str, Any] = {"as_of": result.meta.get("as_of")}
    if len(df) and "mb_score" in df.columns:
        point["iron"] = float((50.0 + df["mb_score"].astype(float)).clip(0, 100).median())
    if len(df) and "sf_inst_net_usd" in df.columns:
        point["silver_net"] = float(df["sf_inst_net_usd"].astype(float).fillna(0).sum())
    return point


class Library:
    """Saved scans + starred tickers, persisted to a JSON file in the cwd."""

    def __init__(self, path: Path | None = None):
        self.path = path or Path(LIBRARY_FILE)
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {"saved_scans": [], "starred": []}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text())
                self.data["saved_scans"] = list(loaded.get("saved_scans", []))
                self.data["starred"] = list(loaded.get("starred", []))
            except (OSError, json.JSONDecodeError):
                pass  # corrupt/unreadable library starts fresh; saving rewrites it

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.data))

    def upsert_scan(self, scan: dict[str, Any]) -> None:
        with self._lock:
            scans = [s for s in self.data["saved_scans"] if s["name"] != scan["name"]]
            scans.append(scan)
            self.data["saved_scans"] = scans
            self._save()

    def delete_scan(self, name: str) -> bool:
        with self._lock:
            before = len(self.data["saved_scans"])
            self.data["saved_scans"] = [
                s for s in self.data["saved_scans"] if s["name"] != name
            ]
            changed = len(self.data["saved_scans"]) != before
            if changed:
                self._save()
            return changed

    def toggle_star(self, symbol: str) -> bool:
        symbol = symbol.upper()
        with self._lock:
            if symbol in self.data["starred"]:
                self.data["starred"].remove(symbol)
                starred = False
            else:
                self.data["starred"].append(symbol)
                starred = True
            self._save()
            return starred


class SavedScanBody(BaseModel):
    name: str
    filter: str = ""
    sort: str = "composite"


def _fetch_tape(req: ScanRequest, cfg: AppConfig) -> list[dict[str, Any]]:
    from ..providers import get_provider

    provider = get_provider(req.provider, cfg)
    pairs = TAPE_YF if req.provider == "yfinance" else TAPE_ETF
    symbols = [s for s, _ in pairs]
    labels = dict(pairs)
    frames, _errors = provider.get_bars(symbols, Timeframe.D1, 3, include_extended=False)
    tape = []
    for symbol in symbols:
        bars = frames.get(symbol)
        if bars is None or len(bars) == 0:
            continue
        last = float(bars["close"].iloc[-1])
        prev = float(bars["close"].iloc[-2]) if len(bars) > 1 else last
        tape.append(
            {
                "symbol": symbol,
                "label": labels[symbol],
                "last": last,
                "chg_pct": (last / prev - 1.0) * 100.0 if prev else 0.0,
            }
        )
    return tape


def _overview_payload(cache: ScanCache) -> dict[str, Any]:
    result, error = cache.snapshot()
    if result is None or len(result.rows) == 0:
        return {"status": error or "warming up"}
    df = result.rows
    n = len(df)

    def col(name: str) -> pd.Series | None:
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else None

    ggr = col("ggr_state")
    mb = col("mb_score")
    bias = col("sf_inst_bias")
    net = col("sf_inst_net_usd")
    day = col("day_chg")

    pct_bull = float((ggr > 0).mean() * 100.0) if ggr is not None else None
    pct_bear = float((ggr < 0).mean() * 100.0) if ggr is not None else None
    regime = (
        None
        if pct_bull is None
        else "RISK-ON"
        if pct_bull >= 55
        else "RISK-OFF"
        if pct_bear >= 55
        else "NEUTRAL"
    )

    iron_now = float((50.0 + mb).clip(0, 100).median()) if mb is not None else None
    history = list(cache.history)
    iron_prev = next(
        (p["iron"] for p in reversed(history[:-1]) if "iron" in p), None
    )
    silver_net = float(net.fillna(0).sum()) if net is not None else None
    silver_hist = [p.get("silver_net") for p in history if "silver_net" in p]

    breadth = {
        "advancers": int((day > 0).sum()) if day is not None else None,
        "decliners": int((day < 0).sum()) if day is not None else None,
        "new_highs": int(df["new_high"].sum()) if "new_high" in df.columns else None,
        "above_sma50_pct": float(df["above_sma50"].mean() * 100.0)
        if "above_sma50" in df.columns
        else None,
        "universe": n,
    }

    top: list[dict[str, Any]] = []
    if "composite" in df.columns:
        ranked = df.sort_values("composite", ascending=False).head(5)
        for symbol, row in ranked.iterrows():
            top.append(
                {
                    "symbol": symbol,
                    "day_chg": _clean(row.get("day_chg")),
                    "composite": _clean(row.get("composite")),
                    "signal": _clean(row.get("signal")),
                    "mb_state": _clean(row.get("mb_state")),
                    "ggr_state": _clean(row.get("ggr_state")),
                    "sf_regime": _clean(row.get("sf_regime")),
                }
            )

    return {
        "status": error or "ok",
        "as_of": result.meta.get("as_of"),
        "gold": {
            "regime": regime,
            "pct_bull": pct_bull,
            "pct_bear": pct_bear,
        },
        "iron": {
            "score": iron_now,
            "prev_score": iron_prev,
            "pct_positive": float((mb > 0).mean() * 100.0) if mb is not None else None,
        },
        "silver": {
            "net_usd": silver_net,
            "pct_inflow": float((bias > 0).mean() * 100.0) if bias is not None else None,
            "history": silver_hist,
        },
        "breadth": breadth,
        "top_signals": top,
    }


def create_app(
    req: ScanRequest,
    cfg: AppConfig,
    *,
    engines: list[Engine] | None = None,
    interval: int = 60,
    scan_fn=run_scan,
    tape_fn=_fetch_tape,
    library_path: Path | None = None,
) -> FastAPI:
    cache = ScanCache()
    library = Library(library_path)
    stop = threading.Event()
    # the dashboard filters client-side against the cache; fetch unfiltered
    base_req = replace(req, filter_expr="", limit=0)

    def loop() -> None:
        while not stop.is_set():
            try:
                cache.store(scan_fn(base_req, cfg, engines=engines))
            except Exception as exc:  # noqa: BLE001 — surfaced in /api/scan status
                cache.store(None, error=f"scan failed: {type(exc).__name__}: {exc}")
            try:
                cache.store_tape(tape_fn(base_req, cfg))
            except Exception:  # noqa: BLE001 — tape is decorative, never fatal
                pass
            stop.wait(interval)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        thread = threading.Thread(target=loop, daemon=True, name="scan-loop")
        thread.start()
        yield
        stop.set()

    app = FastAPI(title="stockscan dashboard", lifespan=lifespan)

    @app.get("/api/scan")
    def api_scan(filter: str | None = Query(default=None)) -> JSONResponse:  # noqa: A002
        payload = cache.payload(filter)
        payload["starred"] = library.as_dict()["starred"]
        return JSONResponse(payload)

    @app.get("/api/overview")
    def api_overview() -> JSONResponse:
        return JSONResponse(_overview_payload(cache))

    @app.get("/api/tape")
    def api_tape() -> dict[str, Any]:
        return {"tape": cache.tape()}

    @app.get("/api/library")
    def api_library() -> dict[str, Any]:
        data = library.as_dict()
        result, _ = cache.snapshot()
        rows: list[dict[str, Any]] = []
        if result is not None and len(result.rows):
            for symbol in data["starred"]:
                if symbol in result.rows.index:
                    row = result.rows.loc[symbol]
                    rows.append(
                        {
                            "symbol": symbol,
                            "last_price": _clean(row.get("last_price")),
                            "day_chg": _clean(row.get("day_chg")),
                            "composite": _clean(row.get("composite")),
                            "signal": _clean(row.get("signal")),
                            "mb_score": _clean(row.get("mb_score")),
                            "ggr_state": _clean(row.get("ggr_state")),
                            "sf_regime": _clean(row.get("sf_regime")),
                        }
                    )
                else:
                    rows.append({"symbol": symbol})
        else:
            rows = [{"symbol": s} for s in data["starred"]]
        data["starred_rows"] = rows
        return data

    @app.post("/api/library/scans")
    def api_save_scan(body: SavedScanBody) -> dict[str, Any]:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="scan name required")
        from datetime import UTC, datetime

        library.upsert_scan(
            {
                "name": name,
                "filter": body.filter,
                "sort": body.sort,
                "updated": datetime.now(UTC).isoformat(),
            }
        )
        return library.as_dict()

    @app.delete("/api/library/scans/{name}")
    def api_delete_scan(name: str) -> dict[str, Any]:
        if not library.delete_scan(name):
            raise HTTPException(status_code=404, detail=f"no saved scan named {name!r}")
        return library.as_dict()

    @app.post("/api/library/star/{symbol}")
    def api_star(symbol: str) -> dict[str, Any]:
        starred = library.toggle_star(symbol)
        return {"symbol": symbol.upper(), "starred": starred}

    @app.get("/api/settings")
    def api_settings() -> dict[str, Any]:
        result, _ = cache.snapshot()
        meta = result.meta if result is not None else {}
        return {
            "provider": req.provider,
            "timeframe": req.timeframe.value,
            "interval": interval,
            "universe": len(req.symbols),
            "symbols": req.symbols,
            "is_delayed": meta.get("is_delayed"),
            "delay_seconds": meta.get("delay_seconds"),
            "note": meta.get("note"),
            "alpaca": {"configured": bool(cfg.alpaca_key_id and cfg.alpaca_secret), "feed": cfg.alpaca_feed},
            "polygon": {"configured": bool(cfg.polygon_api_key), "tier": cfg.polygon_tier},
            "default_filter": req.filter_expr,
            "default_sort": req.sort_by,
        }

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        return {
            "symbols": req.symbols,
            "timeframe": req.timeframe.value,
            "provider": req.provider,
            "interval": interval,
            "default_filter": req.filter_expr,
            "default_sort": req.sort_by,
        }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
