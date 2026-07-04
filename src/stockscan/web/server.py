"""Dashboard server: background rescan loop + JSON API + static page.

The provider is only hit by the background loop; browser polling reads the
cached result, so free-tier rate limits are respected no matter how many tabs
are open. The optional ?filter= is re-applied server-side to the cached frame.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from ..config import AppConfig
from ..engines.base import Engine
from ..filters import FilterError, apply_filter
from ..scan import ScanRequest, ScanResult, run_scan

STATIC_DIR = Path(__file__).parent / "static"


class ScanCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result: ScanResult | None = None
        self._error: str | None = None

    def store(self, result: ScanResult | None, error: str | None = None) -> None:
        with self._lock:
            if result is not None:
                self._result = result
            self._error = error

    def payload(self, filter_expr: str | None) -> dict[str, Any]:
        with self._lock:
            result, error = self._result, self._error
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


def create_app(
    req: ScanRequest,
    cfg: AppConfig,
    *,
    engines: list[Engine] | None = None,
    interval: int = 60,
    scan_fn=run_scan,
) -> FastAPI:
    cache = ScanCache()
    stop = threading.Event()
    # the dashboard filters client-side against the cache; fetch unfiltered
    base_req = replace(req, filter_expr="", limit=0)

    def loop() -> None:
        while not stop.is_set():
            try:
                cache.store(scan_fn(base_req, cfg, engines=engines))
            except Exception as exc:  # noqa: BLE001 — surfaced in /api/scan status
                cache.store(None, error=f"scan failed: {type(exc).__name__}: {exc}")
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
        return JSONResponse(cache.payload(filter))

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
