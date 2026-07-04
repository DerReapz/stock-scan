"""Dashboard API tests with an injected fake scan function."""

import time

import pandas as pd
from fastapi.testclient import TestClient

from stockscan.config import AppConfig
from stockscan.models import Timeframe
from stockscan.scan import ScanRequest, ScanResult


def fake_result() -> ScanResult:
    rows = pd.DataFrame(
        {
            "last_price": [101.0, 55.0],
            "mb_state": [2, -1],
            "mb_score": [35.2, -8.1],
            "ggr_state": [3, 0],
        },
        index=pd.Index(["AAA", "BBB"], name="symbol"),
    )
    return ScanResult(rows=rows, errors={"CCC": "no data"}, meta={"provider": "fake"})


def make_client() -> TestClient:
    from stockscan.web.server import create_app

    req = ScanRequest(symbols=["AAA", "BBB", "CCC"], timeframe=Timeframe.M5)
    app = create_app(
        req, AppConfig(), engines=[], interval=3600, scan_fn=lambda *a, **k: fake_result()
    )
    return TestClient(app)


def wait_for_scan(client: TestClient, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get("/api/scan").json()
        if data["rows"]:
            return data
        time.sleep(0.05)
    raise AssertionError("background scan never populated the cache")


def test_scan_endpoint_and_filter():
    with make_client() as client:
        data = wait_for_scan(client)
        assert {r["symbol"] for r in data["rows"]} == {"AAA", "BBB"}
        assert data["errors"] == {"CCC": "no data"}

        filtered = client.get("/api/scan", params={"filter": "mb_state >= 1"}).json()
        assert [r["symbol"] for r in filtered["rows"]] == ["AAA"]

        bad = client.get("/api/scan", params={"filter": "nope > 1"})
        assert bad.status_code == 400


def test_index_and_config():
    with make_client() as client:
        wait_for_scan(client)
        page = client.get("/")
        assert page.status_code == 200
        assert "stockscan" in page.text
        cfg = client.get("/api/config").json()
        assert cfg["symbols"] == ["AAA", "BBB", "CCC"]
        assert cfg["timeframe"] == "5m"
