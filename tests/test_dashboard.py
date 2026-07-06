"""Dashboard API tests with injected fake scan/tape functions."""

import time

import pandas as pd
from fastapi.testclient import TestClient

from stockscan.config import AppConfig
from stockscan.models import Timeframe
from stockscan.scan import ScanRequest, ScanResult


def fake_result() -> ScanResult:
    rows = pd.DataFrame(
        {
            "last_price": [101.0, 55.0, 20.0],
            "day_chg": [1.5, -2.0, 0.4],
            "above_sma50": [True, False, True],
            "new_high": [True, False, False],
            "mb_state": [2, -1, 1],
            "mb_score": [35.2, -8.1, 12.0],
            "ggr_state": [3, 0, -1],
            "sf_regime": [-1, 1, -1],
            "sf_inst_bias": [0.4, -0.2, 0.1],
            "sf_inst_net_usd": [2.1e9, -0.4e9, 0.7e9],
            "composite": [88.0, 41.0, 62.0],
            "signal": ["Strong Buy", "Avoid", "Watch"],
        },
        index=pd.Index(["AAA", "BBB", "DDD"], name="symbol"),
    )
    return ScanResult(
        rows=rows,
        errors={"CCC": "no data"},
        meta={"provider": "fake", "is_delayed": True, "delay_seconds": 900,
              "as_of": "2026-07-05T00:00:00+00:00"},
    )


def fake_tape(*_a, **_k):
    return [
        {"symbol": "^GSPC", "label": "SPX", "last": 5842.13, "chg_pct": 0.62},
        {"symbol": "^VIX", "label": "VIX", "last": 13.42, "chg_pct": -1.2},
    ]


def make_client(tmp_path=None) -> TestClient:
    from stockscan.web.server import create_app

    req = ScanRequest(symbols=["AAA", "BBB", "CCC", "DDD"], timeframe=Timeframe.M5)
    app = create_app(
        req,
        AppConfig(),
        engines=[],
        interval=3600,
        scan_fn=lambda *a, **k: fake_result(),
        tape_fn=fake_tape,
        library_path=(tmp_path / "library.json") if tmp_path else None,
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


def test_scan_endpoint_and_filter(tmp_path):
    with make_client(tmp_path) as client:
        data = wait_for_scan(client)
        assert {r["symbol"] for r in data["rows"]} == {"AAA", "BBB", "DDD"}
        assert data["errors"] == {"CCC": "no data"}

        filtered = client.get("/api/scan", params={"filter": "mb_state >= 1"}).json()
        assert {r["symbol"] for r in filtered["rows"]} == {"AAA", "DDD"}

        bad = client.get("/api/scan", params={"filter": "nope > 1"})
        assert bad.status_code == 400


def test_overview_aggregates(tmp_path):
    with make_client(tmp_path) as client:
        wait_for_scan(client)
        data = client.get("/api/overview").json()
        assert data["breadth"]["advancers"] == 2
        assert data["breadth"]["decliners"] == 1
        assert data["breadth"]["new_highs"] == 1
        assert data["silver"]["net_usd"] == 2.4e9
        assert data["top_signals"][0]["symbol"] == "AAA"
        assert data["top_signals"][0]["signal"] == "Strong Buy"
        assert data["gold"]["regime"] in ("RISK-ON", "NEUTRAL", "RISK-OFF")


def test_tape_endpoint(tmp_path):
    with make_client(tmp_path) as client:
        wait_for_scan(client)
        tape = client.get("/api/tape").json()["tape"]
        assert tape[0]["label"] == "SPX"


def test_scan_status_fields(tmp_path):
    with make_client(tmp_path) as client:
        data = wait_for_scan(client)
        assert "scanning" in data and data["scanning"] is False
        assert data["last_scan_ms"] is not None
        assert data["scan_count"] >= 1


def test_manual_rescan_triggers_new_scan(tmp_path):
    calls = {"n": 0}

    def counting_scan(*_a, **_k):
        calls["n"] += 1
        return fake_result()

    from stockscan.web.server import create_app

    req = ScanRequest(symbols=["AAA", "BBB"], timeframe=Timeframe.M5)
    app = create_app(
        req, AppConfig(), engines=[], interval=3600,
        scan_fn=counting_scan, tape_fn=fake_tape,
        library_path=tmp_path / "lib.json",
    )
    with TestClient(app) as client:
        wait_for_scan(client)
        first = calls["n"]
        resp = client.post("/api/rescan").json()
        assert resp["queued"] is True
        # the forced pull runs on the background thread
        deadline = time.time() + 5
        while time.time() < deadline and calls["n"] <= first:
            time.sleep(0.05)
        assert calls["n"] > first


def test_library_crud_and_persistence(tmp_path):
    with make_client(tmp_path) as client:
        wait_for_scan(client)
        # star + unstar
        assert client.post("/api/library/star/aaa").json() == {"symbol": "AAA", "starred": True}
        lib = client.get("/api/library").json()
        assert lib["starred"] == ["AAA"]
        assert lib["starred_rows"][0]["signal"] == "Strong Buy"

        # saved scans
        client.post(
            "/api/library/scans",
            json={"name": "Gold Bull Breakouts", "filter": "ggr_state >= 2", "sort": "composite"},
        )
        lib = client.get("/api/library").json()
        assert lib["saved_scans"][0]["name"] == "Gold Bull Breakouts"

        assert client.delete("/api/library/scans/Nope").status_code == 404
        assert client.delete("/api/library/scans/Gold Bull Breakouts").status_code == 200

    # persistence across restart (same file)
    with make_client(tmp_path) as client2:
        wait_for_scan(client2)
        assert client2.get("/api/library").json()["starred"] == ["AAA"]


def test_settings_and_index(tmp_path):
    with make_client(tmp_path) as client:
        wait_for_scan(client)
        settings = client.get("/api/settings").json()
        assert settings["is_delayed"] is True
        assert settings["universe"] == 4
        assert settings["alpaca"]["configured"] is False
        page = client.get("/")
        assert page.status_code == 200
        assert "stockscan" in page.text.lower() or "ORE" in page.text
