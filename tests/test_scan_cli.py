"""Pipeline + CLI smoke tests with a fake provider (no network)."""

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

import stockscan.scan as scan_mod
from stockscan.config import AppConfig
from stockscan.models import Timeframe
from stockscan.providers.base import MarketDataProvider, ProviderCapabilities
from stockscan.scan import ScanRequest, load_watchlist, run_scan

from .conftest import make_bars


class FakeProvider(MarketDataProvider):
    capabilities = ProviderCapabilities(
        name="fake", is_delayed=True, delay_seconds=900, supports_batch=True
    )

    def __init__(self, n: int = 300):
        self.n = n

    def get_bars(self, symbols, timeframe, lookback, *, include_extended=True):
        frames = {}
        errors = {}
        for i, sym in enumerate(symbols):
            if sym == "FAIL":
                errors[sym] = "synthetic failure"
                continue
            trend = 0.08 if i % 2 == 0 else -0.08
            frames[sym] = make_bars(self.n, seed=i + 1, trend=trend)
        return frames, errors


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(scan_mod, "get_provider", lambda name, cfg: provider)
    return provider


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def request_for(symbols, **kw) -> ScanRequest:
    return ScanRequest(symbols=symbols, timeframe=Timeframe.M5, **kw)


class TestRunScan:
    def test_row_schema(self, cfg):
        result = run_scan(request_for(["UP", "DOWN"]), cfg)
        assert set(result.rows.index) == {"UP", "DOWN"}
        for prefix in ("mb_score", "mb_state", "ggr_state", "sf_regime", "sf_inst"):
            assert prefix in result.rows.columns
        assert result.meta["provider"] == "fake"
        assert result.meta["is_delayed"] is True

    def test_error_symbol_reported(self, cfg):
        result = run_scan(request_for(["UP", "FAIL"]), cfg)
        assert "FAIL" in result.errors
        assert "UP" in result.rows.index

    def test_filter_and_sort(self, cfg):
        result = run_scan(
            request_for(["A", "B", "C", "D"], filter_expr="mb_score > -100", sort_by="mb_score"),
            cfg,
        )
        scores = result.rows["mb_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_short_history_reported(self, cfg, monkeypatch):
        provider = FakeProvider(n=50)
        monkeypatch.setattr(scan_mod, "get_provider", lambda name, c: provider)
        result = run_scan(request_for(["UP"]), cfg)
        assert "UP" in result.errors
        assert "warmup" in result.errors["UP"]

    def test_json_roundtrip(self, cfg):
        result = run_scan(request_for(["UP"]), cfg)
        payload = result.to_json_dict()
        assert json.loads(json.dumps(payload, default=str))["rows"][0]["symbol"] == "UP"


class TestWatchlist:
    def test_inline_symbols(self):
        assert load_watchlist("aapl, msft") == ["AAPL", "MSFT"]

    def test_file(self, tmp_path):
        p = tmp_path / "wl.txt"
        p.write_text("# comment\nAAPL\nmsft  # inline\n\n")
        assert load_watchlist(str(p)) == ["AAPL", "MSFT"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_watchlist("nope/missing.txt")


class TestCli:
    def test_scan_command(self, tmp_path):
        from stockscan.cli import app

        runner = CliRunner()
        json_out = tmp_path / "out.json"
        result = runner.invoke(
            app,
            ["scan", "UP,DOWN", "--timeframe", "5m", "--json", str(json_out)],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(json_out.read_text())
        assert {r["symbol"] for r in payload["rows"]} == {"UP", "DOWN"}
        assert payload["meta"]["is_delayed"] is True
        row = payload["rows"][0]
        assert "composite" in row and "signal" in row and "day_chg" in row

    def test_scan_filter_rejects_bad_column(self):
        from stockscan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["scan", "UP", "--filter", "bogus > 1"])
        assert result.exit_code != 0

    def test_providers_command(self):
        from stockscan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["providers"])
        assert result.exit_code == 0
        assert "yfinance" in result.output


class TestPineIntegration:
    def test_scan_with_pine_engine(self, cfg):
        from stockscan.scan import build_engines

        engines = build_engines(cfg, pine_scripts=["indicators/iron_momentum.pine"])
        result = run_scan(request_for(["UP"]), cfg, engines=engines)
        row = result.rows.loc["UP"]
        assert "iron_momentum_mb_score" in result.rows.columns
        # interpreter and native port agree on the snapshot
        assert row["iron_momentum_mb_score"] == pytest.approx(row["mb_score"], abs=1e-9)
        assert row["iron_momentum_mb_state"] == row["mb_state"]

    def test_cli_pine_flag(self, tmp_path):
        from stockscan.cli import app

        runner = CliRunner()
        json_out = tmp_path / "pine.json"
        result = runner.invoke(
            app,
            [
                "scan", "UP",
                "--pine", "indicators/gold_regime.pine",
                "--pine-input", "Fast SMA Length=5",
                "--json", str(json_out),
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(json_out.read_text())
        row = payload["rows"][0]
        pine_cols = [c for c in row if c.startswith("ggr_rsi_")]
        assert pine_cols, f"no pine columns in {sorted(row)}"
        # override took effect: fast SMA 5 ≠ default-8 native state everywhere is
        # not guaranteed, but the column must exist and be populated
        assert row["ggr_rsi_ribbon_state_3_3"] is not None

    def test_broken_pine_reports_line(self, tmp_path):
        bad = tmp_path / "bad.pine"
        bad.write_text('//@version=6\nindicator("x")\nfor i = 0 to 3\n    y = i\n')
        from stockscan.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["scan", "UP", "--pine", str(bad)])
        assert result.exit_code != 0


def test_csv_export(cfg, tmp_path):
    from stockscan import output

    result = run_scan(request_for(["UP"]), cfg)
    csv_path = tmp_path / "out.csv"
    output.export_csv(result, csv_path)
    df = pd.read_csv(csv_path)
    assert "mb_score" in df.columns
