"""Provider tests: yfinance normalization via monkeypatched download."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from stockscan.models import Timeframe, validate_bars
from stockscan.providers.base import RateLimiter
from stockscan.providers.yfinance_provider import YFinanceProvider, _period_for


def fake_yf_frame(symbols: list[str], n: int = 50) -> pd.DataFrame:
    """Mimic yf.download(group_by='ticker') output: MultiIndex columns,
    Eastern-localized intraday index, title-case column names."""
    idx = pd.date_range(
        "2026-06-01 09:30", periods=n, freq="5min", tz="America/New_York"
    )
    frames = {}
    for sym in symbols:
        base = 100.0 + hash(sym) % 50
        close = pd.Series(range(n), index=idx) * 0.1 + base
        frames[sym] = pd.DataFrame(
            {
                "Open": close - 0.05,
                "High": close + 0.1,
                "Low": close - 0.1,
                "Close": close,
                "Adj Close": close,
                "Volume": 1000.0,
            },
            index=idx,
        )
    return pd.concat(frames, axis=1)


@pytest.fixture
def provider(monkeypatch):
    p = YFinanceProvider()
    frame = fake_yf_frame(["AAPL", "MSFT"])

    def fake_download(tickers, **kwargs):
        fake_download.calls.append((tuple(tickers), kwargs))
        return frame

    fake_download.calls = []
    import yfinance

    monkeypatch.setattr(yfinance, "download", fake_download)
    return p, fake_download


def test_batched_normalized_output(provider):
    p, fake = provider
    now = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)  # after all bars closed
    frames, errors = p.get_bars(["AAPL", "MSFT"], Timeframe.M5, 40)
    assert not errors
    assert set(frames) == {"AAPL", "MSFT"}
    df = frames["AAPL"]
    validate_bars(df)
    assert str(df.index.tz) == "UTC"
    assert len(df) == 40  # trimmed to lookback
    assert (df["session"] == "rth").all()
    assert len(fake.calls) == 1  # one batched call for both symbols
    del now


def test_prepost_flag_controls_premarket(provider):
    """include_extended must reach yfinance as prepost — the pre-market switch."""
    p, fake = provider
    p.get_bars(["AAPL"], Timeframe.M5, 10, include_extended=True)
    assert fake.calls[-1][1]["prepost"] is True
    p.get_bars(["AAPL"], Timeframe.M5, 10, include_extended=False)
    assert fake.calls[-1][1]["prepost"] is False


def test_unknown_symbol_reports_error(provider):
    p, _ = provider
    frames, errors = p.get_bars(["AAPL", "NOPE"], Timeframe.M5, 40)
    assert "AAPL" in frames
    assert "NOPE" in errors


def test_incomplete_last_bar_dropped(monkeypatch):
    p = YFinanceProvider()
    # last bar starts "now" → still open → must be dropped
    start = pd.Timestamp.now(tz="America/New_York").floor("5min") - pd.Timedelta(minutes=45)
    idx = pd.date_range(start, periods=10, freq="5min")
    df = pd.DataFrame(
        {
            "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5,
            "Adj Close": 1.5, "Volume": 100.0,
        },
        index=idx,
    )
    frame = pd.concat({"AAPL": df}, axis=1)
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda tickers, **kw: frame)
    frames, _ = p.get_bars(["AAPL"], Timeframe.M5, 100)
    assert len(frames["AAPL"]) == 9


def test_lookback_clamped_to_provider_cap():
    p = YFinanceProvider()
    assert p.clamp_lookback(Timeframe.M1, 100_000) <= 7 * 24 * 60
    assert _period_for(Timeframe.M1, 2000).endswith("d")


def test_rate_limiter_spacing():
    rl = RateLimiter(600)  # 10/sec → no real waiting in tests
    import time

    t0 = time.monotonic()
    for _ in range(5):
        rl.acquire()
    assert time.monotonic() - t0 < 1.0
