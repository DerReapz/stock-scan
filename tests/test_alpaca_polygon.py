"""Alpaca + Polygon provider tests via httpx.MockTransport."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from stockscan.models import Timeframe
from stockscan.providers.alpaca_provider import AlpacaProvider
from stockscan.providers.base import NotConfiguredError
from stockscan.providers.polygon_provider import PolygonProvider


def bar_times(n: int, tf: Timeframe = Timeframe.M5) -> list[datetime]:
    # recent bars ending well in the past hour, aligned to the timeframe
    end = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(hours=2)
    step = tf.to_timedelta()
    start = end - step * n
    return [start + step * i for i in range(n)]


def mock_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", patched)


class TestAlpaca:
    def test_missing_keys(self):
        with pytest.raises(NotConfiguredError, match="ALPACA_KEY_ID"):
            AlpacaProvider(key_id="", secret="")

    def test_batch_pagination_and_normalization(self, monkeypatch):
        times = bar_times(30)
        pages = []
        for chunk, token in ((times[:15], "tok2"), (times[15:], None)):
            pages.append(
                {
                    "bars": {
                        sym: [
                            {
                                "t": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 1000,
                            }
                            for t in chunk
                        ]
                        for sym in ("AAPL", "MSFT")
                    },
                    "next_page_token": token,
                }
            )
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(dict(request.url.params))
            page = 1 if request.url.params.get("page_token") else 0
            return httpx.Response(200, json=pages[page])

        mock_client(monkeypatch, handler)
        provider = AlpacaProvider(key_id="k", secret="s", feed="iex")
        frames, errors = provider.get_bars(["AAPL", "MSFT"], Timeframe.M5, 25)
        assert not errors
        assert len(calls) == 2  # pagination followed
        assert calls[0]["feed"] == "iex"
        assert calls[0]["symbols"] == "AAPL,MSFT"
        df = frames["AAPL"]
        assert len(df) == 25  # trimmed to lookback
        assert str(df.index.tz) == "UTC"
        assert "session" in df.columns

    def test_bad_keys_raise(self, monkeypatch):
        def handler(request):
            return httpx.Response(403, text="forbidden")

        mock_client(monkeypatch, handler)
        provider = AlpacaProvider(key_id="k", secret="s")
        with pytest.raises(NotConfiguredError, match="rejected"):
            provider.get_bars(["AAPL"], Timeframe.M5, 10)


class TestPolygon:
    def test_missing_key(self):
        with pytest.raises(NotConfiguredError, match="POLYGON_API_KEY"):
            PolygonProvider(api_key="")

    def test_fetch_and_delay_trim(self, monkeypatch):
        now = datetime.now(UTC)
        times = bar_times(30)
        # append bars inside the 15-min delay window — they must be trimmed
        fresh = [now - timedelta(minutes=10), now - timedelta(minutes=5)]
        all_times = times + fresh

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["apiKey"] == "key"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "t": int(t.timestamp() * 1000),
                            "o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 500,
                        }
                        for t in all_times
                    ]
                },
            )

        mock_client(monkeypatch, handler)
        provider = PolygonProvider(api_key="key", tier="free")
        frames, errors = provider.get_bars(["AAPL"], Timeframe.M5, 100)
        assert not errors
        df = frames["AAPL"]
        cutoff = now - timedelta(minutes=15)
        assert (df.index + Timeframe.M5.to_timedelta() <= cutoff).all()

    def test_error_reported_per_symbol(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            if "BAD" in str(request.url):
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"results": []})

        mock_client(monkeypatch, handler)
        provider = PolygonProvider(api_key="key", tier="paid")
        frames, errors = provider.get_bars(["BAD", "EMPTY"], Timeframe.M5, 10)
        assert frames == {}
        assert "BAD" in errors and "EMPTY" in errors

    def test_free_tier_has_rate_limiter(self):
        assert PolygonProvider(api_key="k", tier="free")._limiter is not None
        assert PolygonProvider(api_key="k", tier="paid")._limiter is None


def test_registry_wires_configs():
    from stockscan.config import AppConfig
    from stockscan.providers import get_provider

    cfg = AppConfig(alpaca_key_id="k", alpaca_secret="s", polygon_api_key="p")
    assert get_provider("alpaca", cfg).capabilities.name == "alpaca"
    assert get_provider("polygon", cfg).capabilities.name == "polygon"
    assert get_provider("yfinance", cfg).capabilities.name == "yfinance"
    with pytest.raises(Exception, match="Unknown provider"):
        get_provider("bogus", cfg)


def test_provider_instances_are_shared():
    """Same config must reuse one instance so rate limiters span every fetch
    in the process (scan loop + tape); different creds get fresh instances."""
    from stockscan.config import AppConfig
    from stockscan.providers import get_provider

    cfg = AppConfig(polygon_api_key="p", alpaca_key_id="k", alpaca_secret="s")
    a = get_provider("polygon", cfg)
    b = get_provider("polygon", cfg)
    assert a is b
    assert a._limiter is b._limiter
    other = get_provider("polygon", AppConfig(polygon_api_key="different"))
    assert other is not a
    assert get_provider("alpaca", cfg) is get_provider("alpaca", cfg)
