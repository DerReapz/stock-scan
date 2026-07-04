"""Polygon.io aggregates provider.

Free tier: 15-minute delayed data, 5 requests/minute (token-bucket limited,
sequential fetching, bars newer than the delay are trimmed). Paid tiers lift
both restrictions — set ``tier = "paid"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from ..models import Timeframe, validate_bars
from ..sessions import drop_incomplete_last_bar, tag_sessions
from .base import (
    MarketDataProvider,
    NotConfiguredError,
    ProviderCapabilities,
    ProviderError,
    RateLimiter,
    fetch_per_symbol,
)

BASE_URL = "https://api.polygon.io"
FREE_DELAY_SECONDS = 15 * 60

_RANGES = {
    Timeframe.M1: (1, "minute"),
    Timeframe.M5: (5, "minute"),
    Timeframe.M15: (15, "minute"),
    Timeframe.H1: (1, "hour"),
    Timeframe.D1: (1, "day"),
}


class PolygonProvider(MarketDataProvider):
    def __init__(self, api_key: str, tier: str = "free", timeout: float = 30.0):
        if not api_key:
            raise NotConfiguredError(
                "Polygon needs POLYGON_API_KEY (see .env.example). "
                "Free tier is 15-min delayed at 5 requests/minute."
            )
        if tier not in ("free", "paid"):
            raise ProviderError(f"polygon tier must be 'free' or 'paid', got {tier!r}")
        self.api_key = api_key
        self.tier = tier
        self.timeout = timeout
        self._limiter = RateLimiter(5) if tier == "free" else None
        self.capabilities = ProviderCapabilities(
            name="polygon",
            is_delayed=tier == "free",
            delay_seconds=FREE_DELAY_SECONDS if tier == "free" else 0,
            supports_extended_hours=True,
            supports_batch=False,
            max_requests_per_minute=5 if tier == "free" else None,
            note="free tier: 15-min delay, 5 req/min" if tier == "free" else "paid tier",
        )

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        lookback: int,
        *,
        include_extended: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        def fetch_one(symbol: str) -> pd.DataFrame:
            return self._fetch_symbol(symbol, timeframe, lookback, include_extended)

        return fetch_per_symbol(symbols, fetch_one, rate_limiter=self._limiter)

    def _fetch_symbol(
        self, symbol: str, timeframe: Timeframe, lookback: int, include_extended: bool
    ) -> pd.DataFrame:
        mult, unit = _RANGES[timeframe]
        now = datetime.now(UTC)
        start = _start_for(timeframe, lookback, now)
        url = (
            f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{mult}/{unit}/"
            f"{start.strftime('%Y-%m-%d')}/{now.strftime('%Y-%m-%d')}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": self.api_key}
        results: list[dict] = []
        with httpx.Client(timeout=self.timeout) as client:
            while url:
                response = client.get(url, params=params)
                if response.status_code in (401, 403):
                    raise NotConfiguredError(
                        f"Polygon rejected the API key ({response.status_code}): "
                        f"{response.text[:200]}"
                    )
                if response.status_code == 429:
                    raise ProviderError(
                        "Polygon rate limit hit (free tier is 5 requests/minute)"
                    )
                response.raise_for_status()
                payload = response.json()
                results.extend(payload.get("results") or [])
                url = payload.get("next_url")
                params = {"apiKey": self.api_key}  # next_url carries the query
                if url and self._limiter is not None:
                    self._limiter.acquire()
        if not results:
            return pd.DataFrame()
        df = _to_frame(results, timeframe, include_extended)
        if self.tier == "free":
            cutoff = now - timedelta(seconds=FREE_DELAY_SECONDS)
            df = df[df.index + timeframe.to_timedelta() <= cutoff]
        return df.iloc[-lookback:] if lookback else df


def _start_for(timeframe: Timeframe, lookback: int, now: datetime) -> datetime:
    if timeframe.is_intraday:
        bars_per_day = max(1, int(16 * 3600 // timeframe.seconds))  # 04:00-20:00 ET
        days = int(lookback / bars_per_day * 1.9) + 3
    else:
        days = int(lookback * 1.6) + 10
    return now - timedelta(days=days)


def _to_frame(results: list[dict], timeframe: Timeframe, include_extended: bool) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.rename(
        columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = tag_sessions(df, timeframe)
    if not include_extended:
        df = df[df["session"] == "rth"]
    df = drop_incomplete_last_bar(df, timeframe)
    return validate_bars(df)
