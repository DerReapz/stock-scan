"""Market-data provider contract: live (paid) and delayed (free) feeds behind
one interface. Providers return canonical bar frames (UTC index, session
column, completed bars only) so the engines never care where data came from.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from ..models import Timeframe


class ProviderError(RuntimeError):
    pass


class NotConfiguredError(ProviderError):
    """Raised when a provider needs credentials that are not set."""


@dataclass(frozen=True)
class ProviderCapabilities:
    name: str
    is_delayed: bool
    delay_seconds: int = 0
    supports_extended_hours: bool = True
    supports_batch: bool = False
    max_requests_per_minute: int | None = None
    max_lookback: dict[Timeframe, timedelta] = field(default_factory=dict)
    note: str = ""


class RateLimiter:
    """Token bucket; acquire() blocks until a request slot is available."""

    def __init__(self, requests_per_minute: int):
        self.capacity = max(1, requests_per_minute)
        self.tokens = float(self.capacity)
        self.fill_rate = self.capacity / 60.0
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.fill_rate)
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.fill_rate
            time.sleep(wait)


class MarketDataProvider(ABC):
    capabilities: ProviderCapabilities

    @abstractmethod
    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        lookback: int,
        *,
        include_extended: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Fetch at least ``lookback`` completed bars per symbol where the
        provider allows it. Returns (bar frames by symbol, error text for
        symbols that failed)."""

    def clamp_lookback(self, timeframe: Timeframe, lookback: int) -> int:
        cap = self.capabilities.max_lookback.get(timeframe)
        if cap is None:
            return lookback
        max_bars = int(cap.total_seconds() // timeframe.seconds)
        if not timeframe.is_intraday:
            max_bars = cap.days
        return min(lookback, max_bars)


def fetch_per_symbol(
    symbols: Sequence[str],
    fetch_one: Callable[[str], pd.DataFrame],
    *,
    rate_limiter: RateLimiter | None = None,
    max_workers: int = 8,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Run per-symbol fetches concurrently, respecting an optional rate limit.
    Returns (frames, errors-by-symbol)."""

    def task(symbol: str) -> tuple[str, pd.DataFrame | None, str | None]:
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            return symbol, fetch_one(symbol), None
        except Exception as exc:  # noqa: BLE001 — reported per symbol
            return symbol, None, f"{type(exc).__name__}: {exc}"

    workers = 1 if rate_limiter is not None else max_workers
    out: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(symbols)))) as pool:
        for symbol, df, err in pool.map(task, symbols):
            if err is not None:
                errors[symbol] = err
            elif df is None or len(df) == 0:
                errors[symbol] = "no data returned"
            else:
                out[symbol] = df
    return out, errors
