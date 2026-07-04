"""Shared synthetic OHLCV fixture builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockscan.models import Timeframe
from stockscan.sessions import tag_sessions


def make_bars(
    n: int = 300,
    *,
    timeframe: Timeframe = Timeframe.M5,
    start: str = "2026-06-01 13:30",  # 09:30 Eastern in UTC (EDT)
    seed: int = 7,
    trend: float = 0.02,
    vol: float = 0.5,
    base: float = 100.0,
    include_extended: bool = False,
) -> pd.DataFrame:
    """Deterministic pseudo-random walk with plausible OHLC relationships."""
    rng = np.random.default_rng(seed)
    if include_extended and timeframe.is_intraday:
        # Full trading day: 04:00 pre -> 20:00 post Eastern, weekdays only.
        idx = _intraday_index(n, timeframe, session_start="04:00", session_end="20:00")
    elif timeframe.is_intraday:
        idx = _intraday_index(n, timeframe, session_start="09:30", session_end="16:00")
    else:
        idx = pd.bdate_range("2026-01-02", periods=n, tz="UTC")
    steps = rng.normal(trend, vol, n)
    close = base + np.cumsum(steps)
    open_ = np.concatenate([[base], close[:-1]])
    spread = np.abs(rng.normal(0, vol / 2, n)) + 0.01
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = np.abs(rng.normal(1_000_000, 250_000, n)) + 1_000
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx[:n],
    )
    return tag_sessions(df, timeframe)


def _intraday_index(
    n: int, timeframe: Timeframe, session_start: str, session_end: str
) -> pd.DatetimeIndex:
    days = pd.bdate_range("2026-06-01", periods=max(2, n // 4 + 2))
    stamps: list[pd.Timestamp] = []
    for day in days:
        times = pd.date_range(
            f"{day.date()} {session_start}",
            f"{day.date()} {session_end}",
            freq=timeframe.pandas_freq,
            inclusive="left",
            tz="America/New_York",
        )
        stamps.extend(times)
        if len(stamps) >= n:
            break
    return pd.DatetimeIndex(stamps[:n]).tz_convert("UTC")


@pytest.fixture
def bars_rth() -> pd.DataFrame:
    return make_bars(300)


@pytest.fixture
def bars_extended() -> pd.DataFrame:
    return make_bars(400, include_extended=True)


@pytest.fixture
def bars_daily() -> pd.DataFrame:
    return make_bars(300, timeframe=Timeframe.D1)
