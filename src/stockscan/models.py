"""Canonical bar-frame contract and timeframe model."""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum

import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume")
SESSION_VALUES = ("rth", "pre", "post")


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    D1 = "1D"

    @property
    def seconds(self) -> int:
        return {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.H1: 3600,
            Timeframe.D1: 86400,
        }[self]

    @property
    def pandas_freq(self) -> str:
        return {
            Timeframe.M1: "1min",
            Timeframe.M5: "5min",
            Timeframe.M15: "15min",
            Timeframe.H1: "1h",
            Timeframe.D1: "1D",
        }[self]

    @property
    def is_intraday(self) -> bool:
        return self is not Timeframe.D1

    def to_timedelta(self) -> timedelta:
        return timedelta(seconds=self.seconds)

    @classmethod
    def parse(cls, text: str) -> Timeframe:
        normalized = text.strip().lower()
        aliases = {
            "1d": cls.D1, "d": cls.D1, "1day": cls.D1, "day": cls.D1, "daily": cls.D1,
            "60m": cls.H1, "1h": cls.H1, "h": cls.H1, "hour": cls.H1,
        }
        if normalized in aliases:
            return aliases[normalized]
        for tf in cls:
            if tf.value.lower() == normalized:
                return tf
        raise ValueError(
            f"Unknown timeframe {text!r}; expected one of {[tf.value for tf in cls]}"
        )


class BarValidationError(ValueError):
    pass


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the canonical bar-frame contract; returns the frame unchanged."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise BarValidationError("bar frame index must be a DatetimeIndex")
    if df.index.tz is None:
        raise BarValidationError("bar frame index must be timezone-aware (UTC)")
    if str(df.index.tz) != "UTC":
        raise BarValidationError(f"bar frame index must be UTC, got {df.index.tz}")
    if not df.index.is_monotonic_increasing:
        raise BarValidationError("bar frame index must be sorted ascending")
    missing = [c for c in (*BAR_COLUMNS, "session") if c not in df.columns]
    if missing:
        raise BarValidationError(f"bar frame missing columns: {missing}")
    bad_sessions = set(df["session"].unique()) - set(SESSION_VALUES)
    if bad_sessions:
        raise BarValidationError(f"invalid session values: {sorted(bad_sessions)}")
    return df
