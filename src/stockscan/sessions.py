"""US-equity session tagging and completed-bar enforcement.

Bars are stored in UTC; session reasoning happens in US/Eastern. Regular
trading hours (RTH) are weekdays 09:30-16:00 Eastern. Half-days and market
holidays are not modelled yet (an early-close afternoon is tagged rth) —
acceptable for scanning, documented in the README.
"""

from __future__ import annotations

from datetime import UTC, datetime, time

import pandas as pd

from .models import Timeframe

EASTERN = "America/New_York"
RTH_START = time(9, 30)
RTH_END = time(16, 0)


def tag_sessions(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Add the ``session`` column (rth/pre/post) keyed on the bar *start* time."""
    out = df.copy()
    if not timeframe.is_intraday or len(df) == 0:
        out["session"] = "rth"
        return out
    eastern = out.index.tz_convert(EASTERN)
    minutes = eastern.hour * 60 + eastern.minute
    rth = (minutes >= RTH_START.hour * 60 + RTH_START.minute) & (
        minutes < RTH_END.hour * 60 + RTH_END.minute
    )
    pre = minutes < RTH_START.hour * 60 + RTH_START.minute
    out["session"] = pd.Series("post", index=out.index).mask(rth, "rth").mask(pre, "pre")
    return out


def drop_incomplete_last_bar(
    df: pd.DataFrame, timeframe: Timeframe, now: datetime | None = None
) -> pd.DataFrame:
    """Drop the final bar if its interval has not fully elapsed yet.

    Providers label bars by start time; a bar is complete once
    ``start + timeframe`` is in the past. Daily bars complete at 16:00 Eastern
    rather than start+24h.
    """
    if len(df) == 0:
        return df
    now = now or datetime.now(UTC)
    last_start = df.index[-1]
    if timeframe.is_intraday:
        complete = last_start + timeframe.to_timedelta() <= now
    else:
        # Providers label daily bars either at midnight UTC or midnight
        # Eastern of the trading date. Midnight UTC converts to the previous
        # evening in Eastern — roll forward to recover the trading date.
        start_et = last_start.tz_convert(EASTERN)
        if start_et.hour >= 17:
            start_et = start_et + pd.Timedelta(days=1)
        close_et = start_et.replace(
            hour=RTH_END.hour, minute=RTH_END.minute, second=0, microsecond=0
        )
        complete = close_et.tz_convert("UTC") <= now
    return df if complete else df.iloc[:-1]
