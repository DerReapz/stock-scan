from datetime import UTC, datetime

import pandas as pd

from stockscan.models import BarValidationError, Timeframe, validate_bars
from stockscan.sessions import drop_incomplete_last_bar, tag_sessions

from .conftest import make_bars


class TestTagSessions:
    def test_rth_boundaries(self):
        idx = pd.DatetimeIndex(
            [
                "2026-06-01 09:25",  # pre
                "2026-06-01 09:30",  # rth (inclusive start)
                "2026-06-01 15:55",  # rth
                "2026-06-01 16:00",  # post (exclusive end)
                "2026-06-01 04:00",  # pre
                "2026-06-01 19:55",  # post
            ]
        ).tz_localize("America/New_York").tz_convert("UTC")
        df = pd.DataFrame(
            {c: 1.0 for c in ("open", "high", "low", "close", "volume")}, index=idx
        )
        out = tag_sessions(df, Timeframe.M5)
        assert out["session"].tolist() == ["pre", "rth", "rth", "post", "pre", "post"]

    def test_daily_always_rth(self):
        df = make_bars(10, timeframe=Timeframe.D1)
        assert (df["session"] == "rth").all()


class TestDropIncompleteLastBar:
    def test_drops_open_bar(self):
        df = make_bars(10)
        last_start = df.index[-1]
        during = last_start.to_pydatetime() + Timeframe.M5.to_timedelta() / 2
        out = drop_incomplete_last_bar(df, Timeframe.M5, now=during)
        assert len(out) == 9

    def test_keeps_closed_bar(self):
        df = make_bars(10)
        after = df.index[-1].to_pydatetime() + Timeframe.M5.to_timedelta()
        out = drop_incomplete_last_bar(df, Timeframe.M5, now=after)
        assert len(out) == 10

    def test_daily_completes_at_close(self):
        df = make_bars(5, timeframe=Timeframe.D1)
        day = df.index[-1]
        noon_et = datetime(day.year, day.month, day.day, 16, 0, tzinfo=UTC)  # 12:00 ET
        assert len(drop_incomplete_last_bar(df, Timeframe.D1, now=noon_et)) == 4
        evening = datetime(day.year, day.month, day.day, 21, 0, tzinfo=UTC)  # 17:00 ET
        assert len(drop_incomplete_last_bar(df, Timeframe.D1, now=evening)) == 5


class TestValidateBars:
    def test_accepts_canonical(self):
        validate_bars(make_bars(20))

    def test_rejects_naive_index(self):
        df = make_bars(5)
        df.index = df.index.tz_localize(None)
        try:
            validate_bars(df)
            raise AssertionError("should have raised")
        except BarValidationError:
            pass

    def test_timeframe_parse(self):
        assert Timeframe.parse("5M") is Timeframe.M5
        assert Timeframe.parse("1d") is Timeframe.D1
        assert Timeframe.parse("60m") is Timeframe.H1
