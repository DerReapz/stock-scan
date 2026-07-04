"""Yahoo Finance provider — zero-key delayed feed, the out-of-the-box default.

Unofficial API; data is delayed and intraday history is capped (1m ≈ 7 days,
5m/15m/1h ≈ 60 days). One batched download covers the whole watchlist.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta

import pandas as pd

from ..models import BAR_COLUMNS, Timeframe, validate_bars
from ..sessions import drop_incomplete_last_bar, tag_sessions
from .base import MarketDataProvider, ProviderCapabilities

_INTERVALS = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "60m",
    Timeframe.D1: "1d",
}

_MAX_LOOKBACK = {
    Timeframe.M1: timedelta(days=7),
    Timeframe.M5: timedelta(days=59),
    Timeframe.M15: timedelta(days=59),
    Timeframe.H1: timedelta(days=729),
}


class YFinanceProvider(MarketDataProvider):
    capabilities = ProviderCapabilities(
        name="yfinance",
        is_delayed=True,
        delay_seconds=900,
        supports_extended_hours=True,
        supports_batch=True,
        max_requests_per_minute=None,
        max_lookback=_MAX_LOOKBACK,
        note="Unofficial Yahoo Finance data; no API key required.",
    )

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        lookback: int,
        *,
        include_extended: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        import yfinance as yf

        lookback = self.clamp_lookback(timeframe, lookback)
        period = _period_for(timeframe, lookback)
        raw = yf.download(
            tickers=list(symbols),
            interval=_INTERVALS[timeframe],
            period=period,
            prepost=include_extended,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        out: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}
        for symbol in symbols:
            try:
                df = _extract_symbol(raw, symbol, len(symbols) == 1)
            except Exception as exc:  # noqa: BLE001 — reported per symbol
                errors[symbol] = f"{type(exc).__name__}: {exc}"
                continue
            if df is None or df.empty:
                errors[symbol] = "no data returned"
                continue
            df = _normalize(df, timeframe)
            if len(df) == 0:
                errors[symbol] = "no completed bars"
                continue
            out[symbol] = df.iloc[-lookback:] if lookback else df
        return out, errors


def _period_for(timeframe: Timeframe, lookback: int) -> str:
    """Smallest yfinance period string that covers `lookback` bars, padding
    generously for weekends/holidays/session gaps."""
    if timeframe is Timeframe.D1:
        days = int(lookback * 1.6) + 10
        return f"{days}d" if days <= 730 else "max"
    bars_per_day = {
        Timeframe.M1: 390,
        Timeframe.M5: 78,
        Timeframe.M15: 26,
        Timeframe.H1: 7,
    }[timeframe]
    trading_days = lookback / bars_per_day
    days = int(trading_days * 1.7) + 4
    cap = _MAX_LOOKBACK[timeframe].days
    return f"{min(days, cap)}d"


def _extract_symbol(raw: pd.DataFrame, symbol: str, single: bool) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol not in raw.columns.get_level_values(0):
            return None
        df = raw[symbol].copy()
    elif single:
        df = raw.copy()
    else:
        return None
    return df.dropna(how="all")


def _normalize(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    df = df.rename(columns=str.lower)[list(BAR_COLUMNS)].copy()
    if df.index.tz is None:
        # daily bars come back tz-naive, labeled by trading date
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    for col in ("open", "high", "low"):
        df[col] = df[col].astype(float)
    df = tag_sessions(df, timeframe)
    df = drop_incomplete_last_bar(df, timeframe)
    return validate_bars(df)
