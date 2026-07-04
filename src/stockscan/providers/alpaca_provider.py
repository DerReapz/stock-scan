"""Alpaca Market Data v2 provider.

One API covers both feed classes: ``feed=iex`` (free API keys, IEX-only
consolidated tape) and ``feed=sip`` (paid subscription, full consolidated
feed). Multi-symbol batch requests with page_token pagination.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd

from ..models import Timeframe, validate_bars
from ..sessions import drop_incomplete_last_bar, tag_sessions
from .base import MarketDataProvider, NotConfiguredError, ProviderCapabilities, ProviderError

BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

_TIMEFRAMES = {
    Timeframe.M1: "1Min",
    Timeframe.M5: "5Min",
    Timeframe.M15: "15Min",
    Timeframe.H1: "1Hour",
    Timeframe.D1: "1Day",
}


class AlpacaProvider(MarketDataProvider):
    def __init__(self, key_id: str, secret: str, feed: str = "iex", timeout: float = 30.0):
        if not key_id or not secret:
            raise NotConfiguredError(
                "Alpaca needs ALPACA_KEY_ID and ALPACA_SECRET (see .env.example). "
                "Free keys use feed=iex; a paid data subscription unlocks feed=sip."
            )
        if feed not in ("iex", "sip"):
            raise ProviderError(f"alpaca feed must be 'iex' or 'sip', got {feed!r}")
        self.feed = feed
        self.timeout = timeout
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
        }
        self.capabilities = ProviderCapabilities(
            name="alpaca",
            is_delayed=False,
            delay_seconds=0,
            supports_extended_hours=True,
            supports_batch=True,
            max_requests_per_minute=200,
            note=f"feed={feed}"
            + ("" if feed == "sip" else " (IEX only — real-time but thinner than SIP)"),
        )

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: Timeframe,
        lookback: int,
        *,
        include_extended: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        start = _start_for(timeframe, lookback)
        params: dict[str, str] = {
            "symbols": ",".join(symbols),
            "timeframe": _TIMEFRAMES[timeframe],
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": "10000",
            "adjustment": "raw",
            "feed": self.feed,
        }
        raw_bars: dict[str, list[dict]] = {s: [] for s in symbols}
        errors: dict[str, str] = {}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                page_token: str | None = None
                while True:
                    page_params = dict(params)
                    if page_token:
                        page_params["page_token"] = page_token
                    response = client.get(BASE_URL, params=page_params, headers=self._headers)
                    if response.status_code in (401, 403):
                        raise NotConfiguredError(
                            f"Alpaca rejected the API keys ({response.status_code}): "
                            f"{response.text[:200]}"
                        )
                    response.raise_for_status()
                    payload = response.json()
                    for symbol, bars in (payload.get("bars") or {}).items():
                        raw_bars.setdefault(symbol, []).extend(bars)
                    page_token = payload.get("next_page_token")
                    if not page_token:
                        break
        except NotConfiguredError:
            raise
        except httpx.HTTPError as exc:
            return {}, {s: f"alpaca request failed: {exc}" for s in symbols}

        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            bars = raw_bars.get(symbol) or []
            if not bars:
                errors[symbol] = "no data returned"
                continue
            df = _to_frame(bars, timeframe, include_extended)
            if len(df) == 0:
                errors[symbol] = "no completed bars"
                continue
            out[symbol] = df.iloc[-lookback:] if lookback else df
        return out, errors


def _start_for(timeframe: Timeframe, lookback: int) -> datetime:
    if timeframe.is_intraday:
        # ~6.5h RTH (+4h ETH headroom) per trading day; pad for weekends
        bars_per_day = max(1, int(6.5 * 3600 // timeframe.seconds))
        days = int(lookback / bars_per_day * 1.7) + 3
    else:
        days = int(lookback * 1.6) + 10
    return datetime.now(UTC) - timedelta(days=days)


def _to_frame(bars: list[dict], timeframe: Timeframe, include_extended: bool) -> pd.DataFrame:
    df = pd.DataFrame(bars)
    df = df.rename(
        columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = tag_sessions(df, timeframe)
    if not include_extended:
        df = df[df["session"] == "rth"]
    df = drop_incomplete_last_bar(df, timeframe)
    return validate_bars(df)
