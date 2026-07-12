"""Provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    MarketDataProvider,
    NotConfiguredError,
    ProviderCapabilities,
    ProviderError,
    RateLimiter,
)

if TYPE_CHECKING:
    from ..config import AppConfig

PROVIDER_NAMES = ("yfinance", "alpaca", "polygon")

# One provider instance per configuration: rate limiters (Polygon free tier is
# 5 requests/minute) must be shared across every fetch in the process — a fresh
# instance per scan cycle would get a fresh token bucket and burst past the cap
# when the scan and the index-tape fetch run back to back.
_INSTANCES: dict[tuple, MarketDataProvider] = {}


def get_provider(name: str, cfg: AppConfig) -> MarketDataProvider:
    name = name.lower()
    if name == "yfinance":
        key: tuple = ("yfinance",)
    elif name == "alpaca":
        key = ("alpaca", cfg.alpaca_key_id, cfg.alpaca_secret, cfg.alpaca_feed)
    elif name == "polygon":
        key = ("polygon", cfg.polygon_api_key, cfg.polygon_tier)
    else:
        raise ProviderError(f"Unknown provider {name!r}; expected one of {PROVIDER_NAMES}")

    provider = _INSTANCES.get(key)
    if provider is None:
        provider = _build_provider(name, cfg)
        _INSTANCES[key] = provider
    return provider


def _build_provider(name: str, cfg: AppConfig) -> MarketDataProvider:
    if name == "yfinance":
        from .yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    if name == "alpaca":
        from .alpaca_provider import AlpacaProvider

        return AlpacaProvider(
            key_id=cfg.alpaca_key_id, secret=cfg.alpaca_secret, feed=cfg.alpaca_feed
        )
    from .polygon_provider import PolygonProvider

    return PolygonProvider(api_key=cfg.polygon_api_key, tier=cfg.polygon_tier)


__all__ = [
    "PROVIDER_NAMES",
    "MarketDataProvider",
    "NotConfiguredError",
    "ProviderCapabilities",
    "ProviderError",
    "RateLimiter",
    "get_provider",
]
