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


def get_provider(name: str, cfg: AppConfig) -> MarketDataProvider:
    name = name.lower()
    if name == "yfinance":
        from .yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    if name == "alpaca":
        from .alpaca_provider import AlpacaProvider

        return AlpacaProvider(
            key_id=cfg.alpaca_key_id, secret=cfg.alpaca_secret, feed=cfg.alpaca_feed
        )
    if name == "polygon":
        from .polygon_provider import PolygonProvider

        return PolygonProvider(api_key=cfg.polygon_api_key, tier=cfg.polygon_tier)
    raise ProviderError(f"Unknown provider {name!r}; expected one of {PROVIDER_NAMES}")


__all__ = [
    "PROVIDER_NAMES",
    "MarketDataProvider",
    "NotConfiguredError",
    "ProviderCapabilities",
    "ProviderError",
    "RateLimiter",
    "get_provider",
]
