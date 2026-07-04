"""Configuration: TOML file + .env / environment + CLI overrides.

Precedence (highest wins): CLI flags > environment variables > stockscan.toml
> built-in defaults. Engine sections map 1:1 onto the engine config
dataclasses; unknown keys fail loudly.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .engines import (
    GoldRegimeConfig,
    IronMomentumConfig,
    SilverFlowConfig,
)
from .models import Timeframe

DEFAULT_CONFIG_FILENAME = "stockscan.toml"


@dataclass(frozen=True)
class ScanDefaults:
    provider: str = "yfinance"
    timeframe: str = "5m"
    lookback: int = 300
    include_extended: bool = True
    filter: str = ""
    sort: str = "mb_score"
    descending: bool = True
    limit: int = 0


@dataclass(frozen=True)
class AppConfig:
    scan: ScanDefaults = field(default_factory=ScanDefaults)
    iron_momentum: IronMomentumConfig = field(default_factory=IronMomentumConfig)
    silver_flow: SilverFlowConfig = field(default_factory=SilverFlowConfig)
    gold_regime: GoldRegimeConfig = field(default_factory=GoldRegimeConfig)
    pine_scripts: list[str] = field(default_factory=list)
    pine_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    alpaca_key_id: str = ""
    alpaca_secret: str = ""
    alpaca_feed: str = "iex"
    polygon_api_key: str = ""
    polygon_tier: str = "free"

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        load_dotenv()
        data: dict[str, Any] = {}
        config_path = Path(path) if path else Path(DEFAULT_CONFIG_FILENAME)
        if config_path.is_file():
            with open(config_path, "rb") as fh:
                data = tomllib.load(fh)
        elif path is not None:
            raise FileNotFoundError(f"config file not found: {path}")

        scan = _build(ScanDefaults, data.get("scan", {}), "scan")
        engines = data.get("engines", {})
        iron = _build(IronMomentumConfig, engines.get("iron_momentum", {}), "engines.iron_momentum")
        silver = _build(SilverFlowConfig, engines.get("silver_flow", {}), "engines.silver_flow")
        gold_raw = dict(engines.get("gold_regime", {}))
        htf = gold_raw.pop("rsi_htf", "")
        gold = _build(GoldRegimeConfig, gold_raw, "engines.gold_regime")
        if htf:
            gold = replace(gold, rsi_htf=Timeframe.parse(htf))

        provider_cfg = data.get("provider", {})
        alpaca = provider_cfg.get("alpaca", {})
        polygon = provider_cfg.get("polygon", {})

        pine_section = data.get("pine", {})
        pine_scripts = [k for k in pine_section]
        pine_inputs = {k: dict(v) for k, v in pine_section.items() if isinstance(v, dict)}

        return cls(
            scan=scan,
            iron_momentum=iron,
            silver_flow=silver,
            gold_regime=gold,
            pine_scripts=pine_scripts,
            pine_inputs=pine_inputs,
            alpaca_key_id=os.environ.get("ALPACA_KEY_ID", ""),
            alpaca_secret=os.environ.get("ALPACA_SECRET", ""),
            alpaca_feed=str(alpaca.get("feed", os.environ.get("ALPACA_FEED", "iex"))),
            polygon_api_key=os.environ.get("POLYGON_API_KEY", ""),
            polygon_tier=str(polygon.get("tier", os.environ.get("POLYGON_TIER", "free"))),
        )


def _build(cls: type, section: dict[str, Any], where: str):
    valid = {f.name for f in fields(cls)}
    unknown = set(section) - valid
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [{where}]; valid keys: {sorted(valid)}"
        )
    return cls(**section)
