"""Engine protocol: bars in, exported metrics out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class EngineResult:
    snapshot: dict[str, object]
    series: pd.DataFrame | None = field(default=None, repr=False)


@runtime_checkable
class Engine(Protocol):
    prefix: str

    def compute(self, bars: pd.DataFrame, *, keep_series: bool = False) -> EngineResult: ...

    def warmup_bars(self) -> int: ...


def snapshot_from_series(series: pd.DataFrame, prefix: str) -> dict[str, object]:
    """Latest-bar values of every exported column, prefixed for the scan row."""
    last = series.iloc[-1]
    out: dict[str, object] = {}
    for col, value in last.items():
        if isinstance(value, (bool,)) or str(series[col].dtype) == "bool":
            out[f"{prefix}_{col}"] = bool(value)
        elif pd.isna(value):
            out[f"{prefix}_{col}"] = None
        elif isinstance(value, float) and float(value).is_integer() and col.endswith(
            ("state", "regime", "flip", "star", "confluence", "tier")
        ):
            out[f"{prefix}_{col}"] = int(value)
        else:
            out[f"{prefix}_{col}"] = float(value) if isinstance(value, float) else value
    return out
