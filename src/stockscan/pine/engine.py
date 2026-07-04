"""Adapt a PineScript file to the scanner Engine protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..engines.base import EngineResult
from .parser import Script, parse
from .runtime import Interpreter, slugify


class PineEngine:
    def __init__(
        self,
        script: Script,
        *,
        prefix: str | None = None,
        input_overrides: dict[str, Any] | None = None,
        warmup: int = 250,
    ):
        self.script = script
        self.input_overrides = dict(input_overrides or {})
        self._warmup = warmup
        self._prefix = prefix

    @classmethod
    def from_file(
        cls, path: Path, *, input_overrides: dict[str, Any] | None = None
    ) -> PineEngine:
        source = Path(path).read_text()
        script = parse(source, source_name=str(path))
        return cls(script, input_overrides=input_overrides, warmup=250)

    @property
    def prefix(self) -> str:
        if self._prefix:
            return self._prefix
        # discovered on first compute (shorttitle lives in the indicator() call);
        # fall back to the file name until then
        name = Path(self.script.source_name).stem
        return slugify(name)

    def warmup_bars(self) -> int:
        return self._warmup

    def run_series(self, bars: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        interp = Interpreter(self.script, self.input_overrides, symbol=symbol)
        result = interp.run(bars)
        if not self._prefix:
            self._prefix = slugify(
                result.shorttitle or result.title or Path(self.script.source_name).stem
            )
        return result.to_frame(bars.index)

    def compute(self, bars: pd.DataFrame, *, keep_series: bool = False) -> EngineResult:
        series = self.run_series(bars)
        prefix = self.prefix
        last = series.iloc[-1]
        snapshot: dict[str, Any] = {}
        for col in series.columns:
            value = last[col]
            key = f"{prefix}_{col}"
            if isinstance(value, bool) or str(series[col].dtype) == "bool":
                snapshot[key] = bool(value)
            elif pd.isna(value):
                snapshot[key] = None
            else:
                snapshot[key] = float(value) if isinstance(value, (int, float)) else value
        result = EngineResult(snapshot=snapshot)
        if keep_series:
            result.series = series
        return result
