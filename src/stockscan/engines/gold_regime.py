"""Gold Regime (Golden/Grey Ribbon 8/20 SMA + RSI band) — Pine v6 port.

State ladder: 0 = chop (ATR-width gate) | ±1 direction | ±2 +RSI agreement |
±3 +RSI band thrust. The scanner consumes completed bars only, so the Pine
"confirmed" state equals the raw state here, and the optional HTF RSI leg uses
only closed higher-timeframe bars (non-repainting by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import ta
from ..models import Timeframe
from ..sessions import EASTERN
from .base import EngineResult, snapshot_from_series


@dataclass(frozen=True)
class GoldRegimeConfig:
    fast_len: int = 8
    slow_len: int = 20
    atr_len: int = 14
    rsi_len: int = 14
    band_len: int = 20
    band_mult: float = 1.0
    rsi_htf: Timeframe | None = None  # None = chart timeframe (no resample)
    width_thresh: float = 0.5
    use_rsi_tier: bool = True
    use_thrust_tier: bool = True


class GoldRegimeEngine:
    prefix = "ggr"

    def __init__(self, config: GoldRegimeConfig | None = None):
        self.config = config or GoldRegimeConfig()

    def warmup_bars(self) -> int:
        c = self.config
        base = max(c.slow_len, c.rsi_len + c.band_len + 1, c.atr_len) + 5
        return base

    def compute(self, bars: pd.DataFrame, *, keep_series: bool = False) -> EngineResult:
        series = compute_series(bars, self.config)
        result = EngineResult(snapshot=snapshot_from_series(series, self.prefix))
        if keep_series:
            result.series = series
        return result


def _htf_rsi_leg(
    bars: pd.DataFrame, cfg: GoldRegimeConfig
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """RSI/basis/dev computed on closed higher-timeframe bars, projected onto
    the chart index (equivalent to Pine's lookahead_on + [1] non-repainting
    request.security pattern: each chart bar sees the last *closed* HTF value).
    """
    htf = cfg.rsi_htf
    eastern_index = bars.index.tz_convert(EASTERN)
    close_et = pd.Series(bars["close"].to_numpy(), index=eastern_index)
    if htf.is_intraday:
        # Anchor intraday buckets to the 09:30 session open, like TradingView.
        grouped = close_et.resample(htf.pandas_freq, origin="start_day", offset="9h30min")
    else:
        grouped = close_et.resample("1D")
    htf_close = grouped.last().dropna()
    htf_rsi = ta.rsi(htf_close, cfg.rsi_len)
    htf_basis = ta.sma(htf_rsi, cfg.band_len)
    htf_dev = ta.stdev_pop(htf_rsi, cfg.band_len)

    # Chart bar t belongs to the HTF bucket that starts at or before t; the
    # last *closed* HTF bar is the previous bucket, hence the shift(1).
    bucket_index = htf_close.index
    positions = bucket_index.searchsorted(close_et.index, side="right") - 1
    positions = np.clip(positions, 0, len(bucket_index) - 1)

    def project(htf_series: pd.Series) -> pd.Series:
        values = htf_series.shift(1).to_numpy()[positions]
        return pd.Series(values, index=bars.index)

    return project(htf_rsi), project(htf_basis), project(htf_dev)


def compute_series(bars: pd.DataFrame, cfg: GoldRegimeConfig) -> pd.DataFrame:
    close = bars["close"]
    sma_fast = ta.sma(close, cfg.fast_len)
    sma_slow = ta.sma(close, cfg.slow_len)
    atr = ta.atr(bars, cfg.atr_len)

    if cfg.rsi_htf is None:
        rsi = ta.rsi(close, cfg.rsi_len)
        basis = ta.sma(rsi, cfg.band_len)
        dev = ta.stdev_pop(rsi, cfg.band_len)
    else:
        rsi, basis, dev = _htf_rsi_leg(bars, cfg)
    dev = dev * cfg.band_mult
    upper = (basis + dev).clip(upper=100.0)
    lower = (basis - dev).clip(lower=0.0)

    ma_state = pd.Series(
        np.select([sma_fast > sma_slow, sma_fast < sma_slow], [1, -1], default=0),
        index=bars.index,
    )
    rsi_state = pd.Series(
        np.select([rsi > basis, rsi < basis], [1, -1], default=0), index=bars.index
    )
    thrust_state = pd.Series(
        np.select([rsi > upper, rsi < lower], [1, -1], default=0), index=bars.index
    )

    width_n = pd.Series(
        np.where(atr > 0, (sma_fast - sma_slow) / atr, np.nan), index=bars.index
    )
    if cfg.width_thresh > 0:
        width_ok = (width_n.abs() >= cfg.width_thresh).fillna(False)
    else:
        width_ok = pd.Series(True, index=bars.index)

    tier = pd.Series(0, index=bars.index)
    directional = (ma_state != 0) & width_ok
    tier = tier.mask(directional, 1)
    if cfg.use_rsi_tier:
        tier = tier.mask(directional & (rsi_state == ma_state), 2)
    if cfg.use_thrust_tier:
        tier = tier.mask(directional & (thrust_state == ma_state), 3)
    state = ma_state * tier

    # barsSinceFlip resets on any sign change of the state, incl. into/out of 0
    sign = np.sign(state)
    flip = sign != sign.shift()
    bars_since_flip = ta.barssince(pd.Series(flip, index=bars.index)).fillna(0)

    prev_state = state.shift(fill_value=0)
    return pd.DataFrame(
        {
            "state": state,
            "width_atr": width_n,
            "rsi": rsi,
            "rsi_basis": basis,
            "rsi_band_width": upper - lower,
            "thrust": thrust_state,
            "ma_rsi_agree": (ma_state == rsi_state).astype(int),
            "bars_since_flip": bars_since_flip,
            "cross_up": ta.crossover(sma_fast, sma_slow),
            "cross_dn": ta.crossunder(sma_fast, sma_slow),
            "strong_up_enter": (state >= 2) & (prev_state < 2),
            "strong_dn_enter": (state <= -2) & (prev_state > -2),
            "thrust_up_enter": (state == 3) & (prev_state != 3),
            "thrust_dn_enter": (state == -3) & (prev_state != -3),
        },
        index=bars.index,
    )
