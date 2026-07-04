"""Iron Momentum (Momentum Band ±50) — port of the Pine v6 indicator.

Pipeline: raw momentum ratio (velocity or position engine) → tanh squash to
±50 → EMA smoothing → state ladder / RVOL / volume-star events. All series
math mirrors the Pine source bar for bar, including warmup fallbacks
(zero-denominator ratios collapse to 0.0, not NaN, exactly like the Pine
ternaries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .. import ta
from .base import EngineResult, snapshot_from_series


@dataclass(frozen=True)
class IronMomentumConfig:
    mode: Literal["velocity", "position"] = "velocity"
    mom_len: int = 10
    bench_len: int = 50
    sensitivity: float = 2.0
    smooth_len: int = 3
    deadband: float = 5.0
    strong_lvl: float = 30.0
    rvol_len: int = 20
    rvol_trigger: float = 2.0
    rvol_rth_only: bool = True  # Pine: useETH defaults false → RTH-only RVOL
    stars_fresh_only: bool = False


class IronMomentumEngine:
    prefix = "mb"

    def __init__(self, config: IronMomentumConfig | None = None):
        self.config = config or IronMomentumConfig()

    def warmup_bars(self) -> int:
        c = self.config
        return c.bench_len + c.mom_len + c.smooth_len + c.rvol_len

    def compute(self, bars: pd.DataFrame, *, keep_series: bool = False) -> EngineResult:
        series = compute_series(bars, self.config)
        result = EngineResult(snapshot=snapshot_from_series(series, self.prefix))
        if keep_series:
            result.series = series
        return result


def compute_series(bars: pd.DataFrame, cfg: IronMomentumConfig) -> pd.DataFrame:
    close, volume = bars["close"], bars["volume"]

    # ── Engine: raw ratio, both modes computed as in the Pine source ─────────
    if cfg.mode == "velocity":
        mom = close.diff(cfg.mom_len)
        norm = ta.sma(mom.abs(), cfg.bench_len)
        raw = pd.Series(np.where(norm > 0, mom / norm, 0.0), index=bars.index)
    else:
        base = ta.sma(close, cfg.bench_len)
        vol = ta.stdev_pop(close, cfg.bench_len)
        raw = pd.Series(np.where(vol > 0, (close - base) / vol, 0.0), index=bars.index)

    score_raw = 50.0 * np.tanh(raw / cfg.sensitivity)
    score = ta.pine_ema(pd.Series(score_raw, index=bars.index), cfg.smooth_len)

    # ── State ladder: +2 strong bull … -2 strong bear ────────────────────────
    strong = max(cfg.strong_lvl, cfg.deadband)
    state = pd.Series(
        np.select(
            [score >= strong, score > cfg.deadband, score <= -strong, score < -cfg.deadband],
            [2, 1, -2, -1],
            default=0,
        ),
        index=bars.index,
    )
    dir_state = np.sign(state)
    strength_pct = score.abs() / 50.0 * 100.0

    # ── RVOL: all-bar or RTH-only rolling average (current bar included) ─────
    is_rth = bars["session"] == "rth"
    if cfg.rvol_rth_only:
        rth_avg = volume[is_rth].rolling(cfg.rvol_len, min_periods=cfg.rvol_len).mean()
        avg_vol = rth_avg.reindex(bars.index).ffill()
    else:
        avg_vol = ta.sma(volume, cfg.rvol_len)
    rvol = pd.Series(np.where(avg_vol > 0, volume / avg_vol, np.nan), index=bars.index)

    sess_ok = is_rth | (not cfg.rvol_rth_only)
    vol_surge = sess_ok & rvol.notna() & (rvol >= cfg.rvol_trigger)

    star_up_cond = (dir_state == 1) & vol_surge
    star_dn_cond = (dir_state == -1) & vol_surge
    if cfg.stars_fresh_only:
        star_up = star_up_cond & ~star_up_cond.shift(fill_value=False)
        star_dn = star_dn_cond & ~star_dn_cond.shift(fill_value=False)
    else:
        star_up, star_dn = star_up_cond, star_dn_cond

    return pd.DataFrame(
        {
            "score": score,
            "state": state,
            "strength_pct": strength_pct,
            "raw": raw,
            "rvol": rvol,
            "star": np.where(star_up, 1, np.where(star_dn, -1, 0)),
            "cross_zero_up": ta.crossover(score, 0.0),
            "cross_zero_dn": ta.crossunder(score, 0.0),
            "cross_strong_up": ta.crossover(score, strong),
            "cross_strong_dn": ta.crossunder(score, -strong),
        },
        index=bars.index,
    )
