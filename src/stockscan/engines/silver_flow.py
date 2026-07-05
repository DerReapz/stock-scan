"""Silver Flow (Institutional vs Retail) — Pine v6 port.

Size-gated dollar-volume partition: each bar's notional is split between a
"retail" and an "institutional" bucket by where its dollar volume ranks in a
rolling window, accumulated, and projected onto one of three scales (share /
independent / signed). A divergence layer flags quiet-tape accumulation and
distribution, and a confluence layer flags size-independent directional
agreement on an active tape.

Warmup fallbacks mirror the Pine ternaries exactly: comparisons against NaN
are false, so guarded ratios collapse to their fallback (50 for share, 0 for
biases) rather than NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .. import ta
from .base import EngineResult, snapshot_from_series


@dataclass(frozen=True)
class SilverFlowConfig:
    gate_mode: Literal["percentile", "zscore"] = "percentile"
    gate_len: int = 100
    retail_thr: float = 50.0  # percentile mode default; use 0.0 for zscore
    inst_thr: float = 75.0    # percentile mode default; use 1.0 for zscore
    soft_gate: bool = True
    weight_mode: Literal["participation", "directional"] = "participation"
    scale_mode: Literal["share", "independent", "signed"] = "share"
    accum_len: int = 13
    smooth_len: int = 5
    double_smooth: bool = False
    div_len: int = 8
    low_vol_pct: float = 40.0
    active_vol_pct: float = 60.0
    bias_thr: float = 0.25
    edge_only: bool = True


class SilverFlowEngine:
    prefix = "sf"

    def __init__(self, config: SilverFlowConfig | None = None):
        self.config = config or SilverFlowConfig()

    def warmup_bars(self) -> int:
        c = self.config
        return c.gate_len + max(c.accum_len, c.div_len) + c.smooth_len * (
            2 if c.double_smooth else 1
        )

    def compute(self, bars: pd.DataFrame, *, keep_series: bool = False) -> EngineResult:
        series = compute_series(bars, self.config)
        result = EngineResult(snapshot=snapshot_from_series(series, self.prefix))
        if keep_series:
            result.series = series
        return result


def _guarded_ratio(num: pd.Series, denom: pd.Series, fallback: float) -> pd.Series:
    """Pine `denom > 0 ? num / denom : fallback` (NaN denom → fallback)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / denom
    return pd.Series(np.where(denom > 0, ratio, fallback), index=num.index)


def compute_series(bars: pd.DataFrame, cfg: SilverFlowConfig) -> pd.DataFrame:
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]

    # ── 1 · Notional ─────────────────────────────────────────────────────────
    dv = volume * ta.hlc3(bars)

    # ── 2 · Size gate ────────────────────────────────────────────────────────
    if cfg.gate_mode == "percentile":
        gate_val = ta.percentrank(dv, cfg.gate_len)
    else:
        mu = ta.sma(dv, cfg.gate_len)
        sd = ta.stdev_pop(dv, cfg.gate_len)
        gate_val = _guarded_ratio(dv - mu, sd, 0.0)
    lo_thr, hi_thr = cfg.retail_thr, cfg.inst_thr

    span = hi_thr - lo_thr
    if span > 0:
        ramp = ((gate_val - lo_thr) / span).clip(0.0, 1.0)  # NaN propagates, like Pine
    else:
        ramp = (gate_val >= hi_thr).astype(float)
    if cfg.soft_gate:
        inst_w, retail_w = ramp, 1.0 - ramp
    else:
        inst_w = (gate_val >= hi_thr).astype(float)
        retail_w = (gate_val <= lo_thr).astype(float)

    # ── 3 · Gap-aware CLV ────────────────────────────────────────────────────
    prev_c = close.shift().fillna(close)
    hi_t = np.maximum(high, prev_c)
    lo_t = np.minimum(low, prev_c)
    trng = hi_t - lo_t
    clv = pd.Series(
        np.where(trng > 0, ((close - lo_t) - (hi_t - close)) / trng.replace(0, np.nan), 0.0),
        index=bars.index,
    )
    bull_frac = (clv + 1.0) / 2.0
    dir_w = pd.Series(1.0, index=bars.index) if cfg.weight_mode == "participation" else bull_frac

    # ── 4 · Bucket accumulation ──────────────────────────────────────────────
    inst_raw = ta.rolling_sum(dv * inst_w * dir_w, cfg.accum_len)
    retail_raw = ta.rolling_sum(dv * retail_w * dir_w, cfg.accum_len)

    # ── 5 · Normalization: share / independent / signed ─────────────────────
    total = inst_raw + retail_raw
    retail_share_raw = _guarded_ratio(100.0 * retail_raw, total, 50.0)

    inst_gross_l = ta.rolling_sum(dv * inst_w, cfg.accum_len)
    retail_gross_l = ta.rolling_sum(dv * retail_w, cfg.accum_len)
    inst_net_l = ta.rolling_sum(dv * inst_w * clv, cfg.accum_len)
    retail_net_l = ta.rolling_sum(dv * retail_w * clv, cfg.accum_len)
    retail_sgn_raw = _guarded_ratio(100.0 * retail_net_l, retail_gross_l, 0.0)
    inst_sgn_raw = _guarded_ratio(100.0 * inst_net_l, inst_gross_l, 0.0)

    if cfg.scale_mode == "share":
        raw_r, raw_i = retail_share_raw, 100.0 - retail_share_raw
    elif cfg.scale_mode == "signed":
        raw_r, raw_i = retail_sgn_raw, inst_sgn_raw
    else:
        raw_r = ta.percentrank(retail_raw, cfg.gate_len)
        raw_i = ta.percentrank(inst_raw, cfg.gate_len)

    # ── 6 · Smoothing ────────────────────────────────────────────────────────
    r1 = ta.pine_ema(raw_r, cfg.smooth_len)
    i1 = ta.pine_ema(raw_i, cfg.smooth_len)
    retail_line = ta.pine_ema(r1, cfg.smooth_len) if cfg.double_smooth else r1
    inst_line = ta.pine_ema(i1, cfg.smooth_len) if cfg.double_smooth else i1

    # ── Signal layer ─────────────────────────────────────────────────────────
    flip_to_retail = ta.crossover(retail_line, inst_line)
    flip_to_inst = ta.crossunder(retail_line, inst_line)
    flip = flip_to_retail | flip_to_inst
    regime = pd.Series(
        np.select([retail_line > inst_line, retail_line < inst_line], [1, -1], default=0),
        index=bars.index,
    )
    bars_since_flip = ta.barssince(flip).fillna(0)

    # ── Acc/Dist divergence (fast window) ────────────────────────────────────
    dv_pct = ta.percentrank(dv, cfg.gate_len)
    med_vol_pct = ta.rolling_median(dv_pct, cfg.div_len)
    price_chg = close - close.shift(cfg.div_len)

    inst_net = ta.rolling_sum(dv * inst_w * clv, cfg.div_len)
    inst_gross = ta.rolling_sum(dv * inst_w, cfg.div_len)
    inst_bias = _guarded_ratio(inst_net, inst_gross, 0.0)
    retail_net = ta.rolling_sum(dv * retail_w * clv, cfg.div_len)
    retail_gross = ta.rolling_sum(dv * retail_w, cfg.div_len)
    retail_bias = _guarded_ratio(retail_net, retail_gross, 0.0)

    quiet = (med_vol_pct <= cfg.low_vol_pct).fillna(False)
    accum_raw = (price_chg < 0) & quiet & (inst_bias >= cfg.bias_thr)
    dist_raw = (price_chg > 0) & quiet & (inst_bias <= -cfg.bias_thr)

    active = (
        pd.Series(True, index=bars.index)
        if cfg.active_vol_pct <= 0
        else (med_vol_pct >= cfg.active_vol_pct).fillna(False)
    )
    bull_conf_raw = (inst_bias >= cfg.bias_thr) & (retail_bias >= cfg.bias_thr) & active
    bear_conf_raw = (inst_bias <= -cfg.bias_thr) & (retail_bias <= -cfg.bias_thr) & active

    def edge(sig: pd.Series) -> pd.Series:
        return sig & ~sig.shift(fill_value=False) if cfg.edge_only else sig

    accum, dist = edge(accum_raw), edge(dist_raw)
    bull_conf, bear_conf = edge(bull_conf_raw), edge(bear_conf_raw)

    # Slow signed flow, smoothed — the Pine direction tint / panel readout
    inst_flow = ta.pine_ema(inst_sgn_raw, cfg.smooth_len)
    retail_flow = ta.pine_ema(retail_sgn_raw, cfg.smooth_len)

    return pd.DataFrame(
        {
            "retail": retail_line,
            "inst": inst_line,
            "regime": regime,
            "flip": np.where(flip_to_retail, 1, np.where(flip_to_inst, -1, 0)),
            "bars_since_flip": bars_since_flip,
            "inst_bias": inst_bias,
            "retail_bias": retail_bias,
            "inst_net_usd": inst_net_l,  # signed institutional $ flow, accum window
            "inst_flow": inst_flow,
            "retail_flow": retail_flow,
            "tape_pct": med_vol_pct,
            "accum": accum,
            "dist": dist,
            "confluence": np.where(bull_conf, 1, np.where(bear_conf, -1, 0)),
        },
        index=bars.index,
    )
