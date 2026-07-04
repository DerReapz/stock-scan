"""Silver Flow engine: bar-loop Pine reference + structural edge cases."""

import numpy as np
import pandas as pd
import pytest

from stockscan import ta
from stockscan.engines.silver_flow import SilverFlowConfig, compute_series

from .conftest import make_bars


def pine_reference(bars: pd.DataFrame, cfg: SilverFlowConfig) -> pd.DataFrame:
    """Literal per-bar transcription of the Pine pipeline (share lines,
    biases, signals). Slow but unambiguous."""
    n = len(bars)
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    volume = bars["volume"].to_numpy()
    dv = volume * ((high + low + close) / 3.0)

    def percentrank_at(i: int, values: np.ndarray, length: int) -> float:
        if i < length or np.isnan(values[i - length : i + 1]).any():
            return np.nan
        window = values[i - length : i]
        return float((window <= values[i]).sum()) * 100.0 / length

    gate_val = np.array([percentrank_at(i, dv, cfg.gate_len) for i in range(n)])
    span = cfg.inst_thr - cfg.retail_thr
    ramp = np.clip((gate_val - cfg.retail_thr) / span, 0.0, 1.0)
    inst_w = ramp if cfg.soft_gate else (gate_val >= cfg.inst_thr).astype(float)
    retail_w = 1.0 - ramp if cfg.soft_gate else (gate_val <= cfg.retail_thr).astype(float)

    clv = np.zeros(n)
    for i in range(n):
        prev_c = close[i - 1] if i > 0 else close[i]
        hi_t, lo_t = max(high[i], prev_c), min(low[i], prev_c)
        rng = hi_t - lo_t
        clv[i] = ((close[i] - lo_t) - (hi_t - close[i])) / rng if rng > 0 else 0.0
    bull_frac = (clv + 1.0) / 2.0
    dir_w = np.ones(n) if cfg.weight_mode == "participation" else bull_frac

    def wsum(values: np.ndarray, i: int, length: int) -> float:
        if i < length - 1:
            return np.nan
        return float(values[i - length + 1 : i + 1].sum())

    retail_share_raw = np.empty(n)
    inst_bias = np.empty(n)
    retail_bias = np.empty(n)
    med_vol = np.empty(n)
    for i in range(n):
        inst_raw = wsum(dv * inst_w * dir_w, i, cfg.accum_len)
        retail_raw = wsum(dv * retail_w * dir_w, i, cfg.accum_len)
        total = inst_raw + retail_raw
        retail_share_raw[i] = 100.0 * retail_raw / total if total > 0 else 50.0

        ig = wsum(dv * inst_w, i, cfg.div_len)
        iv = wsum(dv * inst_w * clv, i, cfg.div_len)
        inst_bias[i] = iv / ig if ig > 0 else 0.0
        rg = wsum(dv * retail_w, i, cfg.div_len)
        rv = wsum(dv * retail_w * clv, i, cfg.div_len)
        retail_bias[i] = rv / rg if rg > 0 else 0.0

        window = gate_val[max(0, i - cfg.div_len + 1) : i + 1]
        med_vol[i] = (
            np.median(window) if i >= cfg.div_len - 1 and not np.isnan(window).any() else np.nan
        )

    # EMA smoothing with Pine SMA seeding
    def pine_ema(values: np.ndarray, length: int) -> np.ndarray:
        out = np.full(n, np.nan)
        alpha = 2 / (length + 1)
        prev = None
        buf: list[float] = []
        for i, x in enumerate(values):
            if np.isnan(x):
                continue
            if prev is None:
                buf.append(x)
                if len(buf) == length:
                    prev = sum(buf) / length
                    out[i] = prev
            else:
                prev = alpha * x + (1 - alpha) * prev
                out[i] = prev
        return out

    retail_line = pine_ema(retail_share_raw, cfg.smooth_len)
    inst_line = pine_ema(100.0 - retail_share_raw, cfg.smooth_len)

    accum = np.zeros(n, dtype=bool)
    dist = np.zeros(n, dtype=bool)
    for i in range(cfg.div_len, n):
        quiet = not np.isnan(med_vol[i]) and med_vol[i] <= cfg.low_vol_pct
        chg = close[i] - close[i - cfg.div_len]
        accum[i] = chg < 0 and quiet and inst_bias[i] >= cfg.bias_thr
        dist[i] = chg > 0 and quiet and inst_bias[i] <= -cfg.bias_thr

    return pd.DataFrame(
        {
            "retail": retail_line,
            "inst": inst_line,
            "inst_bias": inst_bias,
            "retail_bias": retail_bias,
            "tape_pct": med_vol,
            "accum_raw": accum,
            "dist_raw": dist,
        },
        index=bars.index,
    )


@pytest.mark.parametrize("soft", [True, False])
@pytest.mark.parametrize("weight_mode", ["participation", "directional"])
def test_share_mode_matches_reference(soft, weight_mode):
    bars = make_bars(260, seed=17)
    cfg = SilverFlowConfig(soft_gate=soft, weight_mode=weight_mode, edge_only=False)
    got = compute_series(bars, cfg)
    ref = pine_reference(bars, cfg)
    for col in ("retail", "inst", "inst_bias", "retail_bias", "tape_pct"):
        np.testing.assert_allclose(
            got[col].to_numpy(), ref[col].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=col
        )
    np.testing.assert_array_equal(got["accum"].to_numpy(), ref["accum_raw"].to_numpy())
    np.testing.assert_array_equal(got["dist"].to_numpy(), ref["dist_raw"].to_numpy())


def test_share_mirror_symmetry():
    bars = make_bars(260, seed=17)
    got = compute_series(bars, SilverFlowConfig())
    valid = got["retail"].notna()
    np.testing.assert_allclose(
        (got.loc[valid, "retail"] + got.loc[valid, "inst"]).to_numpy(), 100.0, rtol=1e-9
    )


def test_signed_mode_bounded():
    bars = make_bars(260, seed=17)
    got = compute_series(bars, SilverFlowConfig(scale_mode="signed"))
    assert got["retail"].dropna().between(-100, 100).all()
    assert got["inst"].dropna().between(-100, 100).all()


def test_independent_mode_uses_percentrank():
    bars = make_bars(300, seed=17)
    cfg = SilverFlowConfig(scale_mode="independent")
    got = compute_series(bars, cfg)
    dv = bars["volume"] * ta.hlc3(bars)
    gate_val = ta.percentrank(dv, cfg.gate_len)
    ramp = ((gate_val - 50.0) / 25.0).clip(0, 1)
    clv_dirw = 1.0  # participation
    inst_raw = ta.rolling_sum(dv * ramp * clv_dirw, cfg.accum_len)
    expected = ta.pine_ema(ta.percentrank(inst_raw, cfg.gate_len), cfg.smooth_len)
    np.testing.assert_allclose(
        got["inst"].to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9
    )


def test_zero_range_bar_clv_is_zero():
    bars = make_bars(150, seed=4)
    i = 120
    bars.iloc[i, bars.columns.get_loc("high")] = bars["close"].iloc[i - 1]
    bars.iloc[i, bars.columns.get_loc("low")] = bars["close"].iloc[i - 1]
    bars.iloc[i, bars.columns.get_loc("close")] = bars["close"].iloc[i - 1]
    bars.iloc[i, bars.columns.get_loc("open")] = bars["close"].iloc[i - 1]
    got = compute_series(bars, SilverFlowConfig())
    assert np.isfinite(got.iloc[i]["retail"])  # no NaN poisoning


def test_hard_gate_middle_band_discarded():
    bars = make_bars(260, seed=17)
    cfg = SilverFlowConfig(soft_gate=False)
    got = compute_series(bars, cfg)
    ref = pine_reference(bars, cfg)
    np.testing.assert_allclose(
        got["retail"].to_numpy(), ref["retail"].to_numpy(), rtol=1e-9, atol=1e-9
    )


def test_regime_and_flip_consistency():
    bars = make_bars(300, seed=29)
    got = compute_series(bars, SilverFlowConfig())
    flips = got.index[got["flip"] != 0]
    for t in flips:
        i = got.index.get_loc(t)
        if i == 0:
            continue
        # after a flip, strict dominance must point in the flip's direction
        after = got["retail"].iloc[i] - got["inst"].iloc[i]
        assert np.sign(after) != 0
        assert np.sign(after) == got["flip"].iloc[i]
