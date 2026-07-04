"""Gold Regime engine: reference loop for the state ladder + edge cases."""

import numpy as np
import pandas as pd
import pytest

from stockscan import ta
from stockscan.engines.gold_regime import GoldRegimeConfig, compute_series
from stockscan.models import Timeframe

from .conftest import make_bars


def reference_states(bars: pd.DataFrame, cfg: GoldRegimeConfig) -> np.ndarray:
    """Literal transcription of the Pine tier ladder, one bar at a time."""
    fast = ta.sma(bars["close"], cfg.fast_len).to_numpy()
    slow = ta.sma(bars["close"], cfg.slow_len).to_numpy()
    atr = ta.atr(bars, cfg.atr_len).to_numpy()
    rsi = ta.rsi(bars["close"], cfg.rsi_len)
    basis = ta.sma(rsi, cfg.band_len).to_numpy()
    dev = (ta.stdev_pop(rsi, cfg.band_len) * cfg.band_mult).to_numpy()
    rsi = rsi.to_numpy()

    out = np.zeros(len(bars), dtype=int)
    for i in range(len(bars)):
        upper = min(basis[i] + dev[i], 100.0)
        lower = max(basis[i] - dev[i], 0.0)
        ma_state = 0
        if not np.isnan(fast[i]) and not np.isnan(slow[i]):
            ma_state = 1 if fast[i] > slow[i] else -1 if fast[i] < slow[i] else 0
        rsi_state = 0
        if not np.isnan(rsi[i]) and not np.isnan(basis[i]):
            rsi_state = 1 if rsi[i] > basis[i] else -1 if rsi[i] < basis[i] else 0
        thrust = 0
        if not np.isnan(rsi[i]) and not np.isnan(dev[i]):
            thrust = 1 if rsi[i] > upper else -1 if rsi[i] < lower else 0
        width_n = (fast[i] - slow[i]) / atr[i] if atr[i] and not np.isnan(atr[i]) else np.nan
        width_ok = not np.isnan(width_n) and abs(width_n) >= cfg.width_thresh
        tier = 0
        if ma_state != 0 and width_ok:
            tier = 1
            if cfg.use_rsi_tier and rsi_state == ma_state:
                tier = 2
            if cfg.use_thrust_tier and thrust == ma_state:
                tier = 3
        out[i] = ma_state * tier
    return out


@pytest.mark.parametrize("rsi_tier,thrust_tier", [(True, True), (True, False), (False, False)])
def test_state_matches_reference(rsi_tier, thrust_tier):
    bars = make_bars(300, trend=0.06, seed=13)
    cfg = GoldRegimeConfig(use_rsi_tier=rsi_tier, use_thrust_tier=thrust_tier)
    got = compute_series(bars, cfg)
    np.testing.assert_array_equal(got["state"].to_numpy(), reference_states(bars, cfg))


def test_state_range_and_gate():
    bars = make_bars(300, seed=21)
    got = compute_series(bars, GoldRegimeConfig())
    assert got["state"].between(-3, 3).all()
    # wherever the gate rejects width, state must be 0
    gated = got["width_atr"].abs() < 0.5
    assert (got.loc[gated.fillna(True), "state"] == 0).all()


def test_bars_since_flip_resets_through_zero():
    bars = make_bars(300, seed=13, trend=0.06)
    got = compute_series(bars, GoldRegimeConfig())
    sign = np.sign(got["state"].to_numpy())
    bsf = got["bars_since_flip"].to_numpy()
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1]:
            assert bsf[i] == 0, f"bar {i}: sign change must reset counter"
        else:
            assert bsf[i] == bsf[i - 1] + 1


def test_htf_rsi_uses_only_closed_bars():
    bars = make_bars(400, timeframe=Timeframe.M5, seed=9)
    cfg = GoldRegimeConfig(rsi_htf=Timeframe.H1)
    got = compute_series(bars, cfg)
    # HTF series must be a step function: constant within each hourly bucket
    # (buckets anchored at 09:30 like the engine's resample offset)
    eastern = bars.index.tz_convert("America/New_York")
    buckets = (eastern - pd.Timedelta(minutes=30)).floor("1h") + pd.Timedelta(minutes=30)
    per_bucket = got["rsi"].groupby(buckets).nunique(dropna=False)
    assert (per_bucket <= 1).all()
    # and the value in bucket k must equal the *final* RSI of bucket k-1
    # computed over hourly closes — i.e. no lookahead into the open bucket.
    hourly_close = pd.Series(bars["close"].to_numpy(), index=eastern).resample(
        "1h", origin="start_day", offset="9h30min"
    ).last().dropna()
    hourly_rsi = ta.rsi(hourly_close, cfg.rsi_len)
    for i in range(40, len(bars), 37):
        bucket_start = eastern[i].floor("1h") + pd.Timedelta(minutes=30)
        if eastern[i].minute < 30:
            bucket_start -= pd.Timedelta(hours=1)
        pos = hourly_rsi.index.searchsorted(bucket_start, side="right") - 1
        expected = hourly_rsi.iloc[pos - 1] if pos >= 1 else np.nan
        gotten = got["rsi"].iloc[i]
        if np.isnan(expected):
            assert np.isnan(gotten)
        else:
            assert gotten == pytest.approx(expected, rel=1e-9)


def test_cross_events():
    bars = make_bars(300, seed=21)
    got = compute_series(bars, GoldRegimeConfig())
    fast = ta.sma(bars["close"], 8)
    slow = ta.sma(bars["close"], 20)
    pd.testing.assert_series_equal(
        got["cross_up"], ta.crossover(fast, slow), check_names=False
    )
