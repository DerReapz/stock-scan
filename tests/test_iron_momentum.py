"""Iron Momentum engine vs a literal bar-loop Pine reference."""

import math

import numpy as np
import pandas as pd
import pytest

from stockscan.engines.iron_momentum import IronMomentumConfig, compute_series

from .conftest import make_bars


def pine_reference(bars: pd.DataFrame, cfg: IronMomentumConfig) -> pd.DataFrame:
    """Direct transcription of the Pine script, one bar at a time."""
    n = len(bars)
    close = bars["close"].to_numpy()
    volume = bars["volume"].to_numpy()
    is_rth = (bars["session"] == "rth").to_numpy()

    mom_hist: list[float] = []
    raw_out = np.zeros(n)
    score = np.full(n, np.nan)
    ema_prev = None
    seed_buf: list[float] = []
    rth_queue: list[float] = []
    rvol = np.full(n, np.nan)
    state = np.zeros(n, dtype=int)
    star = np.zeros(n, dtype=int)
    prev_up = prev_dn = False

    strong = max(cfg.strong_lvl, cfg.deadband)
    alpha = 2 / (cfg.smooth_len + 1)

    for i in range(n):
        # velocity engine
        mom = close[i] - close[i - cfg.mom_len] if i >= cfg.mom_len else None
        mom_hist.append(abs(mom) if mom is not None else None)
        window = [m for m in mom_hist[-cfg.bench_len:] if m is not None]
        norm = (
            sum(window) / cfg.bench_len
            if len([m for m in mom_hist[-cfg.bench_len:]]) >= cfg.bench_len
            and len(window) == cfg.bench_len
            else None
        )
        raw = mom / norm if (norm is not None and norm > 0 and mom is not None) else 0.0
        raw_out[i] = raw

        score_raw = 50.0 * math.tanh(raw / cfg.sensitivity)
        if cfg.smooth_len == 1:
            score[i] = score_raw
        elif ema_prev is None:
            seed_buf.append(score_raw)
            if len(seed_buf) == cfg.smooth_len:
                ema_prev = sum(seed_buf) / cfg.smooth_len
                score[i] = ema_prev
        else:
            ema_prev = alpha * score_raw + (1 - alpha) * ema_prev
            score[i] = ema_prev

        s = score[i]
        if not np.isnan(s):
            if s >= strong:
                state[i] = 2
            elif s > cfg.deadband:
                state[i] = 1
            elif s <= -strong:
                state[i] = -2
            elif s < -cfg.deadband:
                state[i] = -1

        # RVOL — Pine pushes the current RTH bar before averaging
        if cfg.rvol_rth_only:
            if is_rth[i]:
                rth_queue.append(volume[i])
                if len(rth_queue) > cfg.rvol_len:
                    rth_queue.pop(0)
            avg = sum(rth_queue) / cfg.rvol_len if len(rth_queue) >= cfg.rvol_len else None
        else:
            avg = (
                sum(volume[i - cfg.rvol_len + 1 : i + 1]) / cfg.rvol_len
                if i >= cfg.rvol_len - 1
                else None
            )
        rvol[i] = volume[i] / avg if avg else np.nan

        sess_ok = is_rth[i] or not cfg.rvol_rth_only
        surge = sess_ok and not np.isnan(rvol[i]) and rvol[i] >= cfg.rvol_trigger
        up_cond = state[i] > 0 and surge
        dn_cond = state[i] < 0 and surge
        up = up_cond and (not cfg.stars_fresh_only or not prev_up)
        dn = dn_cond and (not cfg.stars_fresh_only or not prev_dn)
        star[i] = 1 if up else -1 if dn else 0
        prev_up, prev_dn = up_cond, dn_cond

    return pd.DataFrame(
        {"raw": raw_out, "score": score, "state": state, "rvol": rvol, "star": star},
        index=bars.index,
    )


@pytest.mark.parametrize("rth_only", [False, True])
@pytest.mark.parametrize("fresh_only", [False, True])
def test_matches_pine_reference(rth_only, fresh_only):
    bars = make_bars(400, include_extended=True, trend=0.05, seed=11)
    cfg = IronMomentumConfig(rvol_rth_only=rth_only, stars_fresh_only=fresh_only)
    got = compute_series(bars, cfg)
    ref = pine_reference(bars, cfg)
    for col in ("raw", "score", "rvol"):
        np.testing.assert_allclose(
            got[col].to_numpy(), ref[col].to_numpy(), rtol=1e-9, atol=1e-9, err_msg=col
        )
    np.testing.assert_array_equal(got["state"].to_numpy(), ref["state"].to_numpy())
    np.testing.assert_array_equal(got["star"].to_numpy(), ref["star"].to_numpy())


def test_position_mode_zscore():
    bars = make_bars(200, seed=3)
    got = compute_series(bars, IronMomentumConfig(mode="position"))
    close = bars["close"]
    base = close.rolling(50).mean()
    sd = close.rolling(50).std(ddof=0)
    raw_expected = ((close - base) / sd).where(sd > 0, 0.0).fillna(0.0)
    np.testing.assert_allclose(got["raw"].to_numpy(), raw_expected.to_numpy(), rtol=1e-9)


def test_score_bounded_and_squash_value():
    bars = make_bars(300, trend=0.3, vol=0.05, seed=5)  # hard trend → rails
    got = compute_series(bars, IronMomentumConfig())
    assert got["score"].abs().max() <= 50.0
    # tanh squash: raw == sensitivity → score_raw = 50*tanh(1)
    assert 50 * np.tanh(1.0) == pytest.approx(38.0797, abs=1e-4)


def test_eth_bars_never_star():
    bars = make_bars(400, include_extended=True, trend=0.05, seed=11)
    got = compute_series(bars, IronMomentumConfig(rvol_rth_only=True))
    eth = bars["session"] != "rth"
    assert (got.loc[eth, "star"] == 0).all()


def test_warmup_raw_is_zero_not_nan():
    bars = make_bars(80)
    got = compute_series(bars, IronMomentumConfig())
    assert (got["raw"].iloc[:50] == 0.0).all()
