"""Golden tests for Pine-parity primitives, checked against literal
Pine-recursion reference loops and hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from stockscan import ta


def series(*values: float) -> pd.Series:
    return pd.Series([float(v) for v in values])


class TestSma:
    def test_warmup_and_values(self):
        out = ta.sma(series(1, 2, 3, 4, 5), 3)
        assert out.isna().tolist() == [True, True, False, False, False]
        assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


class TestStdevPop:
    def test_population_ddof(self):
        # Pine ta.stdev([1,2,3,4], 4) = population stdev = sqrt(1.25)
        out = ta.stdev_pop(series(1, 2, 3, 4), 4)
        assert out.iloc[-1] == pytest.approx(np.sqrt(1.25))
        # sample stdev would be sqrt(5/3) — make sure we are NOT that
        assert out.iloc[-1] != pytest.approx(np.sqrt(5 / 3))


class TestPineEma:
    def test_reference_recursion(self):
        rng = np.random.default_rng(1)
        s = pd.Series(rng.normal(0, 1, 50).cumsum() + 100)
        n = 5
        out = ta.pine_ema(s, n)
        # literal Pine recursion: na until SMA seed, then alpha blend
        alpha = 2 / (n + 1)
        ref = [np.nan] * (n - 1)
        prev = s.iloc[:n].mean()
        ref.append(prev)
        for x in s.iloc[n:]:
            prev = alpha * x + (1 - alpha) * prev
            ref.append(prev)
        np.testing.assert_allclose(out.to_numpy(), np.array(ref), rtol=1e-12)

    def test_len_1_identity(self):
        s = series(3, 1, 4, 1, 5)
        pd.testing.assert_series_equal(ta.pine_ema(s, 1), s)

    def test_leading_nans_shift_seed(self):
        s = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0])
        out = ta.pine_ema(s, 3)
        assert out.isna().tolist() == [True, True, True, True, False, False]
        assert out.iloc[4] == pytest.approx(2.0)  # SMA seed of 1,2,3
        assert out.iloc[5] == pytest.approx(0.5 * 4 + 0.5 * 2.0)


class TestRma:
    def test_reference_recursion(self):
        s = series(10, 11, 12, 11, 13, 14, 12)
        n = 3
        out = ta.rma(s, n)
        prev = s.iloc[:3].mean()
        assert out.iloc[2] == pytest.approx(prev)
        for i in range(3, len(s)):
            prev = (1 / n) * s.iloc[i] + (1 - 1 / n) * prev
            assert out.iloc[i] == pytest.approx(prev)


class TestRsi:
    def test_known_sequence(self):
        # Classic all-up sequence: RSI = 100
        up = series(*range(1, 20))
        assert ta.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
        # All-down: RSI = 0
        down = series(*range(20, 1, -1))
        assert ta.rsi(down, 14).iloc[-1] == pytest.approx(0.0)

    def test_wilder_reference(self):
        rng = np.random.default_rng(2)
        s = pd.Series(rng.normal(0, 1, 60).cumsum() + 50)
        n = 14
        out = ta.rsi(s, n)
        # loop reference
        change = s.diff().to_numpy()
        gains = np.clip(change, 0, None)[1:]
        losses = np.clip(-change, 0, None)[1:]
        avg_g = gains[:n].mean()
        avg_l = losses[:n].mean()
        for i in range(n, len(gains)):
            avg_g = (gains[i] + (n - 1) * avg_g) / n
            avg_l = (losses[i] + (n - 1) * avg_l) / n
        expected = 100 - 100 / (1 + avg_g / avg_l)
        assert out.iloc[-1] == pytest.approx(expected, rel=1e-12)
        assert out.iloc[: n].isna().all()  # warmup: n changes need n+1 bars


class TestAtr:
    def test_true_range_gap(self):
        df = pd.DataFrame(
            {
                "high": [10.0, 12.0, 20.0],
                "low": [9.0, 11.0, 19.0],
                "close": [9.5, 11.5, 19.5],
            }
        )
        tr = ta.true_range(df)
        assert tr.iloc[0] == pytest.approx(1.0)  # first bar: high - low
        assert tr.iloc[2] == pytest.approx(20.0 - 11.5)  # gap up vs prev close


class TestPercentRank:
    def test_pine_convention(self):
        # previous n values less than or equal to current
        s = series(1, 2, 3, 4, 5, 0, 3)
        out = ta.percentrank(s, 4)
        assert np.isnan(out.iloc[3])
        assert out.iloc[4] == pytest.approx(100.0)  # 1,2,3,4 all <= 5
        assert out.iloc[5] == pytest.approx(0.0)    # none of 2,3,4,5 <= 0
        assert out.iloc[6] == pytest.approx(50.0)   # 3 and 0 are <= 3 among 3,4,5,0

    def test_ties_count(self):
        s = series(2, 2, 2, 2, 2)
        out = ta.percentrank(s, 3)
        assert out.iloc[3] == pytest.approx(100.0)
        assert out.iloc[4] == pytest.approx(100.0)


class TestCrosses:
    def test_crossover_semantics(self):
        a = series(1, 2, 3, 2, 3)
        b = series(2, 2, 2, 2, 2)
        assert ta.crossover(a, b).tolist() == [False, False, True, False, True]
        # 2 < 2 is false, so dipping back to the level is not a crossunder
        assert ta.crossunder(a, b).tolist() == [False] * 5
        assert ta.crossunder(series(3, 2, 1), 2.0).tolist() == [False, False, True]

    def test_touch_then_cross(self):
        # equality on prev bar counts as "was <=" for crossover
        a = series(2, 3)
        assert ta.crossover(a, 2.0).tolist() == [False, True]

    def test_nan_blocks_cross(self):
        a = pd.Series([np.nan, 3.0])
        assert ta.crossover(a, 2.0).tolist() == [False, False]


class TestBarsSince:
    def test_basic(self):
        cond = pd.Series([False, True, False, False, True, False])
        out = ta.barssince(cond)
        assert np.isnan(out.iloc[0])
        assert out.iloc[1:].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0]
