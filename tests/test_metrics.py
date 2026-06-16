"""Tests for performance metrics.

Hand-computed expected values — no mocking of financial calculations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    ANNUALIZATION_FACTOR_5MIN,
    compute_cumulative_profit_ratio,
    compute_max_drawdown,
    compute_sharpe_ratio,
    periods_per_year_for_timeframe,
)


class TestComputeSharpeRatio:
    def test_constant_values_returns_zero(self) -> None:
        """Constant portfolio values mean zero returns, zero std -> 0."""
        pv = np.array([100.0, 100.0, 100.0, 100.0])
        assert compute_sharpe_ratio(pv) == pytest.approx(0.0)

    def test_known_values(self) -> None:
        """Hand-compute Sharpe for a simple series."""
        pv = np.array([100.0, 110.0, 105.0, 115.0])
        log_returns = np.diff(np.log(pv))
        # log_returns ≈ [0.09531, -0.04652, 0.09091]
        mean_r = float(np.mean(log_returns))
        std_r = float(np.std(log_returns, ddof=1))
        expected = mean_r / std_r * float(ANNUALIZATION_FACTOR_5MIN)
        assert compute_sharpe_ratio(pv) == pytest.approx(expected)

    def test_single_value_returns_zero(self) -> None:
        pv = np.array([100.0])
        assert compute_sharpe_ratio(pv) == pytest.approx(0.0)

    def test_two_values_returns_zero(self) -> None:
        """With only two values, there's a single return — not enough for Sharpe."""
        pv = np.array([100.0, 110.0])
        assert compute_sharpe_ratio(pv) == pytest.approx(0.0)


class TestComputeCumulativeProfitRatio:
    def test_gain(self) -> None:
        pv = np.array([100.0, 110.0, 120.0])
        # 120 / 100 = 1.2
        assert compute_cumulative_profit_ratio(pv) == pytest.approx(1.2)

    def test_loss(self) -> None:
        pv = np.array([100.0, 90.0, 80.0])
        # 80 / 100 = 0.8
        assert compute_cumulative_profit_ratio(pv) == pytest.approx(0.8)

    def test_no_change(self) -> None:
        pv = np.array([100.0, 100.0])
        assert compute_cumulative_profit_ratio(pv) == pytest.approx(1.0)


class TestComputeMaxDrawdown:
    def test_no_drawdown(self) -> None:
        pv = np.array([100.0, 110.0, 120.0, 130.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.0)

    def test_known_25_percent_drawdown(self) -> None:
        # Peak at 200, trough at 150 -> drawdown = 50/200 = 0.25
        pv = np.array([100.0, 200.0, 150.0, 180.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.25)

    def test_full_loss(self) -> None:
        pv = np.array([100.0, 50.0, 0.001])
        # Peak 100, trough 0.001 -> (100 - 0.001)/100 ≈ 0.99999
        assert compute_max_drawdown(pv) == pytest.approx(0.99999, rel=1e-3)

    def test_single_value_returns_zero(self) -> None:
        pv = np.array([100.0])
        assert compute_max_drawdown(pv) == pytest.approx(0.0)


class TestPeriodsPerYearForTimeframe:
    """Hand-computed expected values — formula: MS_PER_YEAR / candle_ms."""

    def test_5m(self) -> None:
        # 5m candle_ms = 5 * 60_000 = 300_000
        # periods = 365*24*60*60*1000 / 300_000
        #         = 31_536_000_000 / 300_000
        #         = 105_120
        assert periods_per_year_for_timeframe("5m") == 105120

    def test_1h(self) -> None:
        # 1h candle_ms = 1 * 3_600_000 = 3_600_000
        # periods = 31_536_000_000 / 3_600_000 = 8760
        assert periods_per_year_for_timeframe("1h") == 8760

    def test_1d(self) -> None:
        # 1d candle_ms = 1 * 86_400_000 = 86_400_000
        # periods = 31_536_000_000 / 86_400_000 = 365
        assert periods_per_year_for_timeframe("1d") == 365

    def test_15m(self) -> None:
        # 15m candle_ms = 15 * 60_000 = 900_000
        # periods = 31_536_000_000 / 900_000 = 35_040
        assert periods_per_year_for_timeframe("15m") == 35040

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe unit"):
            periods_per_year_for_timeframe("1w")


class TestComputeSharpeRatioPeriodsPerYear:
    """Verify periods_per_year param wiring — using hand-computed scaling."""

    _RETURNS = [0.001, -0.0005, 0.002, 0.0015, -0.001]

    def _make_pv(self) -> np.ndarray:
        """Build portfolio values from simple arithmetic returns."""
        pv = [100.0]
        for r in self._RETURNS:
            pv.append(pv[-1] * (1 + r))
        return np.array(pv, dtype=np.float64)

    def test_default_equals_explicit_5m(self) -> None:
        """Default periods_per_year=105120 must match explicit 5m arg."""
        pv = self._make_pv()
        assert compute_sharpe_ratio(pv) == pytest.approx(
            compute_sharpe_ratio(pv, periods_per_year=105120)
        )

    def test_1h_vs_5m_scaling(self) -> None:
        """Sharpe(1h) / Sharpe(5m) == sqrt(8760 / 105120) == sqrt(1/12).

        Because annualization factor = sqrt(periods_per_year) and all
        other inputs are identical, the ratio of Sharpe values equals the
        ratio of their annualization factors:
            sqrt(8760) / sqrt(105120) = sqrt(8760/105120)
        """
        pv = self._make_pv()
        sharpe_5m = compute_sharpe_ratio(pv, periods_per_year=105120)
        sharpe_1h = compute_sharpe_ratio(pv, periods_per_year=8760)
        expected_ratio = np.sqrt(8760 / 105120)
        assert np.isclose(sharpe_1h / sharpe_5m, expected_ratio)
