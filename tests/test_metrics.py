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
