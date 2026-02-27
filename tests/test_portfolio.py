"""Tests for portfolio execution logic.

Hand-computed expected values — no mocking of financial calculations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.portfolio import (
    compute_portfolio_value,
    compute_target_holdings,
    compute_transaction_cost,
    execute_rebalance,
)


class TestComputePortfolioValue:
    def test_cash_only(self) -> None:
        balance = 10000.0
        holdings = np.array([0.0, 0.0])
        prices = np.array([100.0, 50.0])
        result = compute_portfolio_value(balance, holdings, prices)
        assert result == pytest.approx(10000.0)

    def test_mixed_portfolio(self) -> None:
        balance = 5000.0
        holdings = np.array([10.0, 20.0])
        prices = np.array([100.0, 50.0])
        # 5000 + 10*100 + 20*50 = 5000 + 1000 + 1000 = 7000
        result = compute_portfolio_value(balance, holdings, prices)
        assert result == pytest.approx(7000.0)

    def test_zero_balance(self) -> None:
        balance = 0.0
        holdings = np.array([5.0])
        prices = np.array([200.0])
        # 0 + 5*200 = 1000
        result = compute_portfolio_value(balance, holdings, prices)
        assert result == pytest.approx(1000.0)

    def test_negative_balance_raises(self) -> None:
        with pytest.raises(AssertionError, match="Negative balance"):
            compute_portfolio_value(-1.0, np.array([0.0]), np.array([100.0]))

    def test_negative_holdings_raises(self) -> None:
        with pytest.raises(AssertionError, match="Negative holdings"):
            compute_portfolio_value(100.0, np.array([-1.0]), np.array([100.0]))


class TestComputeTransactionCost:
    def test_basic_cost(self) -> None:
        # 1000 * (0.001 + 0.0005) = 1000 * 0.0015 = 1.5
        result = compute_transaction_cost(1000.0, 0.001, 0.0005)
        assert result == pytest.approx(1.5)

    def test_zero_trade(self) -> None:
        result = compute_transaction_cost(0.0, 0.001, 0.0005)
        assert result == pytest.approx(0.0)

    def test_negative_trade_value_raises(self) -> None:
        with pytest.raises(AssertionError, match="Negative trade value"):
            compute_transaction_cost(-100.0, 0.001, 0.0005)


class TestComputeTargetHoldings:
    def test_equal_weights(self) -> None:
        weights = np.array([0.5, 0.5])
        portfolio_value = 10000.0
        prices = np.array([100.0, 200.0])
        # 0.5 * 10000 / 100 = 50, 0.5 * 10000 / 200 = 25
        result = compute_target_holdings(weights, portfolio_value, prices, 1.0)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(25.0)

    def test_max_position_clipping(self) -> None:
        weights = np.array([0.8, 0.8])
        portfolio_value = 10000.0
        prices = np.array([100.0, 100.0])
        # Clipped to [0.5, 0.5] (max_position=0.5), then sum=1.0, no renorm
        # But: clipped to [0.5, 0.5], sum=1.0 <= 1.0 so no renorm
        # 0.5 * 10000 / 100 = 50 each
        result = compute_target_holdings(weights, portfolio_value, prices, 0.5)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(50.0)

    def test_weights_exceeding_one_renormalized(self) -> None:
        weights = np.array([0.8, 0.8])
        portfolio_value = 10000.0
        prices = np.array([100.0, 100.0])
        # max_position=1.0, so no clipping. sum=1.6 > 1.0
        # renorm: [0.5, 0.5], then 0.5*10000/100=50 each
        result = compute_target_holdings(weights, portfolio_value, prices, 1.0)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(50.0)


class TestExecuteRebalance:
    def test_buy_from_cash(self) -> None:
        balance = 10000.0
        holdings = np.array([0.0])
        target = np.array([10.0])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0005

        new_bal, new_hold, cost = execute_rebalance(
            balance, holdings, target, prices, fee, slip
        )

        # Buy 10 shares at 100 = 1000 value
        # Cost = 1000 * 0.0015 = 1.5
        # Total deducted = 1000 + 1.5 = 1001.5
        # New balance = 10000 - 1001.5 = 8998.5
        assert new_hold[0] == pytest.approx(10.0)
        assert cost == pytest.approx(1.5)
        assert new_bal == pytest.approx(8998.5)

    def test_sell_then_buy(self) -> None:
        balance = 0.0
        holdings = np.array([10.0, 0.0])
        target = np.array([0.0, 5.0])
        prices = np.array([100.0, 100.0])
        fee = 0.001
        slip = 0.0

        new_bal, new_hold, cost = execute_rebalance(
            balance, holdings, target, prices, fee, slip
        )

        # Sell 10 shares of asset 0 at 100 = 1000 value
        # Sell cost = 1000 * 0.001 = 1.0
        # Balance after sell = 0 + 1000 - 1.0 = 999.0
        # Buy 5 shares of asset 1 at 100 = 500 value
        # Buy cost = 500 * 0.001 = 0.5
        # Total buy needed = 500 + 0.5 = 500.5
        # Balance after buy = 999.0 - 500.5 = 498.5
        assert new_hold[0] == pytest.approx(0.0)
        assert new_hold[1] == pytest.approx(5.0)
        assert cost == pytest.approx(1.5)
        assert new_bal == pytest.approx(498.5)

    def test_partial_fill(self) -> None:
        balance = 100.0
        holdings = np.array([0.0])
        target = np.array([10.0])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0

        new_bal, new_hold, cost = execute_rebalance(
            balance, holdings, target, prices, fee, slip
        )

        # Want to buy 10 shares at 100 = 1000 + cost 1.0 = 1001
        # Only have 100, fill_ratio = 100 / 1001 ≈ 0.09990...
        # Partial shares = 10 * 0.09990... ≈ 0.9990...
        # Partial value = 0.9990... * 100 ≈ 99.90...
        # Partial cost = 99.90... * 0.001 ≈ 0.0999...
        # Holdings > 0, balance ≈ 0
        assert new_hold[0] > 0
        assert new_hold[0] < 10.0
        assert new_bal == pytest.approx(0.0, abs=0.01)

    def test_no_trade(self) -> None:
        balance = 5000.0
        holdings = np.array([10.0])
        target = np.array([10.0])
        prices = np.array([100.0])

        new_bal, new_hold, cost = execute_rebalance(
            balance, holdings, target, prices, 0.001, 0.0005
        )

        assert new_bal == pytest.approx(5000.0)
        assert new_hold[0] == pytest.approx(10.0)
        assert cost == pytest.approx(0.0)

    def test_round_trip_cost(self) -> None:
        """Buying then selling same shares costs exactly 2 * fee_rate * value."""
        balance = 10000.0
        holdings = np.array([0.0])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0005
        fee_rate = fee + slip  # 0.0015

        # Buy 10 shares
        target_buy = np.array([10.0])
        bal1, hold1, cost1 = execute_rebalance(
            balance, holdings, target_buy, prices, fee, slip
        )

        # Sell all 10 shares back
        target_sell = np.array([0.0])
        bal2, hold2, cost2 = execute_rebalance(
            bal1, hold1, target_sell, prices, fee, slip
        )

        trade_value = 10.0 * 100.0  # 1000
        expected_total_cost = 2 * fee_rate * trade_value  # 2 * 0.0015 * 1000 = 3.0
        assert cost1 + cost2 == pytest.approx(expected_total_cost)
