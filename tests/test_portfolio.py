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

        new_bal, new_hold, cost, traded = execute_rebalance(
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

        new_bal, new_hold, cost, traded = execute_rebalance(
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

        new_bal, new_hold, cost, traded = execute_rebalance(
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

        new_bal, new_hold, cost, traded = execute_rebalance(
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
        bal1, hold1, cost1, traded1 = execute_rebalance(
            balance, holdings, target_buy, prices, fee, slip
        )

        # Sell all 10 shares back
        target_sell = np.array([0.0])
        bal2, hold2, cost2, traded2 = execute_rebalance(
            bal1, hold1, target_sell, prices, fee, slip
        )

        trade_value = 10.0 * 100.0  # 1000
        expected_total_cost = 2 * fee_rate * trade_value  # 2 * 0.0015 * 1000 = 3.0
        assert cost1 + cost2 == pytest.approx(expected_total_cost)


class TestRebalanceThreshold:
    """Tests for the no-trade band (rebalance_threshold) feature.

    All expected values are hand-computed.
    Setup: balance=1000, holdings=[10], price=[100]
      portfolio_value = 1000 + 10*100 = 2000
    """

    def test_band_skips_small_buy(self) -> None:
        """Trade below threshold is skipped entirely.

        Setup: balance=1000, holdings=[10], price=[100] => pv=2000.
        Target [10.5]: delta=0.5 shares, trade_value=0.5*100=$50=2.5% of pv.
        threshold=0.05 => min_trade=$100. $50 < $100 => skipped.
        """
        balance = 1000.0
        holdings = np.array([10.0])
        target = np.array([10.5])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0005

        new_bal, new_hold, cost, traded = execute_rebalance(
            balance, holdings, target, prices, fee, slip, rebalance_threshold=0.05
        )

        assert new_bal == pytest.approx(1000.0)
        assert new_hold[0] == pytest.approx(10.0)
        assert cost == pytest.approx(0.0)

    def test_band_executes_large_buy(self) -> None:
        """Trade above threshold is executed in full.

        Setup: balance=1000, holdings=[10], price=[100] => pv=2000.
        Target [15]: delta=5 shares, trade_value=5*100=$500=25% of pv.
        threshold=0.05 => min_trade=$100. $500 >= $100 => executes.

        Hand-computed:
          buy 5 shares at 100 = $500
          cost = 500 * (0.001 + 0.0005) = 500 * 0.0015 = 0.75
          total deducted = 500 + 0.75 = 500.75
          new_balance = 1000 - 500.75 = 499.25
        """
        balance = 1000.0
        holdings = np.array([10.0])
        target = np.array([15.0])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0005

        new_bal, new_hold, cost, traded = execute_rebalance(
            balance, holdings, target, prices, fee, slip, rebalance_threshold=0.05
        )

        assert new_hold[0] == pytest.approx(15.0)
        assert cost == pytest.approx(0.75)
        assert new_bal == pytest.approx(499.25)

    def test_threshold_zero_identical_to_no_threshold(self) -> None:
        """Explicit threshold=0.0 must produce bit-for-bit identical results.

        Uses the existing buy_from_cash scenario:
          balance=10000, holdings=[0], target=[10], price=[100], fee=0.001, slip=0.0005
          cost = 1000 * 0.0015 = 1.5
          new_balance = 10000 - 1001.5 = 8998.5
        """
        balance = 10000.0
        holdings = np.array([0.0])
        target = np.array([10.0])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0005

        new_bal_default, new_hold_default, cost_default, traded_default = execute_rebalance(
            balance, holdings, target, prices, fee, slip
        )
        new_bal_zero, new_hold_zero, cost_zero, traded_zero = execute_rebalance(
            balance, holdings, target, prices, fee, slip, rebalance_threshold=0.0
        )

        assert new_bal_zero == pytest.approx(new_bal_default)
        assert new_hold_zero[0] == pytest.approx(new_hold_default[0])
        assert cost_zero == pytest.approx(cost_default)

    def test_band_mixed_two_assets(self) -> None:
        """Two assets: one delta below band (skipped), one above (executed).

        Setup:
          balance=1000, holdings=[10, 20], prices=[100, 50]
          portfolio_value = 1000 + 10*100 + 20*50 = 1000 + 1000 + 1000 = 3000
          threshold=0.05 => min_trade = 0.05 * 3000 = $150

        Asset 0: target=10.5, delta=0.5, trade_value=0.5*100=$50 < $150 => SKIPPED
        Asset 1: target=25, delta=5, trade_value=5*50=$250 >= $150 => EXECUTED

        Hand-computed for asset 1 buy:
          buy 5 shares at 50 = $250
          cost = 250 * (0.001 + 0.0005) = 250 * 0.0015 = 0.375
          total deducted = 250 + 0.375 = 250.375
          new_balance = 1000 - 250.375 = 749.625
        """
        balance = 1000.0
        holdings = np.array([10.0, 20.0])
        target = np.array([10.5, 25.0])
        prices = np.array([100.0, 50.0])
        fee = 0.001
        slip = 0.0005

        new_bal, new_hold, cost, traded = execute_rebalance(
            balance, holdings, target, prices, fee, slip, rebalance_threshold=0.05
        )

        # Asset 0 skipped (trade too small)
        assert new_hold[0] == pytest.approx(10.0)
        # Asset 1 executed
        assert new_hold[1] == pytest.approx(25.0)
        assert cost == pytest.approx(0.375)
        assert new_bal == pytest.approx(749.625)

    def test_band_skips_small_sell(self) -> None:
        """Small sell delta below the band is also skipped.

        Setup: balance=1000, holdings=[10], price=[100] => pv=2000.
        Target [9.5]: delta=-0.5 shares, trade_value=0.5*100=$50=2.5% of pv.
        threshold=0.05 => min_trade=$100. $50 < $100 => skipped.
        """
        balance = 1000.0
        holdings = np.array([10.0])
        target = np.array([9.5])
        prices = np.array([100.0])
        fee = 0.001
        slip = 0.0005

        new_bal, new_hold, cost, traded = execute_rebalance(
            balance, holdings, target, prices, fee, slip, rebalance_threshold=0.05
        )

        assert new_bal == pytest.approx(1000.0)
        assert new_hold[0] == pytest.approx(10.0)
        assert cost == pytest.approx(0.0)


class TestTotalTradedValue:
    """Tests for the total_traded_value component of execute_rebalance.

    All expected values are hand-computed.
    """

    def test_buy_traded_value(self) -> None:
        """Full buy: traded value equals shares * price.

        Setup: balance=1000, holdings=[0], prices=[100], target=[5], no band.
        Buy 5 shares at 100 => trade_value = 5 * 100 = 500.
        total_traded_value must be exactly 500.0.
        """
        balance = 1000.0
        holdings = np.array([0.0])
        target = np.array([5.0])
        prices = np.array([100.0])

        _, _, _, traded = execute_rebalance(balance, holdings, target, prices, 0.001, 0.0005)

        assert traded == pytest.approx(500.0)

    def test_sell_traded_value(self) -> None:
        """Full sell: traded value equals shares * price.

        Setup: balance=0, holdings=[10], prices=[100], target=[5], no band.
        Sell 5 shares at 100 => trade_value = 5 * 100 = 500.
        total_traded_value must be exactly 500.0.
        """
        balance = 0.0
        holdings = np.array([10.0])
        target = np.array([5.0])
        prices = np.array([100.0])

        _, _, _, traded = execute_rebalance(balance, holdings, target, prices, 0.001, 0.0005)

        assert traded == pytest.approx(500.0)

    def test_band_skipped_trade_contributes_zero(self) -> None:
        """Trade below threshold is skipped; total_traded_value must be 0.0.

        Setup: balance=1000, holdings=[10], prices=[100] => pv=2000.
        Target=[10.5]: delta=0.5, trade_value=$50 < 5% of 2000=$100 => skipped.
        total_traded_value must be 0.0.
        """
        balance = 1000.0
        holdings = np.array([10.0])
        target = np.array([10.5])
        prices = np.array([100.0])

        _, _, _, traded = execute_rebalance(
            balance, holdings, target, prices, 0.001, 0.0005, rebalance_threshold=0.05
        )

        assert traded == pytest.approx(0.0)

    def test_partial_fill_traded_value_is_filled_amount(self) -> None:
        """Partial fill: total_traded_value equals the executed (filled) buy value.

        Setup: balance=200, holdings=[0], prices=[100], target=[10], no band.
        full_buy_needed = 10*100 + 10*100*0.001 = 1000 + 1.0 = 1001
        fill_ratio = 200 / 1001
        partial_shares = 10 * (200/1001)
        partial_value = partial_shares * 100 = 10 * 100 * (200/1001) = 1000 * (200/1001)

        Hand-computed:
          fill_ratio = 200 / 1001 ≈ 0.19980019980...
          partial_value = 1000 * (200/1001) = 200000/1001 ≈ 199.800199800...

        total_traded_value must equal partial_value (not the intended 1000).
        """
        balance = 200.0
        holdings = np.array([0.0])
        target = np.array([10.0])
        prices = np.array([100.0])

        _, _, _, traded = execute_rebalance(balance, holdings, target, prices, 0.001, 0.0)

        # full_buy_needed = 1000 + 1000*0.001 = 1001
        # fill_ratio = 200 / 1001
        # partial_value = 10 * fill_ratio * 100
        expected_fill_ratio = 200.0 / 1001.0
        expected_partial_value = 10.0 * expected_fill_ratio * 100.0
        assert traded == pytest.approx(expected_partial_value)
        # Sanity: must be less than the intended 1000
        assert traded < 1000.0
