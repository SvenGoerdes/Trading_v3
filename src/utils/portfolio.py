"""Shared portfolio execution logic.

Used by both the Gymnasium environment and the live trading bot.
These functions are the single source of truth for portfolio math.
If they diverge, backtest results become invalid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_portfolio_value(
    balance: float,
    holdings: NDArray[np.float64],
    prices: NDArray[np.float64],
) -> float:
    """Compute total portfolio value: cash + sum(holdings * prices).

    Args:
        balance: Current cash balance.
        holdings: Array of share counts per asset.
        prices: Array of current prices per asset.

    Returns:
        Total portfolio value.
    """
    assert balance >= 0, f"Negative balance: {balance}"
    assert np.all(holdings >= 0), f"Negative holdings: {holdings}"
    assert np.all(prices > 0), f"Non-positive prices: {prices}"

    return balance + float(np.sum(holdings * prices))


def compute_transaction_cost(
    trade_value: float,
    trading_fee_pct: float,
    slippage_pct: float,
) -> float:
    """Compute cost for a single trade.

    Args:
        trade_value: Absolute value of the trade (price * shares).
        trading_fee_pct: Trading fee as a fraction (e.g. 0.001).
        slippage_pct: Slippage as a fraction (e.g. 0.0005).

    Returns:
        Total cost of the trade.
    """
    assert trade_value >= 0, f"Negative trade value: {trade_value}"
    return trade_value * (trading_fee_pct + slippage_pct)


def compute_target_holdings(
    target_weights: NDArray[np.float64],
    portfolio_value: float,
    prices: NDArray[np.float64],
    max_position: float,
) -> NDArray[np.float64]:
    """Convert target weight vector to target share counts.

    Args:
        target_weights: Desired allocation weights per asset (0 to 1).
        portfolio_value: Current total portfolio value.
        prices: Current prices per asset.
        max_position: Maximum weight allowed per asset.

    Returns:
        Target number of shares per asset.
    """
    clipped_weights = np.clip(target_weights, 0.0, max_position)
    total_weight = float(np.sum(clipped_weights))
    if total_weight > 1.0:
        clipped_weights = clipped_weights / total_weight

    target_values = clipped_weights * portfolio_value
    target_shares = target_values / prices
    return np.maximum(target_shares, 0.0)


def execute_rebalance(
    current_balance: float,
    current_holdings: NDArray[np.float64],
    target_holdings: NDArray[np.float64],
    prices: NDArray[np.float64],
    trading_fee_pct: float,
    slippage_pct: float,
) -> tuple[float, NDArray[np.float64], float]:
    """Execute portfolio rebalance with sell-first priority.

    Sells are executed first to free up cash, then buys are executed.
    If insufficient cash for all buys, partial fills are applied
    proportionally.

    Args:
        current_balance: Current cash balance.
        current_holdings: Current share counts per asset.
        target_holdings: Desired share counts per asset.
        prices: Current prices per asset.
        trading_fee_pct: Trading fee as a fraction.
        slippage_pct: Slippage as a fraction.

    Returns:
        Tuple of (new_balance, new_holdings, total_cost).
    """
    assert current_balance >= 0, f"Negative balance: {current_balance}"
    assert np.all(current_holdings >= 0), f"Negative holdings: {current_holdings}"

    deltas = target_holdings - current_holdings
    new_holdings = current_holdings.copy()
    new_balance = current_balance
    total_cost = 0.0

    # Phase 1: Sell (deltas < 0)
    sell_mask = deltas < 0
    if np.any(sell_mask):
        sell_shares = np.abs(deltas[sell_mask])
        sell_values = sell_shares * prices[sell_mask]
        sell_costs = np.array([
            compute_transaction_cost(v, trading_fee_pct, slippage_pct) for v in sell_values
        ])
        new_holdings[sell_mask] -= sell_shares
        new_balance += float(np.sum(sell_values - sell_costs))
        total_cost += float(np.sum(sell_costs))

    # Phase 2: Buy (deltas > 0)
    buy_mask = deltas > 0
    if np.any(buy_mask):
        buy_shares = deltas[buy_mask]
        buy_values = buy_shares * prices[buy_mask]
        buy_costs = np.array([
            compute_transaction_cost(v, trading_fee_pct, slippage_pct) for v in buy_values
        ])
        total_buy_needed = float(np.sum(buy_values + buy_costs))

        if total_buy_needed <= new_balance:
            # Full fill
            new_holdings[buy_mask] += buy_shares
            new_balance -= total_buy_needed
            total_cost += float(np.sum(buy_costs))
        elif new_balance > 0:
            # Partial fill — scale down proportionally
            fill_ratio = new_balance / total_buy_needed
            partial_shares = buy_shares * fill_ratio
            partial_values = partial_shares * prices[buy_mask]
            partial_costs = np.array([
                compute_transaction_cost(v, trading_fee_pct, slippage_pct) for v in partial_values
            ])
            new_holdings[buy_mask] += partial_shares
            new_balance -= float(np.sum(partial_values + partial_costs))
            total_cost += float(np.sum(partial_costs))

    # Guard against floating-point drift below zero
    new_balance = max(new_balance, 0.0)

    assert new_balance >= 0, f"Negative balance after rebalance: {new_balance}"
    assert np.all(new_holdings >= 0), f"Negative holdings after rebalance: {new_holdings}"

    return new_balance, new_holdings, total_cost
