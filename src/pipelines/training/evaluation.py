"""Full evaluation runner for trained agents.

Runs a deterministic episode, collects all per-step data, derives
trade events, and invokes all Layer 2 + Layer 3 logging methods on
the TradingMetricsLogger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import TD3

from src.environments.trading_env import TradingEnv
from src.utils.logger import get_logger
from src.utils.mlflow_metrics import TradingMetricsLogger

logger = get_logger(__name__)


def _derive_trades(
    holdings_history: list[NDArray[np.float64]],
    prices_history: list[NDArray[np.float64]],
    symbols: list[str],
) -> tuple[list[dict[str, Any]], dict[str, list[float]], dict[str, int]]:
    """Derive trade events from holdings changes.

    A trade is detected when holdings change for a given asset between steps.

    Args:
        holdings_history: List of holdings arrays per step.
        prices_history: List of price arrays per step.
        symbols: List of symbol names.

    Returns:
        Tuple of (trades_log, per_asset_pnls, per_asset_trades).
    """
    trades_log: list[dict[str, Any]] = []
    per_asset_pnls: dict[str, list[float]] = {s: [] for s in symbols}
    per_asset_trades: dict[str, int] = {s: 0 for s in symbols}

    # Track open position cost basis per asset
    cost_basis: dict[str, float] = {s: 0.0 for s in symbols}
    position_shares: dict[str, float] = {s: 0.0 for s in symbols}

    for step in range(1, len(holdings_history)):
        prev_holdings = holdings_history[step - 1]
        curr_holdings = holdings_history[step]
        prices = prices_history[step]

        for i, symbol in enumerate(symbols):
            delta = curr_holdings[i] - prev_holdings[i]
            if abs(delta) < 1e-12:
                continue

            price = float(prices[i])

            if delta > 0:
                # Buy
                trade_value = delta * price
                cost_basis[symbol] += trade_value
                position_shares[symbol] += delta

                trades_log.append({
                    "step": step,
                    "symbol": symbol,
                    "side": "buy",
                    "shares": float(delta),
                    "price": price,
                    "value": trade_value,
                })
                per_asset_trades[symbol] += 1

            elif delta < 0:
                # Sell
                sold_shares = abs(delta)
                sell_value = sold_shares * price

                # Compute PnL based on average cost basis
                if position_shares[symbol] > 0:
                    avg_cost = cost_basis[symbol] / position_shares[symbol]
                    pnl = sold_shares * (price - avg_cost)

                    # Update cost basis
                    cost_basis[symbol] -= sold_shares * avg_cost
                    position_shares[symbol] -= sold_shares
                else:
                    pnl = 0.0

                per_asset_pnls[symbol].append(pnl)
                per_asset_trades[symbol] += 1

                trades_log.append({
                    "step": step,
                    "symbol": symbol,
                    "side": "sell",
                    "shares": float(sold_shares),
                    "price": price,
                    "value": sell_value,
                    "pnl": pnl,
                })

    return trades_log, per_asset_pnls, per_asset_trades


def run_full_evaluation(
    model: TD3,
    test_env: TradingEnv,
    price_data: dict[str, NDArray[np.float64]],
    metrics_logger: TradingMetricsLogger,
) -> dict[str, float]:
    """Run agent deterministically and log all diagnostics.

    Collects per-step actions, portfolio values, rewards, holdings,
    and observations. Derives trade list and invokes all Layer 2 + 3
    logging methods.

    Args:
        model: Trained TD3 agent.
        test_env: Evaluation environment.
        price_data: Dict mapping symbol to close price arrays (for timing/regime).
        metrics_logger: TradingMetricsLogger instance.

    Returns:
        Summary dict with key metrics.
    """
    obs, _ = test_env.reset()

    portfolio_values = [test_env.initial_balance]
    rewards_list: list[float] = []
    actions_list: list[NDArray[np.float64]] = []
    holdings_list: list[NDArray[np.float64]] = [np.zeros(test_env.n_assets, dtype=np.float64)]
    observations_list: list[NDArray[np.float32]] = [obs.copy()]
    prices_list: list[NDArray[np.float64]] = [
        test_env.prices[test_env.window_size].copy()
    ]
    total_costs = 0.0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        portfolio_values.append(info["portfolio_value"])
        rewards_list.append(float(reward))
        actions_list.append(info["action"].copy())
        holdings_list.append(info["holdings"].copy())
        observations_list.append(obs.copy())
        total_costs += info["total_cost"]

        # Get current prices for trade derivation
        data_idx = test_env.window_size + test_env.current_step
        if data_idx < test_env.n_steps:
            prices_list.append(test_env.prices[data_idx].copy())
        else:
            prices_list.append(test_env.prices[-1].copy())

    # Derive trades from holdings changes
    trades_log, per_asset_pnls, per_asset_trades = _derive_trades(
        holdings_list, prices_list, test_env.symbols
    )

    # Collect all trade PnLs
    all_trade_pnls = []
    for pnls in per_asset_pnls.values():
        all_trade_pnls.extend(pnls)

    eval_result: dict[str, Any] = {
        "portfolio_values": portfolio_values,
        "rewards": rewards_list,
        "actions": actions_list,
        "observations": observations_list,
        "trade_pnls": np.array(all_trade_pnls, dtype=np.float64),
        "trades_log": trades_log,
        "total_costs": total_costs,
        "per_asset_pnls": per_asset_pnls,
        "per_asset_trades": per_asset_trades,
    }

    # Layer 2: Trading performance
    summary = metrics_logger.log_trading_performance(eval_result)

    # Layer 3: Behavioral analysis
    metrics_logger.log_per_asset_breakdown(eval_result)
    metrics_logger.log_action_analysis(eval_result)
    metrics_logger.log_reward_distribution(eval_result)
    metrics_logger.log_trade_timing_analysis(eval_result, price_data)
    metrics_logger.log_ti_correlation(eval_result)
    metrics_logger.log_regime_analysis(eval_result, price_data)

    logger.info(
        "Full evaluation complete: %d steps, %d trades, final PV=%.2f",
        len(rewards_list),
        len(trades_log),
        portfolio_values[-1],
    )

    return summary
