"""Performance metrics for trading strategies.

All functions operate on portfolio value arrays and return scalar metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 5-minute candles: 12 per hour * 24 hours * 365 days
ANNUALIZATION_FACTOR_5MIN = np.sqrt(365 * 24 * 12)


def compute_sharpe_ratio(
    portfolio_values: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio from portfolio value series.

    Uses log returns annualized for 5-minute candle frequency.

    Args:
        portfolio_values: Array of portfolio values over time.
        risk_free_rate: Risk-free rate per period (default 0).

    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std of returns is zero
        or if fewer than 2 values.
    """
    if len(portfolio_values) < 2:
        return 0.0

    log_returns = np.diff(np.log(portfolio_values))
    if len(log_returns) < 2:
        return 0.0

    excess_returns = log_returns - risk_free_rate
    std = float(np.std(excess_returns, ddof=1))

    if std == 0.0 or not np.isfinite(std):
        return 0.0

    mean_return = float(np.mean(excess_returns))
    return mean_return / std * float(ANNUALIZATION_FACTOR_5MIN)


def compute_cumulative_profit_ratio(
    portfolio_values: NDArray[np.float64],
) -> float:
    """Compute cumulative profit ratio (final / initial).

    Args:
        portfolio_values: Array of portfolio values over time.

    Returns:
        Ratio of final to initial portfolio value.
    """
    assert len(portfolio_values) >= 1, "Need at least one portfolio value"
    assert portfolio_values[0] > 0, "Initial portfolio value must be positive"
    return float(portfolio_values[-1] / portfolio_values[0])


def compute_max_drawdown(
    portfolio_values: NDArray[np.float64],
) -> float:
    """Compute maximum drawdown as a positive fraction.

    Args:
        portfolio_values: Array of portfolio values over time.

    Returns:
        Maximum peak-to-trough decline as a fraction (0 to 1).
    """
    if len(portfolio_values) < 2:
        return 0.0

    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    return float(np.max(drawdowns))


def compute_sortino_ratio(
    portfolio_values: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sortino ratio from portfolio value series.

    Like Sharpe but uses only downside deviation (negative returns).

    Args:
        portfolio_values: Array of portfolio values over time.
        risk_free_rate: Risk-free rate per period (default 0).

    Returns:
        Annualized Sortino ratio. Returns 0.0 if downside std is zero
        or if fewer than 2 values.
    """
    if len(portfolio_values) < 2:
        return 0.0

    log_returns = np.diff(np.log(portfolio_values))
    if len(log_returns) < 2:
        return 0.0

    excess_returns = log_returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = float(np.std(downside_returns, ddof=1))
    if downside_std == 0.0 or not np.isfinite(downside_std):
        return 0.0

    mean_return = float(np.mean(excess_returns))
    return mean_return / downside_std * float(ANNUALIZATION_FACTOR_5MIN)


def compute_calmar_ratio(
    portfolio_values: NDArray[np.float64],
) -> float:
    """Compute Calmar ratio: annualized return / max drawdown.

    Args:
        portfolio_values: Array of portfolio values over time.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero or data too short.
    """
    if len(portfolio_values) < 2:
        return 0.0

    max_dd = compute_max_drawdown(portfolio_values)
    if max_dd == 0.0:
        return 0.0

    total_return = portfolio_values[-1] / portfolio_values[0] - 1.0
    n_periods = len(portfolio_values) - 1
    # Annualize: periods per year for 5-min candles = 365 * 24 * 12
    periods_per_year = 365 * 24 * 12
    annualized_return = total_return * (periods_per_year / n_periods)

    if not np.isfinite(annualized_return):
        return 0.0

    return annualized_return / max_dd


def compute_max_drawdown_duration(
    portfolio_values: NDArray[np.float64],
) -> int:
    """Compute maximum drawdown duration in steps.

    Duration is measured from peak to recovery (or end if no recovery).

    Args:
        portfolio_values: Array of portfolio values over time.

    Returns:
        Number of steps in the longest drawdown period.
    """
    if len(portfolio_values) < 2:
        return 0

    cumulative_max = np.maximum.accumulate(portfolio_values)
    in_drawdown = portfolio_values < cumulative_max

    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def compute_total_return_pct(
    portfolio_values: NDArray[np.float64],
) -> float:
    """Compute total return as a percentage.

    Args:
        portfolio_values: Array of portfolio values over time.

    Returns:
        Total return percentage: (final - initial) / initial * 100.
    """
    assert len(portfolio_values) >= 1, "Need at least one portfolio value"
    assert portfolio_values[0] > 0, "Initial portfolio value must be positive"
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100.0


def compute_win_rate(
    trade_pnls: NDArray[np.float64],
) -> float:
    """Compute win rate: fraction of trades with positive PnL.

    Args:
        trade_pnls: Array of per-trade PnL values.

    Returns:
        Win rate as a fraction (0 to 1). Returns 0.0 if no trades.
    """
    if len(trade_pnls) == 0:
        logger.warning("compute_win_rate called with zero trades, returning 0.0")
        return 0.0

    winning = np.sum(trade_pnls > 0)
    return float(winning / len(trade_pnls))


def compute_profit_factor(
    trade_pnls: NDArray[np.float64],
) -> float:
    """Compute profit factor: gross_profit / gross_loss.

    Args:
        trade_pnls: Array of per-trade PnL values.

    Returns:
        Profit factor. Returns 0.0 if no trades or no losses.
    """
    if len(trade_pnls) == 0:
        logger.warning("compute_profit_factor called with zero trades, returning 0.0")
        return 0.0

    gross_profit = float(np.sum(trade_pnls[trade_pnls > 0]))
    gross_loss = float(np.abs(np.sum(trade_pnls[trade_pnls < 0])))

    if gross_loss == 0.0:
        if gross_profit > 0.0:
            return float("inf")
        return 0.0

    return gross_profit / gross_loss


def compute_reward_skewness(
    rewards: NDArray[np.float64],
) -> float:
    """Compute skewness of reward distribution.

    Args:
        rewards: Array of per-step rewards.

    Returns:
        Skewness value. Returns 0.0 if fewer than 3 values.
    """
    if len(rewards) < 3:
        return 0.0

    result = float(scipy_stats.skew(rewards, bias=False))
    return result if np.isfinite(result) else 0.0


def compute_reward_kurtosis(
    rewards: NDArray[np.float64],
) -> float:
    """Compute excess kurtosis of reward distribution.

    Args:
        rewards: Array of per-step rewards.

    Returns:
        Excess kurtosis value. Returns 0.0 if fewer than 4 values.
    """
    if len(rewards) < 4:
        return 0.0

    result = float(scipy_stats.kurtosis(rewards, bias=False))
    return result if np.isfinite(result) else 0.0
