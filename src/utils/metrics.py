"""Performance metrics for trading strategies.

All functions operate on portfolio value arrays and return scalar metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
