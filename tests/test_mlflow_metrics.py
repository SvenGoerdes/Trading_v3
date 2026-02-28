"""Tests for MLflow metrics, new metric functions, and diagnostics callback.

Hand-computed expected values — no mocking of financial calculations.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.utils.metrics import (
    ANNUALIZATION_FACTOR_5MIN,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_profit_factor,
    compute_reward_kurtosis,
    compute_reward_skewness,
    compute_sortino_ratio,
    compute_total_return_pct,
    compute_win_rate,
)

# ── Helpers ────────────────────────────────────────────────────────────


def _make_portfolio_values(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def _make_trade_pnls(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


# ── Sortino Ratio Tests ───────────────────────────────────────────────


class TestComputeSortinoRatio:
    def test_constant_values_returns_zero(self) -> None:
        pv = _make_portfolio_values([100.0, 100.0, 100.0, 100.0])
        assert compute_sortino_ratio(pv) == pytest.approx(0.0)

    def test_only_positive_returns_returns_zero(self) -> None:
        """All positive returns -> no downside deviation -> 0."""
        pv = _make_portfolio_values([100.0, 110.0, 120.0, 130.0])
        assert compute_sortino_ratio(pv) == pytest.approx(0.0)

    def test_known_values(self) -> None:
        """Hand-compute Sortino for a series with negative returns."""
        # Use 4+ values for meaningful test (need 2+ negative returns for ddof=1)
        pv2 = _make_portfolio_values([100.0, 110.0, 95.0, 105.0, 90.0, 100.0])
        log_returns2 = np.diff(np.log(pv2))
        negative_returns2 = log_returns2[log_returns2 < 0]
        downside_std2 = float(np.std(negative_returns2, ddof=1))
        mean_return2 = float(np.mean(log_returns2))
        expected = mean_return2 / downside_std2 * float(ANNUALIZATION_FACTOR_5MIN)
        assert compute_sortino_ratio(pv2) == pytest.approx(expected)

    def test_single_value_returns_zero(self) -> None:
        pv = _make_portfolio_values([100.0])
        assert compute_sortino_ratio(pv) == pytest.approx(0.0)

    def test_two_values_returns_zero(self) -> None:
        pv = _make_portfolio_values([100.0, 110.0])
        assert compute_sortino_ratio(pv) == pytest.approx(0.0)


# ── Calmar Ratio Tests ────────────────────────────────────────────────


class TestComputeCalmarRatio:
    def test_no_drawdown_returns_zero(self) -> None:
        """Monotonically increasing -> max_dd = 0 -> Calmar = 0."""
        pv = _make_portfolio_values([100.0, 110.0, 120.0, 130.0])
        assert compute_calmar_ratio(pv) == pytest.approx(0.0)

    def test_known_values(self) -> None:
        """Hand-compute Calmar ratio."""
        # 100 -> 200 -> 150 -> 180
        # max_dd = (200-150)/200 = 0.25
        # total_return = 180/100 - 1 = 0.8
        # n_periods = 3
        # periods_per_year = 365*24*12 = 105120
        # annualized_return = 0.8 * (105120/3) = 28032.0
        # calmar = 28032.0 / 0.25 = 112128.0
        pv = _make_portfolio_values([100.0, 200.0, 150.0, 180.0])
        max_dd = compute_max_drawdown(pv)
        total_return = 180.0 / 100.0 - 1.0
        n_periods = 3
        periods_per_year = 365 * 24 * 12
        annualized_return = total_return * (periods_per_year / n_periods)
        expected = annualized_return / max_dd
        assert compute_calmar_ratio(pv) == pytest.approx(expected)

    def test_single_value_returns_zero(self) -> None:
        pv = _make_portfolio_values([100.0])
        assert compute_calmar_ratio(pv) == pytest.approx(0.0)


# ── Max Drawdown Duration Tests ───────────────────────────────────────


class TestComputeMaxDrawdownDuration:
    def test_no_drawdown(self) -> None:
        pv = _make_portfolio_values([100.0, 110.0, 120.0, 130.0])
        assert compute_max_drawdown_duration(pv) == 0

    def test_known_duration(self) -> None:
        # Peak at 200 (idx 1), trough at 150 (idx 2), recovery at 200 (idx 3)
        # Drawdown at idx 2 only -> duration = 1
        pv = _make_portfolio_values([100.0, 200.0, 150.0, 200.0])
        assert compute_max_drawdown_duration(pv) == 1

    def test_no_recovery(self) -> None:
        # Peak at 200 (idx 1), never recovers
        # Drawdown at idx 2, 3, 4 -> duration = 3
        pv = _make_portfolio_values([100.0, 200.0, 150.0, 160.0, 170.0])
        assert compute_max_drawdown_duration(pv) == 3

    def test_single_value(self) -> None:
        pv = _make_portfolio_values([100.0])
        assert compute_max_drawdown_duration(pv) == 0


# ── Total Return Tests ────────────────────────────────────────────────


class TestComputeTotalReturnPct:
    def test_gain(self) -> None:
        pv = _make_portfolio_values([100.0, 120.0])
        assert compute_total_return_pct(pv) == pytest.approx(20.0)

    def test_loss(self) -> None:
        pv = _make_portfolio_values([100.0, 80.0])
        assert compute_total_return_pct(pv) == pytest.approx(-20.0)

    def test_no_change(self) -> None:
        pv = _make_portfolio_values([100.0, 100.0])
        assert compute_total_return_pct(pv) == pytest.approx(0.0)


# ── Win Rate Tests ────────────────────────────────────────────────────


class TestComputeWinRate:
    def test_all_winners(self) -> None:
        pnls = _make_trade_pnls([10.0, 5.0, 1.0])
        assert compute_win_rate(pnls) == pytest.approx(1.0)

    def test_all_losers(self) -> None:
        pnls = _make_trade_pnls([-10.0, -5.0])
        assert compute_win_rate(pnls) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        # 2 winners, 1 loser -> 2/3
        pnls = _make_trade_pnls([10.0, -5.0, 3.0])
        assert compute_win_rate(pnls) == pytest.approx(2.0 / 3.0)

    def test_zero_trades(self) -> None:
        pnls = _make_trade_pnls([])
        assert compute_win_rate(pnls) == pytest.approx(0.0)

    def test_zero_pnl_not_a_win(self) -> None:
        """Trades with exactly 0.0 PnL are not wins."""
        pnls = _make_trade_pnls([0.0, 10.0])
        assert compute_win_rate(pnls) == pytest.approx(0.5)


# ── Profit Factor Tests ──────────────────────────────────────────────


class TestComputeProfitFactor:
    def test_known_values(self) -> None:
        # gross_profit = 10 + 5 = 15, gross_loss = |-3| + |-2| = 5
        # profit_factor = 15 / 5 = 3.0
        pnls = _make_trade_pnls([10.0, -3.0, 5.0, -2.0])
        assert compute_profit_factor(pnls) == pytest.approx(3.0)

    def test_no_losses_returns_inf(self) -> None:
        pnls = _make_trade_pnls([10.0, 5.0])
        assert compute_profit_factor(pnls) == float("inf")

    def test_no_wins_returns_zero(self) -> None:
        pnls = _make_trade_pnls([-10.0, -5.0])
        assert compute_profit_factor(pnls) == pytest.approx(0.0)

    def test_zero_trades(self) -> None:
        pnls = _make_trade_pnls([])
        assert compute_profit_factor(pnls) == pytest.approx(0.0)


# ── Reward Skewness / Kurtosis Tests ─────────────────────────────────


class TestRewardSkewnessKurtosis:
    def test_symmetric_returns_near_zero_skewness(self) -> None:
        rewards = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        assert compute_reward_skewness(rewards) == pytest.approx(0.0, abs=0.01)

    def test_right_skewed(self) -> None:
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float64)
        assert compute_reward_skewness(rewards) > 0

    def test_too_few_for_skewness(self) -> None:
        rewards = np.array([1.0, 2.0], dtype=np.float64)
        assert compute_reward_skewness(rewards) == pytest.approx(0.0)

    def test_too_few_for_kurtosis(self) -> None:
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert compute_reward_kurtosis(rewards) == pytest.approx(0.0)

    def test_kurtosis_on_uniform(self) -> None:
        """Uniform distribution has negative excess kurtosis."""
        rng = np.random.default_rng(42)
        rewards = rng.uniform(-1, 1, size=10_000)
        kurt = compute_reward_kurtosis(rewards)
        # Uniform excess kurtosis ≈ -1.2
        assert kurt < 0


# ── Finiteness Tests ─────────────────────────────────────────────────


class TestFiniteness:
    """All metrics return finite values on random data."""

    def test_all_metrics_finite_on_random_data(self) -> None:
        rng = np.random.default_rng(42)
        # Random walk portfolio values (always positive)
        returns = rng.normal(0.001, 0.01, 500)
        pv = 10000.0 * np.cumprod(1 + returns)
        pv = np.insert(pv, 0, 10000.0)

        assert np.isfinite(compute_sortino_ratio(pv))
        assert np.isfinite(compute_calmar_ratio(pv))
        assert np.isfinite(compute_max_drawdown_duration(pv))
        assert np.isfinite(compute_total_return_pct(pv))

        trade_pnls = rng.normal(0, 10, 50)
        assert np.isfinite(compute_win_rate(trade_pnls))
        pf = compute_profit_factor(trade_pnls)
        assert np.isfinite(pf) or pf == float("inf")

        rewards = rng.normal(0, 1, 100)
        assert np.isfinite(compute_reward_skewness(rewards))
        assert np.isfinite(compute_reward_kurtosis(rewards))


# ── TradingMetricsLogger Tests ───────────────────────────────────────


class TestTradingMetricsLogger:
    """Test that TradingMetricsLogger methods don't crash and produce artifacts."""

    def test_log_trading_performance_creates_artifacts(self, tmp_path: pytest.TempPathFactory) -> None:
        """Smoke test: log_trading_performance runs without error."""
        from src.utils.mlflow_metrics import TradingMetricsLogger

        symbols = ["BTC/USDT", "ETH/USDT"]
        logger = TradingMetricsLogger(symbols=symbols, initial_balance=10000.0)

        eval_result = {
            "portfolio_values": [10000.0, 10100.0, 10050.0, 10200.0],
            "rewards": [0.01, -0.005, 0.015],
            "trade_pnls": np.array([50.0, -30.0, 80.0], dtype=np.float64),
            "total_costs": 5.0,
            "trades_log": [
                {"step": 1, "symbol": "BTC/USDT", "side": "buy", "shares": 0.1, "price": 50000.0, "value": 5000.0},
                {
                    "step": 2, "symbol": "BTC/USDT", "side": "sell",
                    "shares": 0.1, "price": 50500.0, "value": 5050.0, "pnl": 50.0,
                },
            ],
        }

        with patch("src.utils.mlflow_metrics.mlflow"):
            metrics = logger.log_trading_performance(eval_result)

        assert "perf/sharpe_ratio" in metrics
        assert "perf/sortino_ratio" in metrics
        assert "perf/win_rate" in metrics
        assert np.isfinite(metrics["perf/sharpe_ratio"])

    def test_log_trading_performance_zero_trades(self) -> None:
        """Zero trades should not crash, win_rate and profit_factor should be 0."""
        from src.utils.mlflow_metrics import TradingMetricsLogger

        logger = TradingMetricsLogger(symbols=["BTC/USDT"], initial_balance=10000.0)

        eval_result = {
            "portfolio_values": [10000.0, 10000.0, 10000.0],
            "rewards": [0.0, 0.0],
            "trade_pnls": np.array([], dtype=np.float64),
            "total_costs": 0.0,
        }

        with patch("src.utils.mlflow_metrics.mlflow"):
            metrics = logger.log_trading_performance(eval_result)

        assert metrics["perf/win_rate"] == pytest.approx(0.0)
        assert metrics["perf/profit_factor"] == pytest.approx(0.0)
        assert metrics["perf/n_trades"] == pytest.approx(0.0)

    def test_log_action_analysis_smoke(self) -> None:
        """Action heatmap generation doesn't crash."""
        from src.utils.mlflow_metrics import TradingMetricsLogger

        logger = TradingMetricsLogger(symbols=["BTC/USDT", "ETH/USDT"], initial_balance=10000.0)

        eval_result = {
            "actions": np.random.default_rng(42).uniform(0, 1, (100, 2)),
        }

        with patch("src.utils.mlflow_metrics.mlflow"):
            logger.log_action_analysis(eval_result)

    def test_log_reward_distribution_smoke(self) -> None:
        """Reward distribution plots don't crash."""
        from src.utils.mlflow_metrics import TradingMetricsLogger

        logger = TradingMetricsLogger(symbols=["BTC/USDT"], initial_balance=10000.0)

        eval_result = {
            "rewards": np.random.default_rng(42).normal(0, 1, 200),
            "trade_pnls": np.random.default_rng(42).normal(0, 10, 50),
        }

        with patch("src.utils.mlflow_metrics.mlflow"):
            logger.log_reward_distribution(eval_result)

    def test_log_per_asset_breakdown_smoke(self) -> None:
        """Per-asset breakdown doesn't crash."""
        from src.utils.mlflow_metrics import TradingMetricsLogger

        logger = TradingMetricsLogger(symbols=["BTC/USDT", "ETH/USDT"], initial_balance=10000.0)

        eval_result = {
            "per_asset_pnls": {
                "BTC/USDT": [50.0, -30.0, 20.0],
                "ETH/USDT": [-10.0, 40.0],
            },
            "per_asset_trades": {"BTC/USDT": 3, "ETH/USDT": 2},
        }

        with patch("src.utils.mlflow_metrics.mlflow"):
            logger.log_per_asset_breakdown(eval_result)


# ── Trade Derivation Tests ────────────────────────────────────────────


class TestDeriveTradesFromHoldings:
    """Test the trade derivation logic in evaluation.py."""

    def test_buy_then_sell(self) -> None:
        from src.pipelines.training.evaluation import _derive_trades

        symbols = ["BTC/USDT"]
        holdings = [
            np.array([0.0]),   # step 0: no holdings
            np.array([1.0]),   # step 1: bought 1
            np.array([0.0]),   # step 2: sold 1
        ]
        prices = [
            np.array([100.0]),
            np.array([100.0]),
            np.array([110.0]),
        ]

        trades, per_asset_pnls, per_asset_trades = _derive_trades(holdings, prices, symbols)

        assert len(trades) == 2
        assert trades[0]["side"] == "buy"
        assert trades[0]["shares"] == pytest.approx(1.0)
        assert trades[1]["side"] == "sell"
        assert trades[1]["shares"] == pytest.approx(1.0)
        # PnL: bought at 100, sold at 110 -> pnl = 1 * (110 - 100) = 10
        assert trades[1]["pnl"] == pytest.approx(10.0)
        assert per_asset_pnls["BTC/USDT"] == [pytest.approx(10.0)]
        assert per_asset_trades["BTC/USDT"] == 2

    def test_no_trades(self) -> None:
        from src.pipelines.training.evaluation import _derive_trades

        symbols = ["BTC/USDT"]
        holdings = [np.array([0.0]), np.array([0.0])]
        prices = [np.array([100.0]), np.array([100.0])]

        trades, per_asset_pnls, per_asset_trades = _derive_trades(holdings, prices, symbols)

        assert len(trades) == 0
        assert per_asset_trades["BTC/USDT"] == 0

    def test_partial_sell(self) -> None:
        """Sell half the position."""
        from src.pipelines.training.evaluation import _derive_trades

        symbols = ["BTC/USDT"]
        holdings = [
            np.array([0.0]),
            np.array([2.0]),   # buy 2 at 100
            np.array([1.0]),   # sell 1 at 120
        ]
        prices = [
            np.array([100.0]),
            np.array([100.0]),
            np.array([120.0]),
        ]

        trades, per_asset_pnls, _ = _derive_trades(holdings, prices, symbols)

        assert len(trades) == 2
        # PnL: avg cost = 100, sold 1 at 120 -> pnl = 1 * (120-100) = 20
        assert trades[1]["pnl"] == pytest.approx(20.0)


# ── Callback Smoke Tests ─────────────────────────────────────────────


class TestMLflowDiagnosticsCallback:
    """Smoke test that callback instantiates and basic methods don't crash."""

    def test_callback_instantiation(self) -> None:
        from src.pipelines.training.callbacks import MLflowDiagnosticsCallback
        from src.utils.mlflow_metrics import TradingMetricsLogger

        logger = TradingMetricsLogger(symbols=["BTC/USDT"], initial_balance=10000.0)
        callback = MLflowDiagnosticsCallback(metrics_logger=logger)
        assert callback.diagnostics_step_interval == 1000
        assert callback.buffer_log_interval == 10000
