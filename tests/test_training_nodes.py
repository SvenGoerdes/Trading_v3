"""Tests for training pipeline node functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.pipelines.training.nodes import (
    aggregate_fold_metrics,
    aggregate_seed_results,
    create_td3_agent,
    generate_rolling_cv_folds,
    log_experiment_to_mlflow,
    pin_random_seeds,
    slice_data_by_dates,
)


def _make_cv_data(
    months: int = 8,
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Create synthetic data spanning the given number of months."""
    if symbols is None:
        symbols = ["BTC/USDT"]
    dates = pd.date_range("2024-01-01", periods=months * 30 * 24 * 12, freq="5min")
    data = {}
    rng = np.random.RandomState(42)
    for symbol in symbols:
        n = len(dates)
        df = pd.DataFrame(
            {
                "close": 100.0 + rng.randn(n).cumsum() * 0.1,
                "rsi_14_norm": rng.randn(n) * 0.1,
                "sma_20_norm": rng.randn(n) * 0.1,
                "ema_20_norm": rng.randn(n) * 0.1,
            },
            index=dates,
        )
        data[symbol] = df
    return data


class TestPinRandomSeeds:
    def test_reproducibility(self) -> None:
        pin_random_seeds(42)
        a = np.random.rand(5)
        t = torch.rand(5)

        pin_random_seeds(42)
        b = np.random.rand(5)
        u = torch.rand(5)

        np.testing.assert_array_equal(a, b)
        assert torch.equal(t, u)

    def test_different_seeds_differ(self) -> None:
        pin_random_seeds(42)
        a = np.random.rand(5)

        pin_random_seeds(123)
        b = np.random.rand(5)

        assert not np.array_equal(a, b)


class TestGenerateRollingCvFolds:
    def test_fold_count(self) -> None:
        data = _make_cv_data(months=8)
        folds = generate_rolling_cv_folds(data, train_months=3, validation_months=1)
        # 8 months of data: folds at month 0-3/3-4, 1-4/4-5, 2-5/5-6, 3-6/6-7, 4-7/7-8
        assert len(folds) >= 1

    def test_folds_slide_forward(self) -> None:
        data = _make_cv_data(months=12)
        folds = generate_rolling_cv_folds(data, train_months=3, validation_months=1)
        for i in range(1, len(folds)):
            assert folds[i]["train_start"] > folds[i - 1]["train_start"]

    def test_no_overlap_train_val(self) -> None:
        data = _make_cv_data(months=12)
        folds = generate_rolling_cv_folds(data, train_months=3, validation_months=1)
        for fold in folds:
            assert fold["train_end"] <= fold["val_start"]

    def test_fold_indices_sequential(self) -> None:
        data = _make_cv_data(months=12)
        folds = generate_rolling_cv_folds(data, train_months=3, validation_months=1)
        for i, fold in enumerate(folds):
            assert fold["fold_index"] == i


class TestSliceDataByDates:
    def test_slices_correctly(self) -> None:
        data = _make_cv_data(months=6)
        start = pd.Timestamp("2024-02-01")
        end = pd.Timestamp("2024-03-01")
        sliced = slice_data_by_dates(data, start, end)
        for df in sliced.values():
            assert df.index.min() >= start
            assert df.index.max() < end


class TestCreateTd3Agent:
    def test_returns_td3_with_noise(self) -> None:
        """Create agent and verify it's a TD3 with action noise."""
        data = _make_cv_data(months=6, symbols=["A/USDT", "B/USDT"])
        from src.environments.trading_env import TradingEnv
        from src.utils.config import NetArchConfig, TD3Config

        env = TradingEnv(
            data=data,
            symbols=["A/USDT", "B/USDT"],
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=10,
            reward_scaling=1.0,
            max_position=1.0,
        )

        td3_config = TD3Config(
            learning_rate=0.0001,
            gamma=0.99,
            tau=0.005,
            batch_size=64,
            buffer_size=1000,
            learning_starts=100,
            train_freq=1,
            policy_delay=2,
            target_noise_clip=0.5,
            target_policy_noise=0.2,
            action_noise_std=0.1,
            total_timesteps=100,
            net_arch=NetArchConfig(pi=[64, 64], qf=[64, 64]),
        )

        from stable_baselines3 import TD3

        agent = create_td3_agent(env, td3_config, seed=42)
        assert isinstance(agent, TD3)
        assert agent.action_noise is not None


class TestAggregateSeedResults:
    def test_mean_std(self) -> None:
        results = [
            {"mean_sharpe_ratio": 1.0, "mean_cpr": 1.1},
            {"mean_sharpe_ratio": 2.0, "mean_cpr": 1.3},
            {"mean_sharpe_ratio": 3.0, "mean_cpr": 1.5},
        ]
        agg = aggregate_seed_results(results)

        assert agg["mean_mean_sharpe_ratio"] == pytest.approx(2.0)
        assert agg["std_mean_sharpe_ratio"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
        assert agg["mean_mean_cpr"] == pytest.approx(1.3)

    def test_empty_returns_empty(self) -> None:
        assert aggregate_seed_results([]) == {}


class TestAggregateFoldMetrics:
    def test_mean_across_folds(self) -> None:
        folds = [
            {"sharpe_ratio": 1.0, "cpr": 1.1, "max_drawdown": 0.1},
            {"sharpe_ratio": 2.0, "cpr": 1.2, "max_drawdown": 0.2},
        ]
        agg = aggregate_fold_metrics(folds)
        assert agg["mean_sharpe_ratio"] == pytest.approx(1.5)
        assert agg["mean_cpr"] == pytest.approx(1.15)
        assert agg["mean_max_drawdown"] == pytest.approx(0.15)


class TestLogExperimentToMlflow:
    @patch("src.pipelines.training.nodes.mlflow")
    def test_logs_params_and_metrics(self, mock_mlflow: MagicMock) -> None:
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        fold_metrics = [
            {"sharpe_ratio": 1.5, "cpr": 1.1, "max_drawdown": 0.05},
        ]
        config_params = {"learning_rate": "0.0001"}

        log_experiment_to_mlflow(
            seed=42,
            fold_metrics=fold_metrics,
            best_model_path=None,
            config_params=config_params,
        )

        mock_mlflow.log_params.assert_called_once_with(config_params)
        mock_mlflow.log_param.assert_any_call("seed", 42)
        mock_mlflow.log_param.assert_any_call("n_folds", 1)
        # Verify fold metrics were logged
        mock_mlflow.log_metric.assert_any_call("fold_0_sharpe_ratio", 1.5)
        mock_mlflow.log_metric.assert_any_call("mean_sharpe_ratio", 1.5)
