"""Tests for the full evaluation runner.

Regression coverage for the observation/action pairing bug: the collection
loop must record exactly one observation per executed action (the
observation the agent acted on), never the trailing terminal observation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.environments.trading_env import TradingEnv
from src.pipelines.training.evaluation import run_full_evaluation
from src.utils.mlflow_metrics import TradingMetricsLogger


def _make_env_data(
    n_steps: int = 80,
    n_assets: int = 2,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Create synthetic data for evaluation tests."""
    symbols = [f"ASSET{i}/USDT" for i in range(n_assets)]
    dates = pd.date_range("2024-01-01", periods=n_steps, freq="5min")
    rng = np.random.RandomState(7)

    data = {}
    for symbol in symbols:
        returns = rng.normal(0, 0.001, n_steps)
        prices = np.maximum(100.0 * (1 + np.cumsum(returns)), 1.0)
        data[symbol] = pd.DataFrame(
            {
                "close": prices,
                "rsi_14_norm": rng.randn(n_steps) * 0.1,
                "sma_20_norm": rng.randn(n_steps) * 0.1,
            },
            index=dates,
        )
    return data, symbols


def _make_zero_action_model(n_assets: int) -> MagicMock:
    """Stub model whose predict always returns a zero action vector."""
    model = MagicMock()
    model.predict.return_value = (np.zeros(n_assets, dtype=np.float32), None)
    return model


class TestRunFullEvaluation:
    def test_observations_pair_one_to_one_with_actions(self) -> None:
        data, symbols = _make_env_data()
        env = TradingEnv(
            data=data,
            symbols=symbols,
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=10,
            reward_scaling=1.0,
            max_position=1.0,
        )
        model = _make_zero_action_model(len(symbols))
        metrics_logger = MagicMock(spec=TradingMetricsLogger)
        price_data = {s: data[s]["close"].to_numpy() for s in symbols}

        run_full_evaluation(model, env, price_data, metrics_logger)

        eval_result = metrics_logger.log_ti_correlation.call_args.args[0]
        observations = eval_result["observations"]
        actions = eval_result["actions"]
        assert len(observations) > 0
        assert len(observations) == len(actions)
