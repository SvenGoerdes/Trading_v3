"""Tests for the TradingEnv Gymnasium environment.

Target: 90%+ coverage. Hand-computed expected values where applicable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.environments.trading_env import TradingEnv


def _make_env_data(
    n_steps: int = 200,
    n_assets: int = 2,
    symbols: list[str] | None = None,
    base_price: float = 100.0,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Create synthetic data for testing the TradingEnv.

    Returns:
        Tuple of (data_dict, symbols_list).
    """
    if symbols is None:
        symbols = [f"ASSET{i}/USDT" for i in range(n_assets)]

    dates = pd.date_range("2024-01-01", periods=n_steps, freq="5min")
    data = {}

    rng = np.random.RandomState(42)
    for i, symbol in enumerate(symbols):
        # Random walk prices
        returns = rng.normal(0, 0.001, n_steps)
        prices = base_price * (1 + np.cumsum(returns))
        prices = np.maximum(prices, 1.0)  # ensure positive

        df = pd.DataFrame(
            {
                "close": prices,
                "rsi_14_norm": rng.randn(n_steps) * 0.1,
                "sma_20_norm": rng.randn(n_steps) * 0.1,
                "ema_20_norm": rng.randn(n_steps) * 0.1,
            },
            index=dates,
        )
        data[symbol] = df

    return data, symbols


def _make_env(
    n_steps: int = 200,
    n_assets: int = 2,
    window_size: int = 10,
    initial_balance: float = 10000.0,
    reward_scaling: float = 1.0,
    max_position: float = 1.0,
) -> TradingEnv:
    """Create a TradingEnv with synthetic data."""
    data, symbols = _make_env_data(n_steps=n_steps, n_assets=n_assets)
    return TradingEnv(
        data=data,
        symbols=symbols,
        initial_balance=initial_balance,
        trading_fee_pct=0.001,
        slippage_pct=0.0005,
        window_size=window_size,
        reward_scaling=reward_scaling,
        max_position=max_position,
    )


class TestSpaces:
    def test_action_space_shape(self) -> None:
        env = _make_env(n_assets=3)
        assert env.action_space.shape == (3,)

    def test_action_space_bounds(self) -> None:
        env = _make_env()
        assert env.action_space.low[0] == pytest.approx(-1.0)
        assert env.action_space.high[0] == pytest.approx(1.0)

    def test_observation_space_shape(self) -> None:
        env = _make_env(n_assets=2, window_size=10)
        # 10 * 2 * 3 features + 2 weights + 1 cash = 63
        expected = 10 * 2 * 3 + 2 + 1
        assert env.observation_space.shape == (expected,)


class TestReset:
    def test_returns_valid_observation(self) -> None:
        env = _make_env()
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)

    def test_initial_balance_set(self) -> None:
        env = _make_env(initial_balance=5000.0)
        env.reset()
        assert env.balance == pytest.approx(5000.0)

    def test_initial_holdings_zero(self) -> None:
        env = _make_env()
        env.reset()
        assert np.all(env.holdings == 0.0)

    def test_cash_ratio_is_one_at_start(self) -> None:
        env = _make_env(n_assets=2)
        obs, _ = env.reset()
        # Last element is cash_ratio, should be 1.0
        assert obs[-1] == pytest.approx(1.0)

    def test_portfolio_weights_zero_at_start(self) -> None:
        env = _make_env(n_assets=2)
        obs, _ = env.reset()
        # Second-to-last n_assets elements are portfolio weights
        weights = obs[-(env.n_assets + 1) : -1]
        assert np.allclose(weights, 0.0)


class TestStep:
    def test_zero_action_holds_cash(self) -> None:
        """Action of all -1 maps to weight 0, so no buying."""
        env = _make_env()
        env.reset()
        action = np.array([-1.0, -1.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        # With all-zero weights, no trades happen
        assert info["total_cost"] == pytest.approx(0.0)
        assert env.balance == pytest.approx(10000.0)

    def test_reward_is_finite(self) -> None:
        env = _make_env()
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert np.isfinite(reward)

    def test_reward_scaling(self) -> None:
        """Reward with scaling=2 should be 2x reward with scaling=1."""
        data, symbols = _make_env_data(n_steps=200, n_assets=2)

        env1 = TradingEnv(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=10, reward_scaling=1.0, max_position=1.0,
        )
        env2 = TradingEnv(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=10, reward_scaling=2.0, max_position=1.0,
        )

        env1.reset(seed=42)
        env2.reset(seed=42)
        action = np.array([0.5, 0.5], dtype=np.float32)
        _, r1, _, _, _ = env1.step(action)
        _, r2, _, _, _ = env2.step(action)

        if r1 != 0.0:
            assert r2 == pytest.approx(2.0 * r1, rel=1e-6)


class TestTermination:
    def test_full_episode_terminates(self) -> None:
        env = _make_env(n_steps=100, window_size=10)
        env.reset()
        done = False
        steps = 0
        while not done:
            action = np.zeros(env.n_assets, dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == env.max_steps


class TestInvariants:
    def test_balance_never_negative_random_actions(self) -> None:
        """Stress test: random actions should never cause negative balance."""
        env = _make_env(n_steps=200)
        env.reset(seed=42)
        rng = np.random.RandomState(123)
        done = False
        while not done:
            action = rng.uniform(-1, 1, size=env.n_assets).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            assert env.balance >= 0, f"Negative balance: {env.balance}"
            done = terminated or truncated

    def test_holdings_never_negative_random_actions(self) -> None:
        """Stress test: random actions should never cause negative holdings."""
        env = _make_env(n_steps=200)
        env.reset(seed=42)
        rng = np.random.RandomState(456)
        done = False
        while not done:
            action = rng.uniform(-1, 1, size=env.n_assets).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            assert np.all(env.holdings >= 0), f"Negative holdings: {env.holdings}"
            done = terminated or truncated

    def test_observation_always_finite(self) -> None:
        """All observations must be finite throughout an episode."""
        env = _make_env(n_steps=200)
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        rng = np.random.RandomState(789)
        done = False
        while not done:
            action = rng.uniform(-1, 1, size=env.n_assets).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            assert np.all(np.isfinite(obs))
            done = terminated or truncated


class TestObservation:
    def test_portfolio_weights_sum_with_cash(self) -> None:
        """Portfolio weights + cash ratio should sum to ~1.0."""
        env = _make_env(n_assets=2)
        env.reset()
        action = np.array([0.5, 0.3], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)

        weights = obs[-(env.n_assets + 1) : -1]
        cash_ratio = obs[-1]
        total = float(np.sum(weights)) + cash_ratio
        assert total == pytest.approx(1.0, abs=0.01)
