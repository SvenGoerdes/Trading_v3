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


class TestRebalanceThresholdWiring:
    """Verify rebalance_threshold is accepted by TradingEnv and wired to step()."""

    def test_default_threshold_zero(self) -> None:
        """Default TradingEnv has rebalance_threshold=0.0."""
        env = _make_env()
        assert env.rebalance_threshold == pytest.approx(0.0)

    def test_threshold_stored(self) -> None:
        """Explicit rebalance_threshold is stored on the env."""
        data, symbols = _make_env_data(n_steps=200, n_assets=1)
        env = TradingEnv(
            data=data,
            symbols=symbols,
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=10,
            reward_scaling=1.0,
            max_position=1.0,
            rebalance_threshold=0.05,
        )
        assert env.rebalance_threshold == pytest.approx(0.05)

    def test_high_threshold_suppresses_small_trade(self) -> None:
        """With a very high threshold, a modest action change produces zero cost.

        We create a 1-asset env with a fixed price (all steps same price) and
        an extremely high rebalance_threshold (1.0 = 100% of portfolio value).
        Any non-zero target will have a trade value < 100% of PV only when
        the action is very small — but with threshold=1.0 the minimum trade
        required is *equal to* the entire portfolio value, so nearly every
        trade is skipped.

        We use all-zero action (weight=0.0) after a previous step that also
        had weight=0.0 so holdings are 0; then a small positive action that
        maps to weight=0.01 (0.5% of PV buys 0.01*PV/$price shares).
        With threshold=1.0, min_trade=PV, and trade_value=0.01*PV < PV => skipped.
        """
        data, symbols = _make_env_data(n_steps=200, n_assets=1)

        env = TradingEnv(
            data=data,
            symbols=symbols,
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=10,
            reward_scaling=1.0,
            max_position=1.0,
            rebalance_threshold=1.0,  # impossible to meet — every trade skipped
        )
        env.reset()

        # Action maps [-1,1] -> [0,1]: action=-0.98 -> weight=0.01 (1% of PV)
        action = np.array([-0.98], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        # With threshold=1.0, min_trade=10000 >> any actual trade value => skipped
        assert info["total_cost"] == pytest.approx(0.0)
        assert env.balance == pytest.approx(10000.0)


class TestCrossSectionalMomentum:
    """Tests for the config-gated cross-sectional momentum observation feature."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_momentum_env(
        self,
        n_steps: int = 200,
        n_assets: int = 2,
        window_size: int = 20,
        momentum_window: int = 10,
        cross_sectional_momentum: bool = True,
        prices_override: np.ndarray | None = None,
    ) -> TradingEnv:
        """Create a TradingEnv with momentum feature optionally enabled."""
        symbols = [f"ASSET{i}/USDT" for i in range(n_assets)]
        dates = pd.date_range("2024-01-01", periods=n_steps, freq="5min")
        rng = np.random.RandomState(7)
        data = {}
        for i, symbol in enumerate(symbols):
            if prices_override is not None:
                prices = prices_override[:, i]
            else:
                returns = rng.normal(0, 0.001, n_steps)
                prices = 100.0 * (1 + np.cumsum(returns))
                prices = np.maximum(prices, 1.0)

            df = pd.DataFrame(
                {
                    "close": prices,
                    "rsi_14_norm": rng.randn(n_steps) * 0.1,
                },
                index=dates,
            )
            data[symbol] = df
        return TradingEnv(
            data=data,
            symbols=symbols,
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=window_size,
            reward_scaling=1.0,
            max_position=1.0,
            cross_sectional_momentum=cross_sectional_momentum,
            momentum_window=momentum_window,
        )

    # ------------------------------------------------------------------
    # (a) Disabled: obs shape unchanged AND bit-identical to old-style env
    # ------------------------------------------------------------------

    def test_disabled_obs_shape_unchanged(self) -> None:
        """With flag off, obs size equals the pre-feature layout."""
        data, symbols = _make_env_data(n_steps=200, n_assets=2)
        env_old = TradingEnv(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=10, reward_scaling=1.0, max_position=1.0,
        )
        env_new = TradingEnv(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=10, reward_scaling=1.0, max_position=1.0,
            cross_sectional_momentum=False,
        )
        assert env_old.observation_space.shape == env_new.observation_space.shape

    def test_disabled_obs_bit_identical(self) -> None:
        """Env built without new params produces bit-identical obs to flag=False env."""
        data, symbols = _make_env_data(n_steps=200, n_assets=2)
        kwargs = dict(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=10, reward_scaling=1.0, max_position=1.0,
        )
        env_old = TradingEnv(**kwargs)
        env_new = TradingEnv(**kwargs, cross_sectional_momentum=False, momentum_window=12)

        obs_old, _ = env_old.reset(seed=0)
        obs_new, _ = env_new.reset(seed=0)

        np.testing.assert_array_equal(obs_old, obs_new)

    # ------------------------------------------------------------------
    # (b) Enabled: obs size grows by exactly n_assets
    # ------------------------------------------------------------------

    def test_enabled_obs_size_grows_by_n_assets(self) -> None:
        """With flag on, observation size = old size + n_assets."""
        n_assets = 3
        window_size = 20
        data, symbols = _make_env_data(n_steps=200, n_assets=n_assets)
        n_features = 3  # rsi_14_norm, sma_20_norm, ema_20_norm

        env_off = TradingEnv(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=window_size, reward_scaling=1.0, max_position=1.0,
            cross_sectional_momentum=False,
        )
        env_on = TradingEnv(
            data=data, symbols=symbols, initial_balance=10000.0,
            trading_fee_pct=0.001, slippage_pct=0.0005,
            window_size=window_size, reward_scaling=1.0, max_position=1.0,
            cross_sectional_momentum=True, momentum_window=10,
        )

        expected_off = window_size * n_assets * n_features + n_assets + 1
        assert env_off.observation_space.shape == (expected_off,)
        assert env_on.observation_space.shape == (expected_off + n_assets,)

    # ------------------------------------------------------------------
    # (c) Hand-computed 3-asset toy
    # ------------------------------------------------------------------

    def test_hand_computed_momentum_values(self) -> None:
        """Momentum values match hand-computed z-scored returns.

        Prices are constant per asset across all time steps (known values),
        so returns over any window are exact and easy to verify by hand.
        We then spike each asset's price at a single point to create a
        known non-trivial return over the momentum_window.

        Setup:
          3 assets, n_steps=100, window_size=20, momentum_window=5
          We set prices so that:
            Asset 0: doubles  over [idx_past..idx_now] => raw_ret[0] = 1.0
            Asset 1: stays flat                        => raw_ret[1] = 0.0
            Asset 2: loses half                        => raw_ret[2] = -0.5

          mean = (1.0 + 0.0 + (-0.5)) / 3 = 0.5 / 3 ≈ 0.16667
          std  = np.std([1.0, 0.0, -0.5])            ≈ 0.62361
          mom[0] = (1.0   - mean) / (std + 1e-8)
          mom[1] = (0.0   - mean) / (std + 1e-8)
          mom[2] = (-0.5  - mean) / (std + 1e-8)
        """
        n_steps = 100
        n_assets = 3
        window_size = 20
        momentum_window = 5

        # Build a price matrix where we control prices at idx_now and idx_past.
        # For step=0 the agent acts at data_idx=window_size=20.
        # idx_now  = min(20, n_steps-1) = 20
        # idx_past = max(20-5, 0)       = 15
        # We set all prices to base=1.0 everywhere, then override bars 15 and 20.
        prices = np.ones((n_steps, n_assets), dtype=np.float64)
        # Asset 0: price doubles from bar 15 to bar 20
        prices[15, 0] = 1.0
        prices[20, 0] = 2.0
        # Asset 1: flat (stays at 1.0 everywhere — raw_ret = 0)
        # Asset 2: price halves from bar 15 to bar 20
        prices[15, 2] = 2.0
        prices[20, 2] = 1.0

        raw_ret = np.array([
            prices[20, 0] / prices[15, 0] - 1.0,  # 1.0
            prices[20, 1] / prices[15, 1] - 1.0,  # 0.0
            prices[20, 2] / prices[15, 2] - 1.0,  # -0.5
        ])
        mean_ret = raw_ret.mean()
        std_ret = raw_ret.std()
        expected_mom = (raw_ret - mean_ret) / (std_ret + 1e-8)

        env = self._make_momentum_env(
            n_steps=n_steps,
            n_assets=n_assets,
            window_size=window_size,
            momentum_window=momentum_window,
            cross_sectional_momentum=True,
            prices_override=prices,
        )

        obs, _ = env.reset()
        # After reset, current_step=0, data_idx=window_size=20
        # Momentum block is the last n_assets elements of obs
        actual_mom = obs[-n_assets:].astype(np.float64)

        np.testing.assert_allclose(actual_mom, expected_mom, rtol=1e-5, atol=1e-6)

    # ------------------------------------------------------------------
    # (d) Zero cross-asset std: all assets identical => mom values ≈ 0
    # ------------------------------------------------------------------

    def test_zero_std_all_assets_identical_prices(self) -> None:
        """When all assets have identical returns, momentum values are ≈ 0 and finite."""
        n_steps = 100
        n_assets = 3
        # Same price ramp for every asset => identical returns => std = 0
        prices = np.tile(np.linspace(100.0, 110.0, n_steps)[:, None], (1, n_assets))

        env = self._make_momentum_env(
            n_steps=n_steps,
            n_assets=n_assets,
            window_size=20,
            momentum_window=5,
            cross_sectional_momentum=True,
            prices_override=prices,
        )
        obs, _ = env.reset()
        mom = obs[-n_assets:]

        assert np.all(np.isfinite(mom)), "Momentum values must be finite when std=0"
        np.testing.assert_allclose(mom, 0.0, atol=1e-5)

    # ------------------------------------------------------------------
    # (e) No look-ahead: future price changes don't affect current obs
    # ------------------------------------------------------------------

    def test_no_lookahead_future_jump_does_not_affect_current_obs(self) -> None:
        """A price spike at idx_now+1 must not change the momentum at idx_now."""
        n_steps = 100
        n_assets = 2
        window_size = 20
        momentum_window = 5
        # current_step=0 => data_idx=20 => idx_now=20, idx_past=15

        # Base prices: uniform across both assets
        prices_base = np.ones((n_steps, n_assets), dtype=np.float64) * 100.0

        # Variant: same base but with a massive spike AFTER the current decision bar
        prices_spike = prices_base.copy()
        prices_spike[21, 0] = 9999.0  # one bar into the future

        env_base = self._make_momentum_env(
            n_steps=n_steps, n_assets=n_assets, window_size=window_size,
            momentum_window=momentum_window, cross_sectional_momentum=True,
            prices_override=prices_base,
        )
        env_spike = self._make_momentum_env(
            n_steps=n_steps, n_assets=n_assets, window_size=window_size,
            momentum_window=momentum_window, cross_sectional_momentum=True,
            prices_override=prices_spike,
        )

        obs_base, _ = env_base.reset()
        obs_spike, _ = env_spike.reset()

        # Momentum block (last n_assets elements) must be identical
        np.testing.assert_array_equal(obs_base[-n_assets:], obs_spike[-n_assets:])

    # ------------------------------------------------------------------
    # (f) momentum_window > window_size raises AssertionError
    # ------------------------------------------------------------------

    def test_momentum_window_larger_than_window_size_raises(self) -> None:
        """Constructor must raise AssertionError when momentum_window > window_size."""
        data, symbols = _make_env_data(n_steps=200, n_assets=2)
        with pytest.raises(AssertionError, match="momentum_window"):
            TradingEnv(
                data=data, symbols=symbols, initial_balance=10000.0,
                trading_fee_pct=0.001, slippage_pct=0.0005,
                window_size=10, reward_scaling=1.0, max_position=1.0,
                cross_sectional_momentum=True,
                momentum_window=11,  # > window_size=10
            )

    # ------------------------------------------------------------------
    # State validity: _assert_state_valid covers extended obs
    # ------------------------------------------------------------------

    def test_assert_state_valid_covers_momentum_values(self) -> None:
        """_assert_state_valid must pass throughout a full episode with momentum on."""
        env = self._make_momentum_env(
            n_steps=200, n_assets=2, window_size=20, momentum_window=10,
            cross_sectional_momentum=True,
        )
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))

        rng = np.random.RandomState(42)
        done = False
        while not done:
            action = rng.uniform(-1, 1, size=env.n_assets).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite obs encountered: {obs}"
            done = terminated or truncated


class TestTurnoverPenalty:
    """Tests for the turnover_penalty_coef reward adjustment.

    All expected values are hand-computed.
    """

    def test_coef_zero_reward_identical_to_baseline(self) -> None:
        """turnover_penalty_coef=0.0 must produce the exact same reward as no param.

        Two envs with identical data and action; one has coef=0.0, the other
        does not pass the argument at all (defaults to 0.0).  Rewards must
        be identical float-for-float.
        """
        data, symbols = _make_env_data(n_steps=200, n_assets=2)
        kwargs = dict(
            data=data,
            symbols=symbols,
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=10,
            reward_scaling=1.0,
            max_position=1.0,
        )
        env_baseline = TradingEnv(**kwargs)
        env_coef0 = TradingEnv(**kwargs, turnover_penalty_coef=0.0)

        env_baseline.reset(seed=0)
        env_coef0.reset(seed=0)
        action = np.array([1.0, 1.0], dtype=np.float32)

        _, r_baseline, _, _, _ = env_baseline.step(action)
        _, r_coef0, _, _, _ = env_coef0.step(action)

        assert r_coef0 == r_baseline  # exact bit-for-bit equality

    def test_coef_nonzero_reduces_reward_by_penalty(self) -> None:
        """With coef>0, reward = log_return*scaling - coef*turnover.

        Hand-computed scenario:
          initial_balance=10000, action=[1,1] => weights=[0.5, 0.5]
          synthetic data seed: both envs reset(seed=0)
          step 0: data_idx=10, current_prices from _make_env_data with seed 42

        We run the *same* action on two identical envs differing only in coef.
        The difference in rewards must equal coef * turnover, with turnover
        read from the info dict and cross-checked against total_traded_value/old_pv.
        """
        coef = 0.01
        data, symbols = _make_env_data(n_steps=200, n_assets=2)
        common_kwargs = dict(
            data=data,
            symbols=symbols,
            initial_balance=10000.0,
            trading_fee_pct=0.001,
            slippage_pct=0.0005,
            window_size=10,
            reward_scaling=1.0,
            max_position=1.0,
        )
        env0 = TradingEnv(**common_kwargs, turnover_penalty_coef=0.0)
        envX = TradingEnv(**common_kwargs, turnover_penalty_coef=coef)

        env0.reset(seed=0)
        envX.reset(seed=0)
        action = np.array([1.0, 1.0], dtype=np.float32)

        _, r0, _, _, info0 = env0.step(action)
        _, rX, _, _, infoX = envX.step(action)

        # Turnover from info dict; both envs must agree since same action/prices
        turnover = info0["turnover"]
        assert infoX["turnover"] == pytest.approx(turnover)

        # Reward difference must equal coef * turnover exactly
        assert r0 - rX == pytest.approx(coef * turnover)

        # Cross-check: turnover = total_traded_value / initial_balance (all-cash step)
        expected_turnover = info0["total_traded_value"] / 10000.0
        assert turnover == pytest.approx(expected_turnover)

    def test_info_contains_turnover_key(self) -> None:
        """The info dict must always contain a 'turnover' key."""
        env = _make_env()
        env.reset()
        action = np.array([0.5, 0.5], dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert "turnover" in info
        assert np.isfinite(info["turnover"])

    def test_turnover_zero_for_hold_action(self) -> None:
        """A hold action (all -1 => weight 0.0 = no trades) must yield turnover=0.

        Action all-minus-one maps to weight 0.0 for every asset.  When the
        previous state is all-cash (fresh reset), target_shares=0 = current
        holdings, so no trades execute and total_traded_value must be 0.0.
        """
        env = _make_env()
        env.reset()
        # Action = all -1.0 → weight = 0.0 → no buys, no sells (holdings=0)
        action = np.full(env.n_assets, -1.0, dtype=np.float32)
        _, _, _, _, info = env.step(action)

        assert info["turnover"] == pytest.approx(0.0)
        assert info["total_traded_value"] == pytest.approx(0.0)
