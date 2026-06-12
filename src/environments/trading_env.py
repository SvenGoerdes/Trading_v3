"""Custom Gymnasium trading environment for the TD3 agent.

Observation space: windowed technical indicators + portfolio state.
Action space: continuous allocation weights per asset.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pandas import DataFrame

from src.utils.logger import get_logger
from src.utils.portfolio import (
    compute_portfolio_value,
    compute_target_holdings,
    execute_rebalance,
)

logger = get_logger(__name__)

NORM_COLUMN_SUFFIX = "_norm"


class TradingEnv(gym.Env):
    """Multi-asset portfolio trading environment.

    Observation: flat vector of [window_features | portfolio_weights | cash_ratio]
    or, when cross_sectional_momentum is enabled:
    [window_features | portfolio_weights | cash_ratio | momentum(n_assets)].

    Action: Box(-1, 1) mapped to allocation weights via (action + 1) / 2.

    Args:
        data: Dict mapping symbol to DataFrame with datetime index,
              ``close`` column, and ``*_norm`` feature columns.
        symbols: List of symbol names (must match data keys).
        initial_balance: Starting cash balance.
        trading_fee_pct: Trading fee as a fraction.
        slippage_pct: Slippage as a fraction.
        window_size: Number of lookback steps for observation.
        reward_scaling: Multiplier for the log-return reward.
        max_position: Maximum weight per asset (0 to 1).
        rebalance_threshold: Minimum trade size as a fraction of portfolio
            value.  Trades below this threshold are skipped (no-trade band).
            Default 0.0 disables the band.
        turnover_penalty_coef: Coefficient for the turnover penalty term in the
            reward.  Reward becomes ``log_return * reward_scaling -
            turnover_penalty_coef * turnover``, where ``turnover =
            total_traded_value / old_portfolio_value``.  Default 0.0
            reproduces the original reward bit-for-bit.
        cross_sectional_momentum: When True, append ``n_assets`` cross-sectional
            momentum values after ``cash_ratio`` in the observation.  Default False
            reproduces prior observation layout bit-for-bit.
        momentum_window: Lookback bars for the momentum return window.  Must be
            <= window_size.  Default 12.
        allocation_mode: Passed through to ``compute_target_holdings``.
            ``"renormalize"`` (default) reproduces existing behaviour bit-for-bit.
            ``"scaled"`` allows the agent to hold cash.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: dict[str, DataFrame],
        symbols: list[str],
        initial_balance: float,
        trading_fee_pct: float,
        slippage_pct: float,
        window_size: int,
        reward_scaling: float,
        max_position: float,
        rebalance_threshold: float = 0.0,
        turnover_penalty_coef: float = 0.0,
        cross_sectional_momentum: bool = False,
        momentum_window: int = 12,
        allocation_mode: str = "renormalize",
    ) -> None:
        super().__init__()
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.initial_balance = initial_balance
        self.trading_fee_pct = trading_fee_pct
        self.slippage_pct = slippage_pct
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.max_position = max_position
        self.rebalance_threshold = rebalance_threshold
        self.turnover_penalty_coef = turnover_penalty_coef
        self.cross_sectional_momentum = cross_sectional_momentum
        self.momentum_window = momentum_window
        self.allocation_mode = allocation_mode

        self._build_data_arrays(data)

        if self.cross_sectional_momentum:
            assert self.momentum_window <= self.window_size, (
                f"momentum_window ({self.momentum_window}) must be <= "
                f"window_size ({self.window_size})"
            )

        self.max_steps = self.n_steps - self.window_size

        # Action: target weights per asset in [-1, 1], mapped to [0, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation: flattened window features + portfolio weights + cash ratio
        # + optional n_assets cross-sectional momentum values
        obs_size = (
            self.window_size * self.n_assets * self.n_features
            + self.n_assets  # portfolio weights
            + 1  # cash ratio
        )
        if self.cross_sectional_momentum:
            obs_size += self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # State (set in reset)
        self.balance = 0.0
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.current_step = 0

    def _build_data_arrays(self, data: dict[str, DataFrame]) -> None:
        """Inner-join all symbols on datetime index, extract arrays."""
        # Get common datetime index
        common_index = None
        for symbol in self.symbols:
            df = data[symbol]
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        assert common_index is not None and len(common_index) > 0, "No common timestamps"

        # Identify normalized feature columns from first symbol
        sample_df = data[self.symbols[0]]
        norm_cols = sorted([c for c in sample_df.columns if c.endswith(NORM_COLUMN_SUFFIX)])
        self.n_features = len(norm_cols)
        assert self.n_features > 0, "No normalized feature columns found"

        self.n_steps = len(common_index)

        # Build 3D feature array: (n_steps, n_assets, n_features)
        self.features = np.zeros(
            (self.n_steps, self.n_assets, self.n_features), dtype=np.float64
        )
        # Build 2D price array: (n_steps, n_assets)
        self.prices = np.zeros((self.n_steps, self.n_assets), dtype=np.float64)

        for i, symbol in enumerate(self.symbols):
            df = data[symbol].loc[common_index]
            self.features[:, i, :] = df[norm_cols].values
            self.prices[:, i] = df["close"].values

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[NDArray[np.float32], dict]:
        """Reset environment to initial state.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed, options=options)

        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)
        self.current_step = 0

        obs = self._get_observation()
        self._assert_state_valid(obs)

        return obs, {}

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        """Execute one step: rebalance portfolio based on action.

        Args:
            action: Target weights in [-1, 1] per asset.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Map action from [-1, 1] to [0, 1] weights
        target_weights = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0

        data_idx = self.window_size + self.current_step
        current_prices = self.prices[data_idx]

        old_portfolio_value = compute_portfolio_value(
            self.balance, self.holdings, current_prices
        )

        target_shares = compute_target_holdings(
            target_weights,
            old_portfolio_value,
            current_prices,
            self.max_position,
            allocation_mode=self.allocation_mode,
        )

        self.balance, self.holdings, total_cost, total_traded_value = execute_rebalance(
            self.balance,
            self.holdings,
            target_shares,
            current_prices,
            self.trading_fee_pct,
            self.slippage_pct,
            rebalance_threshold=self.rebalance_threshold,
        )

        self.current_step += 1
        terminated = False
        truncated = False

        # Check termination conditions
        if self.current_step >= self.max_steps:
            truncated = True

        new_data_idx = self.window_size + self.current_step
        if not truncated and new_data_idx < self.n_steps:
            new_prices = self.prices[new_data_idx]
        else:
            new_prices = current_prices

        new_portfolio_value = compute_portfolio_value(
            self.balance, self.holdings, new_prices
        )

        if new_portfolio_value <= 0:
            terminated = True

        # Turnover: total executed trade value / pre-trade portfolio value
        if old_portfolio_value > 0:
            turnover = total_traded_value / old_portfolio_value
        else:
            turnover = 0.0

        # Reward: log return scaled, minus turnover penalty
        if old_portfolio_value > 0 and new_portfolio_value > 0:
            reward = float(
                np.log(new_portfolio_value / old_portfolio_value) * self.reward_scaling
                - self.turnover_penalty_coef * turnover
            )
        else:
            reward = 0.0

        obs = self._get_observation()
        self._assert_state_valid(obs)

        cash_ratio = self.balance / new_portfolio_value if new_portfolio_value > 0 else 0.0

        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self.balance,
            "total_cost": total_cost,
            "total_traded_value": total_traded_value,
            "turnover": turnover,
            "holdings": self.holdings.copy(),
            "action": target_weights.copy(),
            "cash_ratio": cash_ratio,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> NDArray[np.float32]:
        """Build the flat observation vector.

        Layout (always): [window_features | portfolio_weights | cash_ratio]
        Layout (cross_sectional_momentum=True): [...above... | momentum(n_assets)]
        """
        data_idx = self.window_size + self.current_step
        # Window features: (window_size, n_assets, n_features) -> flat
        window_start = data_idx - self.window_size
        window_features = self.features[window_start:data_idx].flatten()

        # Portfolio state
        current_prices = self.prices[min(data_idx, self.n_steps - 1)]
        portfolio_value = compute_portfolio_value(
            self.balance, self.holdings, current_prices
        )

        if portfolio_value > 0:
            asset_values = self.holdings * current_prices
            portfolio_weights = asset_values / portfolio_value
            cash_ratio = self.balance / portfolio_value
        else:
            portfolio_weights = np.zeros(self.n_assets)
            cash_ratio = 0.0

        parts: list[NDArray] = [window_features, portfolio_weights, [cash_ratio]]

        if self.cross_sectional_momentum:
            parts.append(self._compute_cross_sectional_momentum(data_idx))

        obs = np.concatenate(parts).astype(np.float32)

        return obs

    def _compute_cross_sectional_momentum(self, data_idx: int) -> NDArray[np.float64]:
        """Compute cross-sectional momentum standardised across assets.

        Uses only prices up to and including ``data_idx`` (no look-ahead).

        Args:
            data_idx: Current bar index (= window_size + current_step).

        Returns:
            Array of shape (n_assets,) with z-scored momentum values.
        """
        idx_now = min(data_idx, self.n_steps - 1)
        idx_past = max(idx_now - self.momentum_window, 0)
        raw_ret = self.prices[idx_now] / self.prices[idx_past] - 1.0
        mom = (raw_ret - raw_ret.mean()) / (raw_ret.std() + 1e-8)
        return mom

    def _assert_state_valid(self, obs: NDArray[np.float32]) -> None:
        """Assert environment state invariants after every transition.

        Args:
            obs: Pre-computed observation to check for finiteness.
        """
        assert self.balance >= 0, f"Negative balance: {self.balance}"
        assert np.all(self.holdings >= 0), f"Negative holdings: {self.holdings}"
        assert np.all(np.isfinite(obs)), f"Non-finite values in observation: {obs}"
