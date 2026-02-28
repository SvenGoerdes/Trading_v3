"""MLflow diagnostics callback for SB3 TD3 training.

Logs training losses, action distributions, and replay buffer stats
to MLflow at configurable intervals during training.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.logger import get_logger
from src.utils.mlflow_metrics import TradingMetricsLogger

logger = get_logger(__name__)

# Keys SB3 stores in self.model.logger.name_to_value after learning_starts
SB3_CRITIC_LOSS_KEY = "train/critic_loss"
SB3_ACTOR_LOSS_KEY = "train/actor_loss"

DIAGNOSTICS_STEP_INTERVAL = 1_000
BUFFER_LOG_INTERVAL = 10_000


class MLflowDiagnosticsCallback(BaseCallback):
    """SB3 callback that logs training diagnostics to MLflow.

    Logs:
    - Every ``diagnostics_step_interval`` steps: critic/actor losses, Q-values.
    - Every ``buffer_log_interval`` steps: action distribution + replay buffer stats.
    - On episode end: episode reward, length, portfolio value.

    Args:
        metrics_logger: TradingMetricsLogger instance (used inside mlflow.start_run).
        diagnostics_step_interval: Steps between loss logging.
        buffer_log_interval: Steps between buffer/action logging.
    """

    def __init__(
        self,
        metrics_logger: TradingMetricsLogger,
        diagnostics_step_interval: int = DIAGNOSTICS_STEP_INTERVAL,
        buffer_log_interval: int = BUFFER_LOG_INTERVAL,
    ) -> None:
        super().__init__(verbose=0)
        self.metrics_logger = metrics_logger
        self.diagnostics_step_interval = diagnostics_step_interval
        self.buffer_log_interval = buffer_log_interval
        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._episode_count: int = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # Accumulate episode stats
        reward = self.locals.get("rewards")
        if reward is not None:
            self._episode_reward += float(np.sum(reward))
            self._episode_length += int(np.size(reward))

        # Detect episode end
        dones = self.locals.get("dones")
        if dones is not None and np.any(dones):
            self._on_episode_end()

        # Log training losses at interval
        if step % self.diagnostics_step_interval == 0:
            self._log_training_losses(step)

        # Log action distribution + buffer stats at interval
        if step % self.buffer_log_interval == 0:
            self._log_buffer_diagnostics(step)

        return True

    def _on_episode_end(self) -> None:
        """Log episode-level metrics."""
        self._episode_count += 1

        info = self.locals.get("infos")
        portfolio_value = 0.0
        if info is not None and len(info) > 0:
            portfolio_value = info[0].get("portfolio_value", 0.0)

        episode_info = {
            "reward": self._episode_reward,
            "length": float(self._episode_length),
            "portfolio_value": portfolio_value,
        }

        if portfolio_value > 0 and self.metrics_logger.initial_balance > 0:
            return_pct = (
                (portfolio_value - self.metrics_logger.initial_balance)
                / self.metrics_logger.initial_balance
                * 100
            )
            episode_info["return_pct"] = return_pct

        self.metrics_logger.log_episode_end(self._episode_count, episode_info)

        # Reset accumulators
        self._episode_reward = 0.0
        self._episode_length = 0

    def _log_training_losses(self, step: int) -> None:
        """Extract and log SB3 training losses."""
        if not hasattr(self.model, "logger"):
            return

        name_to_value = getattr(self.model.logger, "name_to_value", {})
        if not name_to_value:
            return

        info_dict = {}
        if SB3_CRITIC_LOSS_KEY in name_to_value:
            info_dict["critic_loss"] = float(name_to_value[SB3_CRITIC_LOSS_KEY])
        if SB3_ACTOR_LOSS_KEY in name_to_value:
            info_dict["actor_loss"] = float(name_to_value[SB3_ACTOR_LOSS_KEY])

        if info_dict:
            self.metrics_logger.log_training_step(step, info_dict)

    def _log_buffer_diagnostics(self, step: int) -> None:
        """Log action distribution and replay buffer reward stats."""
        if not hasattr(self.model, "replay_buffer"):
            return

        replay_buffer = self.model.replay_buffer
        buffer_pos = replay_buffer.pos
        buffer_full = replay_buffer.full
        buffer_size = replay_buffer.buffer_size

        if buffer_pos == 0 and not buffer_full:
            return

        # Determine valid range
        valid_size = buffer_size if buffer_full else buffer_pos

        # Extract actions and rewards from buffer
        actions = replay_buffer.actions[:valid_size].reshape(valid_size, -1)
        rewards = replay_buffer.rewards[:valid_size].flatten()

        self.metrics_logger.log_action_distribution(step, actions)
        self.metrics_logger.log_replay_buffer_stats(step, rewards)

        logger.debug(
            "Buffer diagnostics at step %d: %d samples, reward_mean=%.4f",
            step,
            valid_size,
            float(np.mean(rewards)),
        )
