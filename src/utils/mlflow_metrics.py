"""MLflow metrics logger for trading diagnostics.

Three-layer logging system:
1. Training diagnostics (losses, action distributions, replay buffer stats)
2. Trading performance (Sharpe, Sortino, Calmar, win rate, profit factor, etc.)
3. Behavioral analysis (per-asset breakdown, action heatmaps, regime analysis)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.logger import get_logger
from src.utils.metrics import (
    compute_calmar_ratio,
    compute_cumulative_profit_ratio,
    compute_max_drawdown,
    compute_max_drawdown_duration,
    compute_profit_factor,
    compute_reward_kurtosis,
    compute_reward_skewness,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_total_return_pct,
    compute_win_rate,
)

matplotlib.use("Agg")

logger = get_logger(__name__)


class TradingMetricsLogger:
    """Wraps MLflow with methods for all 3 diagnostic layers.

    Must be used inside an ``mlflow.start_run()`` context.

    Args:
        symbols: List of traded symbol names.
        initial_balance: Starting cash balance.
    """

    def __init__(self, symbols: list[str], initial_balance: float) -> None:
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.initial_balance = initial_balance

    # ── Layer 1: Training Diagnostics ──────────────────────────────────

    def log_training_step(self, step: int, info_dict: dict[str, float]) -> None:
        """Log per-step training diagnostics (losses, Q-values).

        Args:
            step: Current training step.
            info_dict: Dict with keys like critic_loss, actor_loss, etc.
        """
        for key, value in info_dict.items():
            if np.isfinite(value):
                mlflow.log_metric(f"train/{key}", value, step=step)

    def log_episode_end(self, episode_num: int, episode_info: dict[str, float]) -> None:
        """Log end-of-episode summary metrics.

        Args:
            episode_num: Episode counter.
            episode_info: Dict with episode_reward, length, pv, etc.
        """
        for key, value in episode_info.items():
            if np.isfinite(value):
                mlflow.log_metric(f"episode/{key}", value, step=episode_num)

    def log_action_distribution(
        self, step: int, actions_buffer: NDArray[np.float64]
    ) -> None:
        """Log action distribution statistics and save CSV artifact.

        Args:
            step: Current training step.
            actions_buffer: Array of shape (n_samples, n_assets).
        """
        if len(actions_buffer) == 0:
            return

        mlflow.log_metric("actions/mean", float(np.mean(actions_buffer)), step=step)
        mlflow.log_metric("actions/std", float(np.std(actions_buffer)), step=step)
        mlflow.log_metric("actions/min", float(np.min(actions_buffer)), step=step)
        mlflow.log_metric("actions/max", float(np.max(actions_buffer)), step=step)

        # Per-asset means
        for i, symbol in enumerate(self.symbols):
            col = actions_buffer[:, i] if actions_buffer.ndim > 1 else actions_buffer
            safe_name = symbol.replace("/", "_")
            mlflow.log_metric(f"actions/{safe_name}_mean", float(np.mean(col)), step=step)

        # Save CSV artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / f"action_distribution_step_{step}.csv"
            df = pd.DataFrame(actions_buffer, columns=self.symbols)
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="diagnostics")

    def log_replay_buffer_stats(
        self, step: int, rewards: NDArray[np.float64]
    ) -> None:
        """Log replay buffer reward statistics.

        Args:
            step: Current training step.
            rewards: Reward array from replay buffer.
        """
        if len(rewards) == 0:
            return

        mlflow.log_metric("buffer/reward_mean", float(np.mean(rewards)), step=step)
        mlflow.log_metric("buffer/reward_std", float(np.std(rewards)), step=step)
        mlflow.log_metric("buffer/reward_min", float(np.min(rewards)), step=step)
        mlflow.log_metric("buffer/reward_max", float(np.max(rewards)), step=step)
        mlflow.log_metric("buffer/reward_median", float(np.median(rewards)), step=step)

    # ── Layer 2: Trading Performance ───────────────────────────────────

    def log_trading_performance(self, eval_result: dict[str, Any]) -> dict[str, float]:
        """Compute and log all trading performance metrics.

        Args:
            eval_result: Dict from run_full_evaluation with keys:
                portfolio_values, rewards, trade_pnls, total_costs.

        Returns:
            Dict of all computed metrics.
        """
        pv = np.array(eval_result["portfolio_values"], dtype=np.float64)
        rewards = np.array(eval_result["rewards"], dtype=np.float64)
        trade_pnls = np.array(eval_result["trade_pnls"], dtype=np.float64)

        metrics = {
            "perf/sharpe_ratio": compute_sharpe_ratio(pv),
            "perf/sortino_ratio": compute_sortino_ratio(pv),
            "perf/calmar_ratio": compute_calmar_ratio(pv),
            "perf/max_drawdown": compute_max_drawdown(pv),
            "perf/max_drawdown_duration": float(compute_max_drawdown_duration(pv)),
            "perf/cpr": compute_cumulative_profit_ratio(pv),
            "perf/total_return_pct": compute_total_return_pct(pv),
            "perf/win_rate": compute_win_rate(trade_pnls),
            "perf/profit_factor": compute_profit_factor(trade_pnls),
            "perf/n_trades": float(len(trade_pnls)),
            "perf/total_cost": float(eval_result.get("total_costs", 0.0)),
            "perf/reward_skewness": compute_reward_skewness(rewards),
            "perf/reward_kurtosis": compute_reward_kurtosis(rewards),
        }

        for key, value in metrics.items():
            if np.isfinite(value):
                mlflow.log_metric(key, value)

        # Save equity curve CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            eq_path = Path(tmpdir) / "equity_curve.csv"
            pd.DataFrame({"step": range(len(pv)), "portfolio_value": pv}).to_csv(
                eq_path, index=False
            )
            mlflow.log_artifact(str(eq_path), artifact_path="evaluation")

            # Save trades log CSV if trades exist
            if "trades_log" in eval_result and len(eval_result["trades_log"]) > 0:
                trades_path = Path(tmpdir) / "trades_log.csv"
                pd.DataFrame(eval_result["trades_log"]).to_csv(
                    trades_path, index=False
                )
                mlflow.log_artifact(str(trades_path), artifact_path="evaluation")

        logger.info(
            "Logged trading performance: Sharpe=%.4f, Sortino=%.4f, MaxDD=%.4f, WinRate=%.4f",
            metrics["perf/sharpe_ratio"],
            metrics["perf/sortino_ratio"],
            metrics["perf/max_drawdown"],
            metrics["perf/win_rate"],
        )

        return metrics

    # ── Layer 3: Behavioral Analysis ───────────────────────────────────

    def log_per_asset_breakdown(self, eval_result: dict[str, Any]) -> None:
        """Log per-asset PnL, trade count, and win rate.

        Args:
            eval_result: Dict with per_asset_pnls and per_asset_trades.
        """
        per_asset_pnls = eval_result.get("per_asset_pnls", {})
        per_asset_trades = eval_result.get("per_asset_trades", {})

        for symbol in self.symbols:
            safe_name = symbol.replace("/", "_")
            pnls = np.array(per_asset_pnls.get(symbol, []), dtype=np.float64)

            mlflow.log_metric(f"asset/{safe_name}_total_pnl", float(np.sum(pnls)) if len(pnls) > 0 else 0.0)
            mlflow.log_metric(f"asset/{safe_name}_n_trades", float(per_asset_trades.get(symbol, 0)))
            mlflow.log_metric(f"asset/{safe_name}_win_rate", compute_win_rate(pnls))

    def log_action_analysis(self, eval_result: dict[str, Any]) -> None:
        """Log action heatmap as PNG artifact.

        Args:
            eval_result: Dict with actions array of shape (n_steps, n_assets).
        """
        actions = np.array(eval_result.get("actions", []), dtype=np.float64)
        if len(actions) == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        # Subsample if too many steps for readability
        max_display = 500
        if len(actions) > max_display:
            step_size = len(actions) // max_display
            display_actions = actions[::step_size]
        else:
            display_actions = actions

        im = ax.imshow(display_actions.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(self.n_assets))
        ax.set_yticklabels(self.symbols, fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Asset")
        ax.set_title("Action Heatmap (target weights)")
        fig.colorbar(im, ax=ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "action_heatmap.png"
            fig.savefig(png_path, dpi=100, bbox_inches="tight")
            mlflow.log_artifact(str(png_path), artifact_path="behavioral")
        plt.close(fig)

    def log_reward_distribution(self, eval_result: dict[str, Any]) -> None:
        """Log reward and PnL distribution histograms.

        Args:
            eval_result: Dict with rewards and trade_pnls arrays.
        """
        rewards = np.array(eval_result.get("rewards", []), dtype=np.float64)
        trade_pnls = np.array(eval_result.get("trade_pnls", []), dtype=np.float64)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if len(rewards) > 0:
            axes[0].hist(rewards, bins=50, edgecolor="black", alpha=0.7)
            axes[0].set_title("Reward Distribution")
            axes[0].set_xlabel("Reward")
            axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

        if len(trade_pnls) > 0:
            axes[1].hist(trade_pnls, bins=50, edgecolor="black", alpha=0.7, color="green")
            axes[1].set_title("Trade PnL Distribution")
            axes[1].set_xlabel("PnL")
            axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)

        fig.tight_layout()

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "reward_pnl_distributions.png"
            fig.savefig(png_path, dpi=100, bbox_inches="tight")
            mlflow.log_artifact(str(png_path), artifact_path="behavioral")
        plt.close(fig)

    def log_trade_timing_analysis(
        self,
        eval_result: dict[str, Any],
        price_data: dict[str, NDArray[np.float64]],
    ) -> None:
        """Analyze whether agent buys below SMA and sells above SMA.

        Args:
            eval_result: Dict with trades_log containing step and symbol info.
            price_data: Dict mapping symbol to close price array.
        """
        trades_log = eval_result.get("trades_log", [])
        if len(trades_log) == 0:
            return

        sma_window = 20
        buys_below_sma = 0
        buys_total = 0
        sells_above_sma = 0
        sells_total = 0

        for trade in trades_log:
            symbol = trade.get("symbol", "")
            step = trade.get("step", 0)
            side = trade.get("side", "")

            if symbol not in price_data:
                continue

            prices = price_data[symbol]
            if step < sma_window or step >= len(prices):
                continue

            sma = float(np.mean(prices[step - sma_window : step]))
            current_price = float(prices[step])

            if side == "buy":
                buys_total += 1
                if current_price < sma:
                    buys_below_sma += 1
            elif side == "sell":
                sells_total += 1
                if current_price > sma:
                    sells_above_sma += 1

        buy_below_pct = buys_below_sma / buys_total * 100 if buys_total > 0 else 0.0
        sell_above_pct = sells_above_sma / sells_total * 100 if sells_total > 0 else 0.0

        mlflow.log_metric("timing/buy_below_sma_pct", buy_below_pct)
        mlflow.log_metric("timing/sell_above_sma_pct", sell_above_pct)
        mlflow.log_metric("timing/total_buys", float(buys_total))
        mlflow.log_metric("timing/total_sells", float(sells_total))

        logger.info(
            "Trade timing: %.1f%% buys below SMA20, %.1f%% sells above SMA20",
            buy_below_pct,
            sell_above_pct,
        )

    def log_ti_correlation(self, eval_result: dict[str, Any]) -> None:
        """Log correlation between TI features and actions.

        Args:
            eval_result: Dict with observations and actions arrays.
        """
        observations = eval_result.get("observations", [])
        actions = eval_result.get("actions", [])

        if len(observations) == 0 or len(actions) == 0:
            return

        obs_array = np.array(observations, dtype=np.float64)
        act_array = np.array(actions, dtype=np.float64)

        # Use last n_assets columns before cash_ratio as portfolio weights
        # and remaining earlier columns as TI features
        n_obs_cols = obs_array.shape[1]
        n_portfolio_state = self.n_assets + 1  # weights + cash_ratio
        n_feature_cols = n_obs_cols - n_portfolio_state

        if n_feature_cols <= 0:
            return

        feature_cols = obs_array[:, :n_feature_cols]
        # Subsample features for correlation: take every 10th column
        max_features = min(n_feature_cols, 24)
        feature_indices = np.linspace(0, n_feature_cols - 1, max_features, dtype=int)

        correlations = []
        for fi in feature_indices:
            for ai in range(self.n_assets):
                corr = np.corrcoef(feature_cols[:, fi], act_array[:, ai])[0, 1]
                if np.isfinite(corr):
                    correlations.append(
                        {"feature_idx": int(fi), "asset": self.symbols[ai], "correlation": corr}
                    )

        if not correlations:
            return

        # Log top correlations
        sorted_corrs = sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)
        for i, entry in enumerate(sorted_corrs[:10]):
            safe_asset = entry["asset"].replace("/", "_")
            mlflow.log_metric(
                f"ti_corr/f{entry['feature_idx']}_{safe_asset}",
                entry["correlation"],
            )

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        top_n = min(20, len(sorted_corrs))
        labels = [f"f{c['feature_idx']}-{c['asset']}" for c in sorted_corrs[:top_n]]
        values = [c["correlation"] for c in sorted_corrs[:top_n]]
        colors = ["green" if v > 0 else "red" for v in values]
        ax.barh(range(top_n), values, color=colors, alpha=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Pearson Correlation")
        ax.set_title("Top TI-Action Correlations")
        ax.axvline(x=0, color="black", linewidth=0.5)
        fig.tight_layout()

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = Path(tmpdir) / "ti_action_correlations.png"
            fig.savefig(png_path, dpi=100, bbox_inches="tight")
            mlflow.log_artifact(str(png_path), artifact_path="behavioral")
        plt.close(fig)

    def log_regime_analysis(
        self,
        eval_result: dict[str, Any],
        price_data: dict[str, NDArray[np.float64]],
    ) -> None:
        """Analyze performance across market regimes (uptrend/downtrend/sideways).

        Uses SMA20 slope to classify regime at each step.

        Args:
            eval_result: Dict with portfolio_values and rewards.
            price_data: Dict mapping symbol to close price array.
                Uses the first symbol as the regime reference.
        """
        pv = np.array(eval_result.get("portfolio_values", []), dtype=np.float64)
        if len(pv) < 2:
            return

        # Use first symbol's prices for regime classification
        first_symbol = self.symbols[0]
        if first_symbol not in price_data:
            return

        prices = price_data[first_symbol]
        sma_window = 20
        slope_window = 5
        min_steps = sma_window + slope_window

        if len(prices) < min_steps:
            return

        # Classify each step into regime
        n_steps = min(len(pv) - 1, len(prices) - sma_window)
        regimes = np.full(n_steps, "sideways", dtype=object)

        for i in range(n_steps):
            price_idx = i + sma_window
            if price_idx < sma_window + slope_window:
                continue
            sma_now = float(np.mean(prices[price_idx - sma_window : price_idx]))
            sma_prev = float(np.mean(prices[price_idx - sma_window - slope_window : price_idx - slope_window]))

            if sma_prev == 0:
                continue

            slope_pct = (sma_now - sma_prev) / sma_prev * 100
            if slope_pct > 0.5:
                regimes[i] = "uptrend"
            elif slope_pct < -0.5:
                regimes[i] = "downtrend"

        # Compute per-regime Sharpe
        for regime in ["uptrend", "downtrend", "sideways"]:
            mask = regimes == regime
            count = int(np.sum(mask))
            if count < 2:
                mlflow.log_metric(f"regime/{regime}_sharpe", 0.0)
                mlflow.log_metric(f"regime/{regime}_n_steps", float(count))
                continue

            # Get PV subset for this regime (use indices)
            indices = np.where(mask)[0]
            regime_pv = pv[indices]
            regime_sharpe = compute_sharpe_ratio(regime_pv)

            mlflow.log_metric(f"regime/{regime}_sharpe", regime_sharpe)
            mlflow.log_metric(f"regime/{regime}_n_steps", float(count))

        logger.info(
            "Regime analysis: uptrend=%d, downtrend=%d, sideways=%d steps",
            int(np.sum(regimes == "uptrend")),
            int(np.sum(regimes == "downtrend")),
            int(np.sum(regimes == "sideways")),
        )
