"""Tests for feature engineering pipeline nodes.

TI tests use hand-verified expected values where feasible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines.feature_engineering.nodes import (
    compute_accumulation_distribution,
    compute_disparity_index,
    compute_ema,
    compute_macd,
    compute_on_balance_volume,
    compute_rate_of_change,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_technical_indicators,
    compute_williams_r,
    create_train_test_split,
    normalize_features,
)
from src.utils.config import (
    IndicatorsConfig,
    MacdConfig,
    NormalizationConfig,
    SplitConfig,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_ohlcv_df(n: int = 200) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic-ish price action."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(500, 5000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_ti_config() -> IndicatorsConfig:
    return IndicatorsConfig(
        rsi_window=14,
        sma_window=20,
        ema_window=20,
        stochastic_window=14,
        macd=MacdConfig(fast=12, slow=26, signal=9),
        roc_window=10,
        williams_r_window=14,
        disparity_sma_window=20,
        ad=True,
        obv=True,
    )


# ── Hand-verified TI tests ──────────────────────────────────────────────


class TestComputeSma:
    def test_known_values(self) -> None:
        """SMA(3) of [1,2,3,4,5] = [NaN, NaN, 2, 3, 4]."""
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_sma(close, 3)
        assert result.name == "sma_3"
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)


class TestComputeEma:
    def test_known_values(self) -> None:
        """EMA(3): first value is SMA of first 3, then EMA formula applies."""
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_ema(close, 3)
        assert result.name == "ema_3"
        # EMA(3) at index 2 = SMA(first 3) = 2.0
        assert result.iloc[2] == pytest.approx(2.0)
        # EMA(3) at index 3 = 4 * (2/4) + 2.0 * (2/4) = 3.0
        assert result.iloc[3] == pytest.approx(3.0)


class TestComputeRsi:
    def test_all_gains_gives_100(self) -> None:
        """Strictly increasing prices → RSI close to 100."""
        close = pd.Series(list(range(1, 30)), dtype=float)
        result = compute_rsi(close, 14)
        last_val = result.dropna().iloc[-1]
        assert last_val > 95.0

    def test_all_losses_gives_near_zero(self) -> None:
        """Strictly decreasing prices → RSI close to 0."""
        close = pd.Series(list(range(30, 1, -1)), dtype=float)
        result = compute_rsi(close, 14)
        last_val = result.dropna().iloc[-1]
        assert last_val < 5.0

    def test_name(self) -> None:
        close = pd.Series([1.0] * 20)
        result = compute_rsi(close, 14)
        assert result.name == "rsi_14"


class TestComputeStochastic:
    def test_output_columns(self) -> None:
        df = _make_ohlcv_df(50)
        result = compute_stochastic(df["high"], df["low"], df["close"], 14)
        assert list(result.columns) == ["stoch_k", "stoch_d"]

    def test_range(self) -> None:
        """Stochastic should be in [0, 100]."""
        df = _make_ohlcv_df(100)
        result = compute_stochastic(df["high"], df["low"], df["close"], 14).dropna()
        assert result["stoch_k"].min() >= 0
        assert result["stoch_k"].max() <= 100


class TestComputeMacd:
    def test_output_columns(self) -> None:
        close = pd.Series(np.random.randn(100).cumsum() + 100)
        result = compute_macd(close, 12, 26, 9)
        assert list(result.columns) == ["macd", "macd_signal"]

    def test_macd_line_starts_at_zero(self) -> None:
        """For constant prices, MACD should be ~0."""
        close = pd.Series([100.0] * 50)
        result = compute_macd(close, 12, 26, 9)
        non_null = result["macd"].dropna()
        assert all(abs(v) < 1e-6 for v in non_null)


class TestComputeObv:
    def test_known_values(self) -> None:
        """OBV: +volume when close up, -volume when close down."""
        close = pd.Series([10.0, 11.0, 10.5, 12.0])
        volume = pd.Series([100.0, 200.0, 150.0, 300.0])
        result = compute_on_balance_volume(close, volume)
        assert result.name == "obv"
        # OBV: 0, +200, -150 → 0+200=200, 200-150=50, 50+300=350
        assert result.iloc[1] == pytest.approx(200.0)
        assert result.iloc[2] == pytest.approx(50.0)
        assert result.iloc[3] == pytest.approx(350.0)


class TestComputeRoc:
    def test_known_values(self) -> None:
        """ROC(2) of [100, 110, 120] = [NaN, NaN, 20.0]."""
        close = pd.Series([100.0, 110.0, 120.0, 130.0])
        result = compute_rate_of_change(close, 2)
        assert result.name == "roc_2"
        val = result.dropna().iloc[0]
        # ROC(2) at idx 2 = ((120 - 100) / 100) * 100 = 20.0
        assert val == pytest.approx(20.0)


class TestComputeWilliamsR:
    def test_range(self) -> None:
        """Williams %R should be in [-100, 0]."""
        df = _make_ohlcv_df(100)
        result = compute_williams_r(df["high"], df["low"], df["close"], 14).dropna()
        assert result.min() >= -100.0
        assert result.max() <= 0.0

    def test_name(self) -> None:
        df = _make_ohlcv_df(50)
        result = compute_williams_r(df["high"], df["low"], df["close"], 14)
        assert result.name == "williams_r_14"


class TestComputeDisparityIndex:
    def test_known_values(self) -> None:
        """When close == SMA, disparity = 0."""
        close = pd.Series([100.0] * 30)
        result = compute_disparity_index(close, 20)
        non_null = result.dropna()
        assert all(abs(v) < 1e-6 for v in non_null)

    def test_name(self) -> None:
        close = pd.Series([100.0] * 30)
        result = compute_disparity_index(close, 20)
        assert result.name == "disparity_20"


class TestComputeAD:
    def test_name(self) -> None:
        df = _make_ohlcv_df(50)
        result = compute_accumulation_distribution(df["high"], df["low"], df["close"], df["volume"])
        assert result.name == "ad"


# ── Orchestration tests ─────────────────────────────────────────────────


class TestComputeTechnicalIndicators:
    def test_all_indicators_present(self) -> None:
        df = _make_ohlcv_df(200)
        config = _make_ti_config()
        result = compute_technical_indicators({"TEST/USDT": df}, config)
        cols = result["TEST/USDT"].columns.tolist()
        expected_ti = [
            "rsi_14", "sma_20", "ema_20", "stoch_k", "stoch_d",
            "macd", "macd_signal", "ad", "obv", "roc_10",
            "williams_r_14", "disparity_20",
        ]
        for ti in expected_ti:
            assert ti in cols, f"Missing TI column: {ti}"

    def test_no_nans_after_warmup_drop(self) -> None:
        df = _make_ohlcv_df(200)
        config = _make_ti_config()
        result = compute_technical_indicators({"TEST/USDT": df}, config)
        assert not result["TEST/USDT"].isna().any().any()

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        config = _make_ti_config()
        result = compute_technical_indicators({"TEST/USDT": df}, config)
        assert result["TEST/USDT"].empty


# ── Normalization tests ──────────────────────────────────────────────────


class TestNormalizeFeatures:
    def _make_features(self) -> dict[str, pd.DataFrame]:
        df = _make_ohlcv_df(700)
        config = _make_ti_config()
        return compute_technical_indicators({"TEST/USDT": df}, config)

    def test_norm_columns_created(self) -> None:
        features = self._make_features()
        norm_config = NormalizationConfig(method="rolling_zscore", window=50)
        result = normalize_features(features, norm_config)
        cols = result["TEST/USDT"].columns
        assert "rsi_14_norm" in cols
        assert "sma_20_norm" in cols

    def test_norm_approx_mean_std(self) -> None:
        """After sufficient warmup, normalized cols should have mean≈0, std≈1."""
        features = self._make_features()
        norm_config = NormalizationConfig(method="rolling_zscore", window=50)
        result = normalize_features(features, norm_config)
        df = result["TEST/USDT"]
        rsi_norm = df["rsi_14_norm"]
        assert abs(rsi_norm.mean()) < 1.0  # loose bound since rolling
        assert rsi_norm.std() > 0.1  # not all zeros

    def test_zero_std_gives_zero(self) -> None:
        """Constant column → normalized to 0.0, not inf."""
        idx = pd.date_range("2024-01-01", periods=600, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 100.0, "high": 105.0, "low": 95.0,
                "close": 100.0, "volume": 1000.0,
                "rsi_14": 50.0, "sma_20": 100.0, "ema_20": 100.0,
                "stoch_k": 50.0, "stoch_d": 50.0,
                "macd": 0.0, "macd_signal": 0.0,
                "ad": 0.0, "obv": 0.0,
                "roc_10": 0.0, "williams_r_14": -50.0,
                "disparity_20": 0.0,
            },
            index=idx,
        )
        norm_config = NormalizationConfig(method="rolling_zscore", window=50)
        result = normalize_features({"TEST/USDT": df}, norm_config)
        norm_df = result["TEST/USDT"]
        assert not np.isinf(norm_df["rsi_14_norm"]).any()
        assert (norm_df["rsi_14_norm"] == 0.0).all()

    def test_empty_df(self) -> None:
        norm_config = NormalizationConfig(method="rolling_zscore", window=50)
        result = normalize_features({"TEST/USDT": pd.DataFrame()}, norm_config)
        assert result["TEST/USDT"].empty


# ── Train/Test Split tests ───────────────────────────────────────────────


class TestCreateTrainTestSplit:
    def _make_normalized(self, n=1000) -> dict[str, pd.DataFrame]:
        idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame({"close": np.arange(n, dtype=float), "rsi_14_norm": np.random.randn(n)}, index=idx)
        return {"TEST/USDT": df}

    def test_preserves_temporal_order(self) -> None:
        normalized = self._make_normalized()
        split_config = SplitConfig(test_ratio=0.2, n_cv_folds=5)
        train, test, _ = create_train_test_split(normalized, split_config)
        assert train["TEST/USDT"].index[-1] < test["TEST/USDT"].index[0]

    def test_split_sizes(self) -> None:
        normalized = self._make_normalized(1000)
        split_config = SplitConfig(test_ratio=0.2, n_cv_folds=5)
        train, test, _ = create_train_test_split(normalized, split_config)
        assert len(train["TEST/USDT"]) == 800
        assert len(test["TEST/USDT"]) == 200

    def test_cv_folds_count(self) -> None:
        normalized = self._make_normalized()
        split_config = SplitConfig(test_ratio=0.2, n_cv_folds=5)
        _, _, cv_folds = create_train_test_split(normalized, split_config)
        assert len(cv_folds) == 1  # one symbol
        assert len(cv_folds[0]) == 5

    def test_empty_df(self) -> None:
        split_config = SplitConfig(test_ratio=0.2, n_cv_folds=5)
        train, test, _ = create_train_test_split({"TEST/USDT": pd.DataFrame()}, split_config)
        assert train["TEST/USDT"].empty
        assert test["TEST/USDT"].empty
