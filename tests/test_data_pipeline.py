"""Tests for data engineering pipeline nodes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipelines.data_engineering.nodes import (
    clean_ohlcv,
    clean_single_symbol,
    detect_candle_gaps,
    fetch_ohlcv,
    timeframe_to_milliseconds,
)
from src.utils.config import AppConfig, DataConfig, DataPaths, MissingCandlePolicy

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_ohlcv_df(
    start: str = "2024-01-01",
    periods: int = 100,
    freq: str = "5min",
    drop_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with a UTC datetime index."""
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000.0,
        },
        index=idx,
    )
    if drop_indices:
        df = df.drop(df.index[drop_indices])
    return df


def _make_config(tmp_path, ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20) -> AppConfig:
    """Build a minimal AppConfig for cleaning tests."""
    paths = DataPaths(
        raw=str(tmp_path / "raw"),
        intermediate=str(tmp_path / "intermediate"),
        features=str(tmp_path / "features"),
        normalized=str(tmp_path / "normalized"),
        splits=str(tmp_path / "splits"),
    )
    return AppConfig(
        symbols=["TEST/USDT"],
        timeframe="5m",
        initial_balance=10000.0,
        trading_fee_pct=0.001,
        slippage_pct=0.0005,
        data=DataConfig(
            lookback_days=120,
            missing_candle_policy=MissingCandlePolicy(
                ffill_limit=ffill_limit,
                stale_flag_limit=stale_flag_limit,
                segment_split_beyond=segment_split_beyond,
            ),
            paths=paths,
        ),
        indicators=None,  # type: ignore[arg-type]
        normalization=None,  # type: ignore[arg-type]
        split=None,  # type: ignore[arg-type]
    )


# ── timeframe_to_milliseconds ───────────────────────────────────────────


class TestTimeframeToMilliseconds:
    def test_5m(self) -> None:
        assert timeframe_to_milliseconds("5m") == 300_000

    def test_1h(self) -> None:
        assert timeframe_to_milliseconds("1h") == 3_600_000

    def test_1d(self) -> None:
        assert timeframe_to_milliseconds("1d") == 86_400_000

    def test_15m(self) -> None:
        assert timeframe_to_milliseconds("15m") == 900_000

    def test_invalid_unit(self) -> None:
        with pytest.raises(ValueError, match="Unknown timeframe unit"):
            timeframe_to_milliseconds("5x")


# ── fetch_ohlcv ─────────────────────────────────────────────────────────


class TestFetchOhlcv:
    def test_single_batch(self, tmp_path: Path) -> None:
        """Mock exchange returns fewer than limit candles → single batch."""
        candles = [
            [1704067200000, 100, 105, 95, 102, 1000],
            [1704067500000, 102, 106, 96, 104, 1100],
        ]
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = candles

        paths = DataPaths(
            raw=str(tmp_path / "raw"),
            intermediate="",
            features="",
            normalized="",
            splits="",
        )
        result = fetch_ohlcv(
            symbols=["BTC/USDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-01-02",
            paths=paths,
            exchange=exchange,
        )
        assert "BTC/USDT" in result
        assert len(result["BTC/USDT"]) == 2
        assert list(result["BTC/USDT"].columns) == ["open", "high", "low", "close", "volume"]

    def test_multi_batch_pagination(self, tmp_path: Path) -> None:
        """Mock exchange returns exactly limit candles, then fewer."""
        batch1 = [[1704067200000 + i * 300_000, 100, 105, 95, 102, 1000] for i in range(1000)]
        batch2 = [[batch1[-1][0] + 300_000, 100, 105, 95, 102, 1000]]
        exchange = MagicMock()
        exchange.fetch_ohlcv.side_effect = [batch1, batch2]

        paths = DataPaths(raw=str(tmp_path / "raw"), intermediate="", features="", normalized="", splits="")
        result = fetch_ohlcv(
            symbols=["BTC/USDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2025-01-01",
            paths=paths,
            exchange=exchange,
        )
        assert len(result["BTC/USDT"]) == 1001
        assert exchange.fetch_ohlcv.call_count == 2

    def test_empty_response(self, tmp_path: Path) -> None:
        exchange = MagicMock()
        exchange.fetch_ohlcv.return_value = []

        paths = DataPaths(raw=str(tmp_path / "raw"), intermediate="", features="", normalized="", splits="")
        result = fetch_ohlcv(
            symbols=["BTC/USDT"],
            timeframe="5m",
            start_date="2024-01-01",
            end_date="2024-01-02",
            paths=paths,
            exchange=exchange,
        )
        assert result["BTC/USDT"].empty


# ── detect_candle_gaps ───────────────────────────────────────────────────


class TestDetectCandleGaps:
    def test_no_gaps(self) -> None:
        df = _make_ohlcv_df(periods=10)
        gaps = detect_candle_gaps(df, "5m")
        assert gaps.empty

    def test_detects_single_gap(self) -> None:
        df = _make_ohlcv_df(periods=10, drop_indices=[3, 4])
        gaps = detect_candle_gaps(df, "5m")
        assert len(gaps) == 1
        assert gaps.iloc[0]["missing_count"] == 2

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        gaps = detect_candle_gaps(df, "5m")
        assert gaps.empty


# ── clean_single_symbol ─────────────────────────────────────────────────


class TestCleanSingleSymbol:
    def test_ffill_small_gap(self) -> None:
        """Gaps ≤ 3 should be forward-filled and flagged is_filled=True."""
        df = _make_ohlcv_df(periods=20, drop_indices=[5, 6])
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        assert len(cleaned) == 20
        assert cleaned["is_filled"].sum() >= 2

    def test_stale_flag_medium_gap(self) -> None:
        """Gaps between ffill_limit+1 and stale_flag_limit get is_stale=True."""
        indices_to_drop = list(range(5, 12))  # 7 missing
        df = _make_ohlcv_df(periods=30, drop_indices=indices_to_drop)
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        stale_rows = cleaned[cleaned["is_stale"]]
        assert len(stale_rows) > 0

    def test_segment_split_large_gap(self) -> None:
        """Gaps > segment_split_beyond create new segment_id values.

        With stale_flag_limit=20, a gap of 50 candles means 20 get ffilled and
        30 get dropped, leaving a 30-candle gap in the cleaned output (> 20).
        """
        indices_to_drop = list(range(10, 60))  # 50 missing
        df = _make_ohlcv_df(periods=80, drop_indices=indices_to_drop)
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        assert cleaned["segment_id"].nunique() > 1

    def test_idempotent_clean(self) -> None:
        """Cleaning already-clean data should not change it (besides added columns)."""
        df = _make_ohlcv_df(periods=20)
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        assert len(cleaned) == 20
        assert cleaned["is_filled"].sum() == 0
        assert cleaned["is_stale"].sum() == 0
        assert cleaned["segment_id"].nunique() == 1

    def test_empty_df(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        assert cleaned.empty


class TestCleanOhlcv:
    def test_saves_parquet(self, tmp_path) -> None:
        config = _make_config(tmp_path)
        raw = {"TEST/USDT": _make_ohlcv_df(periods=20)}
        result = clean_ohlcv(raw, config)
        assert "TEST/USDT" in result
        parquet_path = tmp_path / "intermediate" / "TESTUSDT_clean.parquet"
        assert parquet_path.exists()


class TestOhlcValidation:
    def test_negative_prices_flagged(self) -> None:
        df = _make_ohlcv_df(periods=10)
        df.iloc[3, df.columns.get_loc("close")] = -5.0
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        assert pd.isna(cleaned.iloc[3]["close"])

    def test_invalid_ohlc_relationship(self) -> None:
        """low > high should trigger a warning (data kept, but logged)."""
        df = _make_ohlcv_df(periods=10)
        df.iloc[2, df.columns.get_loc("low")] = 200.0
        df.iloc[2, df.columns.get_loc("high")] = 50.0
        cleaned = clean_single_symbol(df, "5m", ffill_limit=3, stale_flag_limit=20, segment_split_beyond=20)
        assert len(cleaned) == 10  # data not dropped, just warned
