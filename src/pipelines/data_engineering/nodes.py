"""Data Engineering nodes.

Functions for fetching, validating, and cleaning OHLCV data from Binance.
"""

from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

from src.utils.config import AppConfig, DataPaths
from src.utils.logger import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def create_exchange() -> ccxt.binance:
    """Create a Binance exchange instance with rate limiting."""
    return ccxt.binance({"enableRateLimit": True})


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert a ccxt timeframe string (e.g. '5m', '1h') to milliseconds.

    Args:
        timeframe: Candle interval string like '1m', '5m', '1h', '1d'.

    Returns:
        Equivalent duration in milliseconds.

    Raises:
        ValueError: If the timeframe unit is not recognised.
    """
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in multipliers:
        raise ValueError(f"Unknown timeframe unit '{unit}' in '{timeframe}'")
    return amount * multipliers[unit]


def fetch_ohlcv_single_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single symbol with pagination.

    Args:
        exchange: ccxt exchange instance.
        symbol: Trading pair (e.g. 'BTC/USDT').
        timeframe: Candle interval string.
        since_ms: Start timestamp in milliseconds.
        until_ms: End timestamp in milliseconds.

    Returns:
        DataFrame with datetime index and OHLCV columns.
    """
    all_candles: list[list] = []
    cursor = since_ms
    limit = 1000

    while cursor < until_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not candles:
            break

        all_candles.extend(candles)
        last_ts = candles[-1][0]
        cursor = last_ts + timeframe_to_milliseconds(timeframe)

        if len(candles) < limit:
            break
        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df[df.index <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    return df


def fetch_ohlcv(
    symbols: list[str],
    timeframe: str,
    start_date: str,
    end_date: str,
    paths: DataPaths,
    exchange: ccxt.Exchange | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all symbols and save raw Parquet files.

    Args:
        symbols: List of trading pairs.
        timeframe: Candle interval string.
        start_date: ISO date string for start.
        end_date: ISO date string for end.
        paths: DataPaths with directory locations.
        exchange: Optional exchange instance (created if None).

    Returns:
        Dict mapping symbol to its raw DataFrame.
    """
    if exchange is None:
        exchange = create_exchange()

    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    raw_dir = Path(paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        logger.info("Fetching %s (%s)", symbol, timeframe)
        df = fetch_ohlcv_single_symbol(exchange, symbol, timeframe, since_ms, until_ms)
        logger.info("  -> %d candles", len(df))

        safe_name = symbol.replace("/", "")
        out_path = raw_dir / f"{safe_name}_{timeframe}.parquet"
        df.to_parquet(out_path)
        result[symbol] = df

    return result


def detect_candle_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Detect gaps in the candle index.

    Returns:
        DataFrame with columns ['gap_start', 'gap_end', 'missing_count'].
    """
    if df.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_count"])

    freq_ms = timeframe_to_milliseconds(timeframe)
    freq = pd.Timedelta(milliseconds=freq_ms)

    expected = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing = expected.difference(df.index)

    if missing.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_count"])

    gaps = []
    gap_start = missing[0]
    prev = missing[0]
    count = 1

    for ts in missing[1:]:
        if ts - prev == freq:
            count += 1
            prev = ts
        else:
            gaps.append({"gap_start": gap_start, "gap_end": prev, "missing_count": count})
            gap_start = ts
            prev = ts
            count = 1

    gaps.append({"gap_start": gap_start, "gap_end": prev, "missing_count": count})
    return pd.DataFrame(gaps)


def clean_single_symbol(
    df: pd.DataFrame,
    timeframe: str,
    ffill_limit: int,
    stale_flag_limit: int,
    segment_split_beyond: int,
) -> pd.DataFrame:
    """Clean OHLCV data for a single symbol.

    Policy:
    - Reindex to full expected range.
    - Forward-fill gaps ≤ ffill_limit candles.
    - Flag stale data for gaps between ffill_limit+1 and stale_flag_limit.
    - Split into segments for gaps > segment_split_beyond.
    - Validate: no negative prices, OHLC relationship (low ≤ open,close ≤ high).

    Adds columns: is_filled, is_stale, segment_id.
    """
    if df.empty:
        return df

    freq_ms = timeframe_to_milliseconds(timeframe)
    freq = pd.Timedelta(milliseconds=freq_ms)

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_index)

    was_missing = df["close"].isna()

    df = df.ffill(limit=stale_flag_limit)

    df = df.dropna(subset=["close"])

    if df.empty:
        return df

    df["is_filled"] = False
    df["is_stale"] = False

    missing_timestamps = was_missing[was_missing].index.tolist()
    if missing_timestamps:
        gap_runs: list[list] = []
        current_run = [missing_timestamps[0]]
        for ts in missing_timestamps[1:]:
            if ts - current_run[-1] == freq:
                current_run.append(ts)
            else:
                gap_runs.append(current_run)
                current_run = [ts]
        gap_runs.append(current_run)

        for run in gap_runs:
            gap_len = len(run)
            for ts in run:
                if ts in df.index:
                    if gap_len <= ffill_limit:
                        df.loc[ts, "is_filled"] = True
                    elif gap_len <= stale_flag_limit:
                        df.loc[ts, "is_filled"] = True
                        df.loc[ts, "is_stale"] = True

    segment_id = 0
    segments = pd.Series(0, index=df.index, dtype=int)
    prev_idx = None
    for idx in df.index:
        if prev_idx is not None:
            gap_candles = int((idx - prev_idx) / freq) - 1
            if gap_candles > segment_split_beyond:
                segment_id += 1
        segments[idx] = segment_id
        prev_idx = idx
    df["segment_id"] = segments

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            neg_mask = df[col] < 0
            if neg_mask.any():
                logger.warning("Negative prices found in '%s', setting to NaN", col)
                df.loc[neg_mask, col] = pd.NA

    if all(c in df.columns for c in price_cols):
        bad_ohlc = (
            (df["low"] > df["open"])
            | (df["low"] > df["close"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
        )
        if bad_ohlc.any():
            logger.warning("%d rows with invalid OHLC relationship", bad_ohlc.sum())

    return df


def clean_ohlcv(raw_data: dict[str, pd.DataFrame], config: AppConfig) -> dict[str, pd.DataFrame]:
    """Clean all symbols and save intermediate Parquet files."""
    policy = config.data.missing_candle_policy
    out_dir = Path(config.data.paths.intermediate)
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}
    for symbol, df in raw_data.items():
        logger.info("Cleaning %s (%d raw candles)", symbol, len(df))
        gaps = detect_candle_gaps(df, config.timeframe)
        if not gaps.empty:
            logger.info("  %d gap(s) detected", len(gaps))

        cleaned = clean_single_symbol(
            df,
            config.timeframe,
            ffill_limit=policy.ffill_limit,
            stale_flag_limit=policy.stale_flag_limit,
            segment_split_beyond=policy.segment_split_beyond,
        )
        logger.info("  -> %d clean candles", len(cleaned))

        safe_name = symbol.replace("/", "")
        out_path = out_dir / f"{safe_name}_clean.parquet"
        cleaned.to_parquet(out_path)
        result[symbol] = cleaned

    return result
