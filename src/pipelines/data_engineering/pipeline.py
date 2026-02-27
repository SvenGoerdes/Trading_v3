"""Data Engineering Pipeline.

Fetches and cleans OHLCV candle data from Binance via ccxt.
Handles missing candles according to the configured policy.
"""

from datetime import datetime, timedelta, timezone

from src.pipelines.data_engineering.nodes import clean_ohlcv, fetch_ohlcv
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging


def run() -> None:
    """Execute the data engineering pipeline."""
    setup_logging()
    logger = get_logger(__name__)
    config = get_config()

    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(tz=timezone.utc) - timedelta(days=config.data.lookback_days)).strftime("%Y-%m-%d")

    logger.info("Data Engineering Pipeline — %s to %s", start_date, end_date)
    logger.info("Symbols: %s", config.symbols)
    logger.info("Timeframe: %s", config.timeframe)

    raw = fetch_ohlcv(
        symbols=config.symbols,
        timeframe=config.timeframe,
        start_date=start_date,
        end_date=end_date,
        paths=config.data.paths,
    )

    clean = clean_ohlcv(raw, config)
    logger.info("Data engineering complete. %d symbols cleaned.", len(clean))
