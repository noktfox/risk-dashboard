import logging
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
import joblib

from config import LOG_LEVEL, LOG_FILENAME, LOG_DIR, MARKET_TZ, MARKET_CLOSE


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_level: str = LOG_LEVEL, log_file: Path = LOG_DIR / LOG_FILENAME) -> None:
    level = getattr(logging, log_level.upper())
    handlers = []
    if log_file:
        ensure_dir(log_file.parent)
        handlers.append(logging.FileHandler(log_file, mode="a"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def cache_to_file(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_from_cache(path: Path) -> Any:
    if path.exists():
        return joblib.load(path)
    return None


def get_last_modified(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime)


def is_outdated(path: Path, market_close: time = MARKET_CLOSE) -> bool:
    """Check if a path is outdated based on daily market data. Used to refresh cached files by daily basis."""

    # Return true if the path does not exist
    if not path.exists():
        return True

    # Check if last data refresh is earlier than last trading day close
    last_modified = datetime.fromtimestamp(path.stat().st_mtime)
    now = datetime.now(ZoneInfo(MARKET_TZ))
    # Determine last trading (business) day based current week day
    day_of_wk: int = now.weekday()
    if day_of_wk == 0:
        # Change from Sun to Fri
        last_trading_day = now.date() - timedelta(days=2)
    elif day_of_wk == 6:
        # Change from Mon to Fri
        last_trading_day = now.date() - timedelta(days=3)
    elif day_of_wk == 5:
        # Change from Sat to Fri
        last_trading_day = now.date() - timedelta(days=1)
    else:
        last_trading_day = now.date() - timedelta(days=1)
    # Combine last trading day with market close time
    reference_date = datetime.combine(last_trading_day, market_close)
    return last_modified < reference_date