import logging
from pathlib import Path
from typing import Any
import joblib
from config import LOG_LEVEL, LOG_FILENAME, LOG_DIR, MARKET_TZ, MARKET_CLOSE


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_level: str = LOG_LEVEL, log_file: Path = LOG_DIR / LOG_FILENAME) -> None:
    level = getattr(logging, log_level.upper())
    handlers = [logging.StreamHandler()]
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

