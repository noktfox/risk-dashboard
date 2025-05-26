from pathlib import Path
from typing import Any
import joblib


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def cache_to_file(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_from_cache(path: Path) -> Any:
    if path.exists():
        return joblib.load(path)
    return None

