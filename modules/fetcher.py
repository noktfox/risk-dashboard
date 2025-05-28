import pandas as pd
import yfinance as yf

from config import RAW_DATA_DIR, CACHE_DATA_DIR
from modules.utils import ensure_dir, is_outdated


class DataFetcher:
    """
    Handles fetching and caching of price and metadata with yfinance.
    """

    def __init__(self):
        # Check that data directories exist
        ensure_dir(CACHE_DATA_DIR)
        ensure_dir(RAW_DATA_DIR)

    def fetch_price(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        Check cache first, if missing, fetch from yfinance and cache as CSV.
        """
        cache_file = CACHE_DATA_DIR / f"{ticker}_{period}_{interval}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache_file)
        return df

    def fetch_sector(self, ticker: str) -> str:
        """
        Fetch sector metadata for a ticker.
        """
        info = yf.Ticker(ticker).info
        return info.get("sector")