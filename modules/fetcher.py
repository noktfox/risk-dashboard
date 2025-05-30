import pandas as pd
import yfinance as yf

import requests
import urllib.error

from config import RAW_DATA_DIR, CACHE_DATA_DIR, TICKERS_FILENAME, TICKERS_URL, PRICE_DATA_PERIOD, PRICE_DATA_INTERVAL
from modules.utils import ensure_dir, is_outdated


class DataFetcher:
    """
    Handles fetching and caching of price and metadata with yfinance.
    """

    def __init__(self):
        # Check that data directories exist
        ensure_dir(CACHE_DATA_DIR)
        ensure_dir(RAW_DATA_DIR)

    def fetch_ticker_history(
            self,
            ticker: str,
            period: str = PRICE_DATA_PERIOD,
            interval: str = PRICE_DATA_INTERVAL
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        :return:
            pd.DataFrame: Dataframe of a ticker's historical data over a set period and intervals, indexed by date.
        """

        # Read data from cache if exists and not outdated (from last trading day close)
        cache_file = CACHE_DATA_DIR / f"{ticker}_{period}_{interval}.csv"
        if cache_file.exists() and not is_outdated(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        # Write ticker data from yfinance API and cache to disk
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache_file)
        return df

    def fetch_sector(self, ticker: str) -> str:
        """Fetch the sector a ticker."""
        tickers_csv_path = RAW_DATA_DIR / TICKERS_FILENAME
        df = pd.read_csv(tickers_csv_path).set_index("ticker")
        return df.at[ticker, "sector"]

    def fetch_sector_tickers(self, sector: str) -> list:
        """Fetch all tickers in a given sector."""
        tickers_df: pd.DataFrame = pd.read_csv(RAW_DATA_DIR / TICKERS_FILENAME)
        return tickers_df[tickers_df["sector"] == sector]["ticker"].tolist()

    def load_tickers(self) -> None:
        """Fetch all tickers in the S&P 500 and save to disk."""
        tickers_csv_path = RAW_DATA_DIR / TICKERS_FILENAME

        if not tickers_csv_path.exists():
            # Scrape tickers from set url
            url = TICKERS_URL
            try:
                tables = pd.read_html(url, header=0)
            except requests.exceptions.RequestException:
                raise RuntimeError(f"Network error fetching tickers from {url}")
            except urllib.error.URLError:
                raise RuntimeError(f"Network error fetching tickers from {url}")
            except ValueError:
                raise ValueError(f"No tables parsed from {url} (HTML layout changed)")

            # Get first table found from url (based on url page formatting)
            sp500 = tables[0]
            try:
                tickers_df = pd.DataFrame({
                    # Convert ticker string to yfinance formatting
                    "ticker": sp500["Symbol"].str.replace(".", "-", regex=False),
                    "sector": sp500["GICS Sector"]
                })
            except KeyError:
                raise KeyError(f"Incorrect table parsed from {url} (HTML layout changed)")

            # Write list of tickers/sectors to disk
            tickers_df.to_csv(tickers_csv_path, index=False)

    def validate_ticker(self, ticker: str) -> None:
        """Check that a ticker exists in both disk-saved tickers and in yfinance database."""
        csv_path = RAW_DATA_DIR / TICKERS_FILENAME

        # Check that ticker exists in scraped list of tickers
        tickers = pd.read_csv(csv_path)
        if ticker not in tickers["ticker"].values:
            raise KeyError(f"Ticker '{ticker}' does not exist in '{TICKERS_FILENAME}'.")

        # Check that ticker exists in yfinance database
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info or "shortName" not in info:
            raise KeyError("Ticker does not exist in yfinance database.")