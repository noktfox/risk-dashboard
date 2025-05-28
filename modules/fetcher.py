import pandas as pd
import yfinance as yf

from config import RAW_DATA_DIR, CACHE_DATA_DIR, TICKERS_FILENAME
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
        if cache_file.exists() and not is_outdated(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(cache_file)
        return df

    def fetch_sector(self, ticker: str) -> str:
        """
        Fetch sector metadata for a ticker.
        """
        csv_path = RAW_DATA_DIR / TICKERS_FILENAME
        df = pd.read_csv(csv_path).set_index("ticker")
        return df.at[ticker, "sector"]


    def fetch_sector_tickers(self, sector: str) -> list:
        """
        Fetch all tickers in a given sector.
        """
        df = pd.read_csv(RAW_DATA_DIR / TICKERS_FILENAME)
        return df[df["sector"] == sector]["ticker"].tolist()

    def fetch_tickers(self):
        """
        Fetch all tickers in the S&P 500
        """
        csv_path = RAW_DATA_DIR / TICKERS_FILENAME
        if not csv_path.exists():
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url, header=0)
            sp500 = tables[0]
            ticker_df = pd.DataFrame({
                "ticker": sp500["Symbol"].str.replace(".", "-", regex=False), # convert to yahoo finance formatting
                "sector": sp500["GICS Sector"]
            })
            ticker_df.to_csv(csv_path, index=False)