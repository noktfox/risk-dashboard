import pandas as pd
import yfinance as yf

import sys
import logging
logger = logging.getLogger(__name__)

from config import BENCHMARK_TICKER

from modules.clusterer import Clusterer
from modules.features import FeatureEngineer
from modules.fetcher import DataFetcher
from modules.risk_grouper import RiskGrouper
from modules.utils import configure_logging


def main():
    print("Risk Comparer: find similar-risk peers for a ticker within the same sector")

    configure_logging()

    fetcher = DataFetcher()
    feat_eng = FeatureEngineer()

    # Get all stock tickers on the market
    try:
        fetcher.load_tickers()
    except Exception as e:
        logger.error("Data fetch error: %s", e)
        print(f"Sorry, we couldn't fetch market tickers at this time.")
        sys.exit(1)

    # Get input ticker
    while True:
        ticker = input("Ticker symbol to analyze (e.g. AAPL): ")
        # Format ticker to yfinance standards
        ticker = ticker.upper()
        ticker = ticker.replace(".", "-")
        # Validate input ticker
        try:
            fetcher.validate_ticker(ticker)
            break
        except KeyError as e:
            logger.error("Invalid ticker: %s", e)
            print("We don't have information on that ticker.")

    sector: str = fetcher.fetch_sector(ticker)
    sector_tickers: list = fetcher.fetch_sector_tickers(sector)

    # Build feature matrix of all tickers in the same sector
    benchmark_prices = fetcher.fetch_ticker_history(BENCHMARK_TICKER)
    features: list = []
    for t in sector_tickers:
        ticker_prices = fetcher.fetch_ticker_history(t)
        feature_vect = feat_eng.build_features(ticker_prices, benchmark_prices)
        features.append(feature_vect)
    feature_matrix: pd.DataFrame = pd.DataFrame(features, index=sector_tickers)

    clusterer = Clusterer(sector)
    clusterer.fit(feature_matrix)
    cluster_labels = clusterer.predict(feature_matrix)

    # Get similar-risk stock tickers
    risk_grouper = RiskGrouper()
    peer_tickers = risk_grouper.group(ticker, feature_matrix, cluster_labels)
    # Quit program if no other peer tickers exist is the same risk cluster
    if len(peer_tickers) == 0:
        print("There are no similar-risk peers for this ticker based on our cluster model. Must be unique!")
        sys.exit(1)

    # Output results
    print(f"Sector: {sector}")
    print(f"Risk analysis for {ticker}:")
    input_ticker_prices = fetcher.fetch_ticker_history(ticker)
    print(feat_eng.build_features(input_ticker_prices, benchmark_prices).to_string())


    print("\nRisk analysis for similar-risk tickers:\n")
    for t in peer_tickers:
        ticker_obj = yf.Ticker(t)
        company_name = ticker_obj.info['longName']
        print(f"{t}: {company_name}")
        print(feature_matrix[feature_matrix.index == t].squeeze().to_string())
        print("\n")


if __name__ == "__main__":
    main()