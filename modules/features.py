import pandas as pd
import numpy as np

from config import TRADING_DAYS

class FeatureEngineer:
    """
    Computes risk metrics for a stock and builds a stock's risk-feature vector.
    """

    def __init__(self, trading_days: int=TRADING_DAYS):
        self.trading_days = trading_days

    def calculate_log_returns(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Calculate daily log returns of a stock.

        :arg:
            price_df (pd.DataFrame): Dataframe of a stock's historical data over a set period and intervals,
                                     indexed by date.
        """
        close_prices = price_df['Close']
        return np.log(close_prices / close_prices.shift(1)).dropna()

    def calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """
        Calculate annualized volatility of a stock based on daily returns.

        :args:
            returns (pd.Series): Series of price returns over a set period and interval.
                                 Expects daily interval to correctly convert to annualized metric.
        """
        return returns.std(ddof=0) * np.sqrt(self.trading_days)

    def calculate_annualized_return(self, returns: pd.Series) -> float:
        """
        Calculate annualized return of a stock based on daily returns.
        Uses geometric compounding formula.

        :args:
            returns (pd.Series): Series of price returns over a set period and interval.
                                 Expects daily interval to correctly convert to annualized metric.
        """
        if returns.empty:
            return 0.0
        total_growth = np.prod(1 +returns)
        n_intervals = len(returns)
        return total_growth ** (self.trading_days / n_intervals) - 1

    def calculate_beta(self, stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta relative to a benchmark index.

        :args:
            stock_returns (pd.Series): Series of price returns for a stock over a set period and interval.
            benchmark_returns (pd.Series): Series of price returns over a set period and interval for the benchmark index.
        """

        # Align and remove missing dates of stock returns and benchmark returns
        aligned_ret, aligned_bench = stock_returns.align(benchmark_returns, join='inner')
        # Calculate beta
        covariance = aligned_ret.cov(aligned_bench)
        variance = aligned_bench.var(ddof=0)
        return covariance / variance if variance != 0 else np.nan

    def build_features(self, price_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.Series:
        """
        Build feature vector of a stock, including annual return, volatility, and beta.

        :args:
            price_df (pd.DataFrame): Dataframe of a stock's historical data over a set period and intervals,
                                     indexed by date.
            benchmark_df (pd.DataFrame): Dataframe of a benchmark's historical data over a set period and intervals,
                                     indexed by date.
        """

        # Calculate log returns of stock and benchmark
        log_returns: pd.Series = self.calculate_log_returns(price_df)
        benchmark_returns: pd.Series = self.calculate_log_returns(benchmark_df)

        # Compute features and build feature vector of the stock
        return pd.Series({
            'annual_return': self.calculate_annualized_return(log_returns),
            'volatility': self.calculate_annualized_volatility(log_returns),
            'beta': self.calculate_beta(log_returns, benchmark_returns)
        })
