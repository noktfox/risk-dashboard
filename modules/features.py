import pandas as pd
import numpy as np

from config import TRADING_DAYS
class FeatureEngineer:
    """
    Computes risk metrics and features for clustering.
    """

    def __init__(self, trading_days: int=TRADING_DAYS):
        self.trading_days = trading_days

    def calculate_log_returns(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Compute daily log returns.
        """
        close_prices = price_df['Close']
        return np.log(close_prices / close_prices.shift(1)).dropna()

    def calculate_annualized_volatility(self, returns: pd.Series) -> float:
        """
        Annualized volatility based on daily returns.
        """
        return returns.std(ddof=0) * np.sqrt(self.trading_days)

    def calculate_annualized_return(self, returns: pd.Series) -> float:
        """
        Annualized return based on daily returns.
        """
        if returns.empty:
            return 0.0
        total_growth = np.prod(1 +returns)
        n_periods = len(returns)
        return total_growth ** (self.trading_days / n_periods) - 1

    def calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Beta relative to a benchmark (e.g. market index).
        """
        aligned_ret, aligned_bench = returns.align(benchmark_returns, join='inner')
        covariance = aligned_ret.cov(aligned_bench)
        variance = aligned_bench.var(ddof=0)
        return covariance / variance if variance != 0 else np.nan

    def build_features(self, price_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.Series:
        """
        Build feature vector including annual return, volatility, and beta.
        """
        log_returns = self.calculate_log_returns(price_df)
        benchmark_returns = self.calculate_log_returns(benchmark_df)
        return pd.Series({
            'annual_return': self.calculate_annualized_return(log_returns),
            'volatility': self.calculate_annualized_volatility(log_returns),
            'beta': self.calculate_beta(log_returns, benchmark_returns)
        })
