import numpy as np
import pandas as pd
from config import DEFAULT_N_RECOMMEND

class Recommender:
    """
    Recommends similar-risk tickers within the same cluster.
    """

    def __init__(self, n_recommend=DEFAULT_N_RECOMMEND):
        self.n_recommend = n_recommend

    def recommend(self,
                  ticker: str,
                  feature_matrix: pd.DataFrame,
                  cluster_labels: pd.Series) -> list[str]:
        """
        Given a ticker, gets the four closest tickers in its risk cluster.
        """

        label = cluster_labels[ticker]
        neighbor_tickers = cluster_labels[cluster_labels == label].index.drop(ticker)
        neighbor_tickers = neighbor_tickers.tolist()
        if not neighbor_tickers:
            return []

        ticker_vector = feature_matrix.loc[ticker].values
        vectors = feature_matrix.loc[neighbor_tickers].values

        distances = np.linalg.norm(vectors - ticker_vector, axis=1)
        order = np.argsort(distances)
        sorted_neighbors = [neighbor_tickers[i] for i in order]
        return sorted_neighbors[:self.n_recommend]