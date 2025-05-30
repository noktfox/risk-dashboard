import numpy as np
import pandas as pd

from config import DEFAULT_N_PEERS

class RiskGrouper:
    """
    Group similar-risk tickers within the same cluster of a sector space.
    """

    def __init__(self, n_peers=DEFAULT_N_PEERS):
        self.n_peers = n_peers


    def group(
            self,
            ticker: str,
            feature_matrix: pd.DataFrame,
            cluster_labels: pd.Series
    ) -> list[str]:
        """
        Gets 'n' closest peer tickers in the same risk cluster.

        :arg:
            ticker (str): Ticker to get the closest peer tickers for.
            feature_matrix (pd.DataFrame): Matrix of all sector tickers risk features with tickers as indices.
            cluster_labels (pd.Series): Series of the closest cluster center of each ticker in a sector,
                                        with tickers as indices.

        :return:
            list[str]: List of the closest peer tickers in the same risk cluster.
        """

        # Get the cluster label of the given ticker and all other tickers in that cluster
        ticker_cluster: int = cluster_labels[ticker]
        peer_tickers = cluster_labels[cluster_labels == ticker_cluster].index.drop(ticker)
        peer_tickers = peer_tickers.tolist()

        # Return an empty list if no other peers in the same cluster
        if not peer_tickers:
            return []

        # Get the risk vector of the given ticker and peer tickers in the sector space
        ticker_vector = feature_matrix.loc[ticker].values
        vectors = feature_matrix.loc[peer_tickers].values

        # Calculate the distance (Euclidean norm) between the ticker and all peer tickers
        distances = np.linalg.norm(vectors - ticker_vector, axis=1)
        # Order peers based from closest to longest distance
        order = np.argsort(distances)
        sorted_peers = [peer_tickers[i] for i in order]

        # Return 'n' closest peer tickers
        return sorted_peers[:self.n_peers]