import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import MODEL_DIR, MODEL_FILENAME
from modules.utils import ensure_dir, load_from_cache, is_outdated, cache_to_file

class Clusterer:
    """
    Fits a clustering model to group stocks in a given sector based on risk features.
    Handles saving/loading of the model.
    """

    def __init__(self, sector: str):
        self.sector: str = sector.replace(" ", "_").lower()
        self.file_name: str = f"{self.sector}_cluster_model.joblib"
        ensure_dir(MODEL_DIR)
        self.model = None

    def fit(self, feature_matrix: pd.DataFrame) -> KMeans:
        """
        Fit a KMeans clustering model to a feature matrix and save it to disk.

        :arg:
            feature_matrix (pd.DataFrame): The matrix of all sector tickers risk features with tickers as indices.

        :return:
            KMeans: The sector tickers clustering model
        """
        
        # Check that there is no existing model and model is not outdated
        if not self.model and is_outdated(MODEL_DIR / MODEL_FILENAME):
            # Use elbow method to determine number of clusters
            inertias: dict = self.calculate_inertias(feature_matrix)
            n_clusters: int = self.find_elbow(inertias)
            # Fit model
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
            self.model.fit(feature_matrix)
            # Save model to disk
            cache_to_file(self.model, MODEL_DIR / self.file_name)
        return self.model

    def load_model(self) -> KMeans:
        """Load a clustering model from disk if not already loaded."""
        if self.model is None:
            self.model = load_from_cache(MODEL_DIR / MODEL_FILENAME)
        return self.model

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """
        Predict cluster labels for each ticker in a feature matrix.

        :arg:
            feature_matrix (pd.DataFrame): The matrix of all sector tickers risk features with tickers as indices.

        :return:
            pd.Series: Series of the closest cluster center of each ticker with tickers as indices.
        """
        index_labels: np.array = self.load_model().predict(feature_matrix)
        return pd.Series(index_labels, index=feature_matrix.index)

    def calculate_inertias(self, feature_matrix: pd.DataFrame, k_range=range(1, 11)) -> dict:
        """
        Calculate inertias for each clustering model fitted to a range of 'k' clusters.
        Used for the elbow method in evaluating 'n' clusters.

        :arg:
            feature_matrix (pd.DataFrame): The matrix of all sector tickers risk features with tickers as indices.
            k_range: the range of 'k' clusters to be evaluated at.

        :return:
            dict: inertia value for each clustering model fitted to key 'k' clusters.
        """
        inertias: dict = {}
        for k in k_range:
            k_model: KMeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
            k_model.fit(feature_matrix)
            inertias[k] = k_model.inertia_
        return inertias

    def find_elbow(self, inertia_dict: dict) -> int:
        """
        Finds the 'k' elbow point of inertias (point of greatest steepest change in inertia).
        This is most optimal 'n' clusters.

        :arg:
            inertia_dict (dict): inertia value for each clustering model fitted to key 'k' clusters.

        :return:
            int: the 'k' elbow point
        """
        inertias: list = list(inertia_dict.values())
        # Compute second derivative of each 'k' inertia using finite difference method
        second_diffs: list = [inertias[i+2] - 2*inertias[i+2] + inertias[i] for i in range(len(inertias) - 2)]
        # Find point of greatest second derivative (steepest change)
        elbow_k: int = second_diffs.index(max(second_diffs))
        # Add 2 to align indices (second_diffs[i] corresponds to inertia_dict[i+1]) and convert index to count
        elbow_k += 2
        return elbow_k