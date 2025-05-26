import joblib
from sklearn.cluster import KMeans
from config import MODEL_DIR, MODEL_FILENAME
from modules.utils import ensure_dir, load_from_cache, is_outdated, cache_to_file

class Clusterer:
    """
    Fits a clustering model to stock feature data and handles saving/loading of the model.
    """

    def __init__(self, sector: str):
        self.sector = sector.replace(" ", "_")

        self.file_name = f"{self.sector}_cluster_model.joblib"
        ensure_dir(MODEL_DIR)

        self.model = None

    def fit(self, feature_matrix):
        """
        Fit a KMeans clustering model to the feature matrix and save it.
        """
        inertias = self.calculate_inertias(feature_matrix)
        n_clusters = self.find_elbow(inertias)
        self.model = KMeans(n_clusters=n_clusters, random_state=42) # Uses lloyd algorithm by default
        self.model.fit(feature_matrix)
        joblib.dump(self.model, MODEL_DIR / self.file_name)
        return self.model

    def load_model(self):
        """
        Load the clustering model from disk if not already loaded.
        """
        if self.model is None:
            self.model = load_from_cache(MODEL_DIR / MODEL_FILENAME)
        return self.model

    def predict(self, feature_matrix):
        """
        Predict cluster labels for a feature matrix.
        """
        return self.load_model().predict(feature_matrix)

    def calculate_inertias(self, feature_matrix, k_range=range(1, 11)): 
        """
        Calculate inertia for a range of 'k' clusters.
        Used for the elbow method in evaluating 'n' clusters.
        """
        inertias = {}
        for k in k_range:
            k_model = KMeans(n_clusters=k, n_init="auto", random_state=42)
            k_model.fit(feature_matrix)
            inertias[k] = k_model.inertia_
        return inertias

    def find_elbow(self, inertia_dict):
        """
        Finds the 'k' elbow point of a dictionary of inertias for a range of 'k' clusters.
        """
        values = list(inertia_dict.values())
        # Compute second derivative of each 'k' inertia
        second_diffs = [values[i+2] - 2*values[i+2] + values[i] for i in range(len(values) - 2)]
        # Align indices - second_diffs[i] corresponds to inertia_dict[i+1]
        # Convert index to count - add 1 to get number of clusters
        print(second_diffs)
        elbow_k = second_diffs.index(max(second_diffs)) + 2
        return elbow_k