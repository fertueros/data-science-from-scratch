from knn_base import KNNBase
from collections import Counter
import numpy as np

class KNNClassifier(KNNBase):
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distance between x and all examples in the training set
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label among k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
