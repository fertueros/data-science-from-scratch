#from .distance_metrics import*
from ..metrics.pairwise import*

class KNNBase:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return euclidean_distances(x1, x2)
        elif self.metric == 'manhattan':
            return manhattan_distances(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    