from ..metrics.pairwise import*
from ._kd_tree import KDTree

class KNNBase:
    def __init__(self, k=5, metric='euclidean', algorithm = 'brute_force'):
        self.k = k
        self.metric = metric
        self.algorithm = algorithm

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.algorithm == 'kd_tree':
            self.tree = KDTree()
            self.tree.root = self.tree._build_tree(X, np.arange(len(X)))
    
    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return euclidean_distances(x1, x2)
        elif self.metric == 'manhattan':
            return manhattan_distances(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    