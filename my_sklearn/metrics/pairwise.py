import numpy as np

def euclidean_distances(x1, x2):
    """
    Compute euclidena distance between two points
    """
    return np.sqrt(np.sum((x1 - x2)**2))

def manhattan_distances(x1, x2):
    """
    Compute euclidena distance between two points
    """
    return np.sqrt(np.abs(x1 - x2))