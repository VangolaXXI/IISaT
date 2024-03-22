from collections import Counter
import numpy as np
from distances import euclidean_distance, manhattan_distance, chebyshev_distance

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.dist_func = self._get_distance_function(distance_metric)

    def _get_distance_function(self, metric):
        if metric == 'euclidean':
            return euclidean_distance
        elif metric == 'manhattan':
            return manhattan_distance
        elif metric == 'chebyshev':
            return chebyshev_distance
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [self.dist_func(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
