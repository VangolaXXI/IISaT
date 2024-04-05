import numpy as np

def generate_synthetic_data():
    np.random.seed(0)
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, size=1000)
    return X, y

def train_test_split(X, y, test_size=0.2):
    np.random.seed(0)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_samples = int(test_size * len(X))
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
