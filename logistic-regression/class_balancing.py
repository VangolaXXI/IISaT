import numpy as np

def balance_classes(X, y):
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    min_samples = min(len(class_0_indices), len(class_1_indices))
    balanced_indices = np.concatenate((class_0_indices[:min_samples], class_1_indices[:min_samples]))

    return X[balanced_indices], y[balanced_indices]
