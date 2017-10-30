import numpy as np
from scipy import stats


class Mode:
    def fit(self, x, y):
        pass

    def predict(self, X: np.array):
        X = X[0]
        return np.array([stats.mode(X)[0]])
