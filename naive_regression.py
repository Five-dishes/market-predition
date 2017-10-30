import numpy as np


class NaiveRegression:
    def fit(self, x, y):
        pass

    def predict(self, X: np.array):
        X = X[0]
        return np.array([np.mean(X[np.array([-7, -14, -21, -28])])])
