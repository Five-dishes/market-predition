import numpy as np


class WeekRegression:
    def fit(self, x, y):
        pass

    def predict(self, X: np.array):
        X = X[0]
        return np.array([np.mean(X[np.array(range(-7, 0))])])
