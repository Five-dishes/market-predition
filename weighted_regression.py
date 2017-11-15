import numpy as np
import pandas as pd
import operator


class WeightRegression:
    def __init__(self, smooth):
        self.weights = np.array([3, 1.2, 0.7, 0.5, 0.6, 0.8, 1.2]*4).astype(np.float64)
        time_decay_mask = np.arange(0.7, 1.5, (1.5-0.7)/28)
        self.weights *= time_decay_mask
        self.smooth = smooth

    def fit(self, x, y):
        pass

    def predict(self, X: np.array):
        X = X[0]
        min_ = X.min()
        max_ = X.max()
        error_dict = {}
        if min_ != max_:
            for pred in np.arange(min_, max_, 0.05):
                error_dict[pred] = (((X - pred)*self.weights)**2).mean()
            best_pred = min(error_dict.items(), key=operator.itemgetter(1))[0]
            if self.smooth:
                return np.array([best_pred]).round()
            else:
                return np.array([best_pred])
        else:
            return np.array([min_])


if __name__ == '__main__':
    wr = WeightRegression(smooth=False)
    print(wr.weights)
    matrix = pd.read_csv(
        'processed_2.csv', header=None, sep=',', encoding='gbk').values
    for row in range(30):
        X, y = matrix[row][1:-1], matrix[row][-1]
        print(y)
        print(wr.predict([X]))
