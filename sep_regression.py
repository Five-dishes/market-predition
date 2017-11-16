import numpy as np
import pandas as pd
import operator
from copy import deepcopy
from sklearn.cluster import k_means


def gen_weight(km: [np.array, np.array, float], history_len=28):
    centers = km[0].reshape(-1).astype(np.float64)
    labels = km[1]
    index_sort = centers.argsort()  # 将中心点的index按中心点的大小排序
    target_class = labels[0]  # 找到[-7, -14...]天的均值所属的类
    one_period_weights = np.zeros_like(labels).astype(np.float64)
    one_period_weights[labels == labels[0]] = 1  # 与之同类的权重设为1
    if target_class == index_sort[0] and centers[index_sort[1]] != 0.0:
        # 设置中间那个类的权重，最大的那个类权重为0
        one_period_weights[labels == index_sort[1]] = centers[index_sort[0]] / centers[index_sort[1]]
    elif target_class == index_sort[2] and centers[index_sort[1]] != 0.0:
        # 设置中间那个类的权重，最小的那个类权重为0
        one_period_weights[labels == index_sort[1]] = centers[index_sort[1]] / centers[index_sort[2]]
    n_period = history_len // len(labels) + 1
    weights = [one_period_weights.reshape(1, -1)] * n_period
    weights = np.concatenate(weights).reshape(-1)
    return weights[-28:]


class SepRegression:
    def __init__(self, smooth):
        self.weights = None
        self.smooth = smooth
        self.history_data = None

    def fit(self, X):  # 周期识别
        sequence = deepcopy(X)
        history = np.arange(-1, -len(sequence), -1).astype(np.int)  # [-1, -2 ,-3...]
        for period in range(7, 8, 7):
            sliced_seq = []
            slices = []
            for i in range(0, period):
                cur_slice = history[history % period == i]
                slices.append(cur_slice)
                sliced_seq.append(sequence[cur_slice])
            # slices: [ [mod 7 = 0], [mod 7 = 1] ...]
            # sliced_seq: [ sequence[mod 7 = 0], sequence[mod 7 = 1] ...]
            means = []
            for seq in sliced_seq:
                seq.sort()
                seq = seq[2:]  # delete small number
                means.append(seq.mean())  # [[-7, -14...天的销量的平均值], [-6, -13..天的~]...]
            means = np.array(means).reshape(-1, 1)
            km = k_means(means, n_clusters=3, max_iter=100)
            self.weights = gen_weight(km)
        # self.history_data = sequence[-28: 0]

    def predict(self, X: np.array):
        X = X[0]
        X = X[-28:]
        min_ = X.min()
        max_ = X.max()
        error_dict = {}
        if min_ != max_:
            for pred in np.arange(min_, max_, 0.02):
                error_dict[pred] = (((X - pred)*self.weights)**2).mean()
            best_pred = min(error_dict.items(), key=operator.itemgetter(1))[0]
            if self.smooth:
                return np.array([best_pred]).round()
            else:
                return np.array([best_pred])
        else:
            return np.array([min_])


if __name__ == '__main__':
    wr = SepRegression(smooth=False)
    matrix = pd.read_csv(
        'processed_2.csv', header=None, sep=',', encoding='gbk').values
    wr.fit(matrix[0:45][:, 1:-1])
    for row in range(30):
        X, y = matrix[row][1:-1], matrix[row][-1]
        print(y)
        print(wr.predict([X]))
