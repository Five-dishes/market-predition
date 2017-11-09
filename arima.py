import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA


class ARIMA_:
    def __init__(self, pdq):
        self.model = None
        self.error = False
        self.pdq = pdq

    def fit(self, X):
        model = ARIMA(X, order=self.pdq)
        try:
            self.error = False
            self.model = model.fit(disp=0)
        except np.linalg.linalg.LinAlgError:
            print('LinAlgError')
            self.error = True
        except Exception as e:
            print('Unknown Error {}'.format(e))
            self.error = True

    def predict(self, X: np.array):
        if self.error:
            return np.array([1000.0])
        result = self.model.forecast()[0]
        return np.array([result])


if __name__ == '__main__':
    matrix = pd.read_csv(
        'processed_2.csv', header=None, sep=',', encoding='gbk').values.astype(np.float64)
    X = matrix[0:78][:, 1:]
    series = np.concatenate((X[0][:-1], X[:, -1]))
    spliter = int(len(series) * 0.8)
    train, test = series[0: spliter], series[spliter:]
    model = ARIMA_((21, 0, 0))

    for i in range(len(test)):
        model.fit(train)
        result = model.predict(train)
        print(result)
        train = np.concatenate((train, result.reshape(1)))
    predicted = train[-len(test):]
    print(predicted)
    print(test)
    print('MSE: {}'.format(((predicted - test)**2).mean()))
