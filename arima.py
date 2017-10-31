import numpy as np
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
            self.error = True
        except:
            self.error = True

    def predict(self, X: np.array):
        if self.error:
            return np.array([1000.0])
        result = self.model.forecast()[0]
        return np.array([result])
