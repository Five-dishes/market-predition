import xgboost as xgb
import numpy as np


class XGboost:
    def __init__(self):
        self.params = {
            "objective": "reg:linear",
            "eta" : 0.05, "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "silent": 1,
        }
        self.num_round = 30

    def fit(self, x, y):
        dtrain = xgb.DMatrix(x, y)
        self.gbm = xgb.train(self.params, dtrain, self.num_round)
    def predict(self, X: np.array):
        dpredict = xgb.DMatrix(X)
        predict = self.gbm.predict(dpredict)
        return np.array([predict])