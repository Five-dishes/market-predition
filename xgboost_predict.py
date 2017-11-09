import xgboost as xgb
import numpy as np


class XGboost:
    def __init__(self):
        self.params = {
            "objective": "reg:linear",
            "eta" : 0.15, "max_depth": 3,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "silent": 1}
        self.num_round = 30

    def fit(self, x, y):
        dtrain = xgb.DMatrix(x, y)
        self.gbm = xgb.train(self.params, dtrain, self.num_round)
        #self.gbm = xgb.XGBRegressor(max_depth = 3,learning_rate=0.03,n_estimators=1600,reg_alpha=1,reg_lambda=0)
        #self.gbm.fit(x, y)

    def predict(self, X: np.array):
        dpredict = xgb.DMatrix(X)
        predict = self.gbm.predict(dpredict)
        return np.array([predict])