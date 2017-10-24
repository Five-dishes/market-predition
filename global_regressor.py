# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle


df = pd.read_csv('./processed_2.csv', sep=',', header=None)
data_set = shuffle(df).values
dim = len(data_set[0]) - 1
print("Feature dim is: {}".format(dim))
X: pd.DataFrame = data_set[:, 0:dim]
Y: pd.DataFrame = data_set[:, dim]


# define base model
def model_wrapper(feature_dim):
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(feature_dim, input_dim=feature_dim,
                        kernel_initializer='normal', activation='relu'))
        model.add(Dense(feature_dim, input_dim=feature_dim,
                        kernel_initializer='normal', activation='relu'))
        model.add(Dense(feature_dim, input_dim=feature_dim,
                        kernel_initializer='normal', activation='relu'))
        model.add(Dense(feature_dim, input_dim=feature_dim,
                        kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    return baseline_model


seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=model_wrapper(dim), nb_epoch=100, batch_size=5, verbose=1)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


