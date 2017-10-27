import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# scale train and test data to [-1, 1]
def scale(train: np.array, test: np.array):
    # fit scaler
    scaler = RobustScaler()
    scaler = scaler.fit(train.reshape(-1, 1))

    train_scaled = scaler.transform(train.reshape(-1, 1))
    test_scaled = scaler.transform(test.reshape(-1, 1))
    return scaler, train_scaled, test_scaled


df = pd.read_csv('processed_2.csv', sep=',', header=None, encoding='gbk')

print(df.head())
df = df.assign(diff=df[2].values - df[1].values)
df.drop([1, 2], axis=1, inplace=True)

groups = df.groupby([0])

for mid_class, group in groups:
    print('Predicting {} -----------------'.format(mid_class))
    period = len(group)
    train, test = group['diff'].values[0: 2 * period // 3], \
                  group['diff'].values[2 * period // 3:]
    scaler, train, test = scale(train, test)
    X, y = train[:-1], train[1:]
    X = X.reshape(len(X), 1, 1)
    y = y.reshape(len(y), 1)
    X_test, y_test = test[:-1], test[1:]
    X_test = X_test.reshape(len(X_test), 1, 1)
    y_test = y_test.reshape(len(y_test), 1)

    print('X shape: {}, y shape: {}'.format(X.shape, y.shape))

    batch_size = 1

    model = Sequential()
    model.add(LSTM(4, activation='relu',
                   batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
                   stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    nb_epochs = 1000

    for i in range(nb_epochs):
        model.fit(X, y, epochs=1,
                  batch_size=batch_size,
                  verbose=1,
                  shuffle=False)
        model.reset_states()

    model.predict(X)

    for i in range(len(test)):
        yhat = model.predict(X_test, batch_size=1)
        print('Predicted: {}, Expected: {}'.format(yhat, y_test[i]))

    break
