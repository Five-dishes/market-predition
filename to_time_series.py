import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import History
from copy import deepcopy


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
print('total mid-class: {}'.format(len(groups)))

batch_size = 1

model = Sequential()
model.add(LSTM(7, activation='relu',
               batch_input_shape=(batch_size, 1, 1),
               stateful=True))
# model.add(LSTM(4, return_sequences=True, activation='relu',
#                stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

epsilon = 0.0001
def all_01(a: np.array) -> bool:
    if np.all(np.abs(a - 0) > epsilon or np.abs(x - 1) > epsilon):
        return True
    return False

def predict(group: pd.DataFrame, regenerate_weights: bool = False):
    weights = model.get_weights()

    if not regenerate_weights:
        weights = [np.random.permutation(w) for w in weights]
    else:
        print('Regenerating random weights==========')
        weights_1 = []
        for w in weights:
            if all_01(w):
                weights_1.append(np.random.permutation(w))
            else:
                weights_1.append((np.random.rand(*w.shape) - 0.5) / 10)
        weights = weights_1

    model.set_weights(weights)

    period = len(group)
    spliter = 4 * period // 5
    train, test = group['diff'].values[0: spliter], \
                  group['diff'].values[spliter:]
    scaler, train, test = scale(train, test)
    X, y = train[:-1], train[1:]
    X = X.reshape(len(X), 1, 1)
    y = y.reshape(len(y), 1)
    X_test, y_test = test[:-1], test[1:]
    X_test = X_test.reshape(len(X_test), 1, 1)
    y_test = y_test.reshape(len(y_test), 1)

    print('X shape: {}, y shape: {}'.format(X.shape, y.shape))


    nb_epochs = 500

    train_loss = 0
    train_loss_malformed = 0

    for i in range(nb_epochs):
        hist = model.fit(X, y, epochs=1,
                  batch_size=batch_size,
                  verbose=1,
                  shuffle=False)
        model.reset_states()

        train_loss = hist.history['loss'][0]
        if train_loss < 0.52:
            break
        if train_loss > 10000:
            train_loss_malformed += 1
            if train_loss_malformed > 50:
                return False, None, None
        if np.isnan(train_loss):
            return False, None, None
        else:
            print(train_loss)

    model.predict(X, batch_size=1)

    score = model.evaluate(X_test, y_test, batch_size=1, verbose=1)
    print('\nMSE 1: {}'.format(score))
    return True, train_loss, score


for mid_class, group in groups:
    suc = False
    train_loss = -1
    score = -1

    while not suc:
        print('Predicting {} -----------------'.format(mid_class))
        suc, train_loss, score = predict(group, score != -1)

    out_str = '{}, {}, {}\n'.format(mid_class, score, train_loss)

    print('Appending class {} -----------------'.format(mid_class))
    with open('out.csv', 'a') as f:
        f.write(out_str)
        f.flush()
