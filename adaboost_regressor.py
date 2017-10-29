import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import RobustScaler

all_classes = pd.read_csv('processed_2.csv',
        header=None, sep=',', encoding='gbk')

groups = all_classes.groupby([0])
print('# mid-class: {}'.format(len(groups)))

# scale train and test data to [-1, 1]
def scale(train: np.array, test: np.array):
    # fit scaler
    scaler = RobustScaler(with_scaling=False)
    scaler = scaler.fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

for mid_class, group in groups:
    print('Current mid-class: {}'.format(mid_class))

    # split date:
    spliter = int(len(group) * 0.8)
    group = group.drop([0], axis=1)
    matrix = group.values
    train, test = matrix[0: spliter], matrix[spliter:]
    scaler, train, test = scale(train, test)

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test= test[:, :-1], test[:, -1]

    model = AdaBoostRegressor(loss='square')
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print('score: {}'.format(score))
    y_pred = model.predict(X_test)

    predicted_matrix = np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1)
    test_matrix = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    pred_origin = scaler.inverse_transform(predicted_matrix)[:, -1]
    test_origin = scaler.inverse_transform(test_matrix)[:, -1]

    print('Predicted\t\tExpected')
    for a, b in zip(pred_origin, test_origin):
        print ('{:10.2f}\t\t{:10.2f}'.format(a, b))

    break
