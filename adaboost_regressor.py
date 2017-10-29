import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

all_classes = pd.read_csv('processed_2.csv',
        header=None, sep=',', encoding='gbk')

groups = all_classes.groupby([0])
print('# mid-class: {}'.format(len(groups)))

# scale train and test data to [-1, 1]
def scale(train: np.array, test: np.array):
    # fit scaler
    scaler = RobustScaler(with_scaling=True)
    scaler = scaler.fit(train)

    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


def push(x, y):
    push_len = len(y)
    assert len(x) >= push_len
    x[:-push_len] = x[push_len:]
    x[-push_len:] = y
    return x


def perform_on(model, train, test, scaler):
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test= test[:, :-1], test[:, -1]

    model.fit(X_train, y_train)

    series = y_train
    y_pred = X_train[-1]

    # 使用预测出的序列进行下一轮预测
    for i in range(0, len(X_test)):
        pred = model.predict(y_pred.reshape(1, -1))
        push(y_pred, pred)

    y_pred = y_pred[-len(y_test):]

    predicted_matrix = np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1)
    test_matrix = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    pred_origin = scaler.inverse_transform(predicted_matrix)[:, -1]
    test_origin = scaler.inverse_transform(test_matrix)[:, -1]

    '''
    print('Predicted\t\tExpected')
    for a, b in zip(pred_origin, test_origin):
        print ('{:10.2f}\t\t{:10.2f}'.format(a, b))
    '''

    mse = ((pred_origin - test_origin) ** 2).mean()
    print('MSE: {}'.format(mse))


for mid_class, group in groups:
    print('Current mid-class: {}'.format(mid_class))

    # split date:
    group = group.drop([x for x in range(30)], axis=0)
    group = group.drop([0], axis=1)
    spliter = int(len(group) * 0.8)
    matrix = group.values
    train, test = matrix[0: spliter], matrix[spliter:]
    scaler, train, test = scale(train, test)

    random_state = None

    models = [
            AdaBoostRegressor(
                n_estimators=10,
                learning_rate=0.1,
                random_state=random_state,
                loss='linear',
                ),
            AdaBoostRegressor(
                base_estimator=SVR(),
                n_estimators=20,
                learning_rate=0.4,
                loss='square',
                random_state=random_state,
                ),

            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=1,
                random_state=random_state,
                loss='ls',
                ),
            RandomForestRegressor(
                max_features=4,
                max_depth=2,
                random_state=random_state,
                )
            ]
    names = [
            'AdaBoost-DTR',
            'AdaBoost-SVR',
            'GradientBoostingRegressor',
            'RandomForestRegressor',
            ]

    for name, model in zip(names, models):
        print(name)
        perform_on(model, train, test, scaler)

    break
