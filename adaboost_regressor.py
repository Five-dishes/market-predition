import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

all_classes = pd.read_csv('processed_2.csv',
# all_classes = pd.read_csv('special.csv',
        header=None, sep=',', encoding='gbk')

groups = all_classes.groupby([0])
print('# mid-class: {}'.format(len(groups)))

# scale train and test data to [-1, 1]
def scale(train: np.array, test: np.array):
    # fit scaler
    scaler = RobustScaler(with_scaling=False, with_centering=False)
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


def select_feature(x):
    short_term_features = [-4, -3, -2, -1]
    # short_term_features = []
    long_term_features = [-28, -21, -14, -7]
    selected = x[short_term_features + long_term_features]
    return selected


def perform_on(model, train, test, scaler):
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test= test[:, :-1], test[:, -1]

    model.fit(X_train, y_train)

    y_pred = y_train[-30:]
    feature_list = []

    # 使用预测出的序列进行下一轮预测
    for i in range(0, len(X_test)):
        # print('Predicting with:', y_pred)
        # print('Expecting:', np.concatenate((X_test[i], y_test[i].reshape(1))))
        features = select_feature(y_pred)
        feature_list.append(features)
        pred = model.predict(features.reshape(1, -1))
        push(y_pred, pred)
        # print('Got:', y_pred)

    y_pred = y_pred[-len(y_test):]

    predicted_matrix = np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1)
    test_matrix = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
    pred_origin = scaler.inverse_transform(predicted_matrix)[:, -1]
    test_origin = scaler.inverse_transform(test_matrix)[:, -1]

    print('Predicted\t\tExpected')
    for x, a, b in zip(feature_list, pred_origin, test_origin):
        print('[', end='')
        for f in x:
            print('{:4.2f}\t'.format(f), end='')
        print(']', end='')
        print('\t{:4.2f}\t\t{:4.2f}'.format(a, b))

    mse = ((pred_origin - test_origin) ** 2).mean()
    print('MSE: {}'.format(mse))


for mid_class, group in groups:
    print('Current mid-class: {}'.format(mid_class))

    # split date:
    group = group.drop([x for x in range(28)], axis=0)
    group = group.drop([0], axis=1)
    spliter = int(len(group) - min(30, len(group) * 0.2))
    matrix = group.values
    train, test = matrix[0: spliter], matrix[spliter:]
    scaler, train, test = scale(train, test)

    random_state = None

    models = [
            LinearRegression(),

            KNeighborsRegressor(n_neighbors=1),

            KNeighborsRegressor(n_neighbors=2),

            KNeighborsRegressor(n_neighbors=3),

            KNeighborsRegressor(n_neighbors=4),

            KNeighborsRegressor(n_neighbors=5),

            AdaBoostRegressor(
                base_estimator=LinearRegression(),
                n_estimators=10,
                learning_rate=0.1,
                random_state=random_state,
                loss='linear',
                ),

            AdaBoostRegressor(
                base_estimator=KNeighborsRegressor(n_neighbors=3),
                n_estimators=5,
                learning_rate=0.1,
                random_state=random_state,
                loss='linear',
                ),

            AdaBoostRegressor(
                n_estimators=50,
                learning_rate=0.1,
                random_state=random_state,
                loss='linear',
                ),

            '''
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
                max_depth=3,
                random_state=random_state,
                )
            '''
            ]
    names = [
            'Linear Regressor',
            'KNN 1',
            'KNN 2',
            'KNN 3',
            'KNN 4',
            'KNN 5',
            'AdaBoost Linear Regressor',
            'AdaBoost KNN 3',
            'AdaBoost-DTR',
            # 'AdaBoost-SVR',
            # 'GradientBoostingRegressor',
            # 'RandomForestRegressor',
            ]

    for name, model in zip(names, models):
        print(name)
        perform_on(model, train, test, scaler)

    break
