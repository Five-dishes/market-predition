import pandas as pd
import numpy as np
import operator
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from naive_regression import NaiveRegression
from week_regression import WeekRegression
from mode_regression import Mode


def select_feature(x):
    short_term_features = [-4, -3, -2, -1]
    # short_term_features = []
    long_term_features = [-28, -21, -14, -7]
    selected = x[short_term_features + long_term_features]
    return selected


def push(x, y):
    push_len = len(y)
    assert len(x) >= push_len
    x[:-push_len] = x[push_len:]
    x[-push_len:] = y
    return x


def predict(model, matrix, predict_len=30):
    X, y = matrix[:, :-1], matrix[:, -1]
    model.fit(X, y)
    y_pred = deepcopy(y[-predict_len:])
    feature_list = []
    for i in range(0, predict_len):
        features = deepcopy(y_pred[-28:])
        feature_list.append(features)
        pred = model.predict(features.reshape(1, -1))
        push(y_pred, pred)
    return y_pred


def evaluate_on(model, train, test):
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test= test[:, :-1], test[:, -1]

    model.fit(X_train, y_train)

    y_pred = deepcopy(y_train[-28:])
    feature_list = []

    # 使用预测出的序列进行下一轮预测
    for i in range(0, len(X_test)):
        # features = select_feature(y_pred)
        features = deepcopy(y_pred)
        feature_list.append(features)
        pred = model.predict(features.reshape(1, -1))
        push(y_pred, pred)

    y_pred = y_pred[-len(y_test):]

    mse = ((y_pred - y_test) ** 2).mean()
    print('MSE \t\t {:4.2f}'.format(mse))
    show_sequence = False

    if show_sequence and mse > 30:
        show_feature = False
        print('Predicted\t\tExpected')
        for x, a, b in zip(feature_list, y_pred, y_test):
            if show_feature:
                print('[', end='')
                for f in x:
                    print('{:4.2f}\t'.format(f), end='')
                print(']', end='')
            print('\t{:4.2f}\t\t{:4.2f}'.format(a, b))
    return mse


if __name__ == '__main__':
    all_classes = pd.read_csv(
        'processed_2.csv', header=None, sep=',', encoding='gbk')

    appeared_mid_class = all_classes[0].unique()

    groups = all_classes.groupby([0])

    models = {
        'Week Average': WeekRegression(),
        'Mode': Mode(),
        'Naive Baseline': NaiveRegression(),
        'Linear Regression': LinearRegression(),
        'KNN 3': KNeighborsRegressor(n_neighbors=3),
        'Adaboost LR': AdaBoostRegressor(
            base_estimator=LinearRegression(), loss='linear',
        ),
            # n_estimators=10, learning_rate=0.1,),

        'Adaboost DTR': AdaBoostRegressor(
            loss='linear',
        ),
            # n_estimators=10, learning_rate=0.1,),

        'Adaboost SVR': AdaBoostRegressor(
            base_estimator=SVR(), loss='linear',
        ),
            # n_estimators=10, learning_rate=0.1,),

        'Random Forest': RandomForestRegressor(
            max_features=4, max_depth=3),

        'GBR': GradientBoostingRegressor(
            # n_estimators=100, learning_rate=0.1,
            max_depth=1, loss='ls'),
        'SVR': SVR(),
    }

    large_class_dict = {}
    template = pd.read_csv('template.csv', sep=',', header=0, encoding='gbk')
    mid_class_template = template['编码'].unique()
    for mid_class in mid_class_template:
        if mid_class < 100:
            continue
        large_class = mid_class // 100
        for date in range(20150501, 20150531):
            large_class_dict[(large_class, date)] = 0

    mid_class_record = []

    for mid_class in mid_class_template:
        if mid_class < 100:
            continue
        if mid_class not in appeared_mid_class:
            for date in range(20150501, 20150531):
                mid_class_record.append((mid_class, date, 0))
            continue

        group = groups.get_group(mid_class)
        large_class = mid_class // 100
        print('Current mid-class: {} ------------------'.format(mid_class))
        matrix = group.drop([0], axis=1).values
        matrix = matrix.astype(np.float32)

        '''
        scaler = RobustScaler(with_scaling=True, with_centering=False)
        scaler = scaler.fit(matrix)
        matrix = scaler.transform(matrix)
        '''

        spliter = int(len(group) - min(28, int(len(group) * 0.2)))
        train, test = matrix[0: spliter], matrix[spliter:]

        mse_dict = {}
        for name in models:  # type=str
            print(name.ljust(20), ':', end='')
            model = models[name]
            mse_dict[name] = evaluate_on(model, train, test)
        best_model = min(mse_dict.items(), key=operator.itemgetter(1))[0]
        print('Best model: {}'.format(best_model))
        results = predict(models[best_model], matrix)
        print(matrix[:, -1])
        print(results)

        date = 20150501
        for result in results:
            large_class_dict[(large_class, date)] += result
            mid_class_record.append((mid_class, date, result))
            date += 1

    mid_class_df = pd.DataFrame.from_records(mid_class_record)
    large_class_tuple = [(*k, v) for k, v in large_class_dict.items()]
    large_class_df = pd.DataFrame.from_records(large_class_tuple)
    out = mid_class_df.append(large_class_df)
    out.columns = ['编码', '日期', '销量']

    out.to_csv('results.csv', sep=',', index=None, encoding='gbk')

