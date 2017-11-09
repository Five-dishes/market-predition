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
from hmmlearn.hmm import GaussianHMM
from weighted_regression import WeightRegression
from sep_regression import SepRegression
from arima import ARIMA_
from print_util import check_results
import time


def push(x, y):  # 将y插入x尾部，将x头部与y等长的序列丢弃
    push_len = len(y)
    assert len(x) >= push_len
    x[:-push_len] = x[push_len:]
    x[-push_len:] = y
    return x


def predict(model, matrix, predict_len=30, is_time_related=False,
            history_len=28, predict_one_week=False):
    X, y = matrix[:, :-1], matrix[:, -1]
    feature_list = []

    if predict_one_week:
        period_to_predict = 7
    else:
        period_to_predict = predict_len

    if not is_time_related:
        model.fit(X, y)
        y_pred = deepcopy(y[- max(history_len, predict_len):])
        for i in range(0, period_to_predict):
            features = deepcopy(y_pred[-history_len:])
            feature_list.append(features)
            pred = model.predict(features.reshape(1, -1))
            push(y_pred, pred)

    else:
        y_pred = deepcopy(np.concatenate((matrix[0][:-1], matrix[:, -1])))
        for i in range(0, period_to_predict):
            model.fit(y_pred)
            features = deepcopy(y_pred)
            feature_list.append(features)
            pred = model.predict(features.reshape(1, -1))
            push(y_pred, pred)

    if predict_one_week:
        predicted = np.tile((y_pred[-7:]), [5])
        y_pred = predicted[:predict_len]

    y_pred = y_pred.clip(min=0.0)
    return y_pred[-predict_len:], feature_list


def evaluate_on(model, train, test, is_time_related):
    X_test, y_test = test[:, :-1], test[:, -1]
    y_pred, feature_list  = predict(model, train,
                                    predict_len=len(y_test),
                                    is_time_related=is_time_related,
                                    predict_one_week=True)

    mse = ((y_pred - y_test) ** 2).mean()
    print('MSE \t\t {:4.2f}'.format(mse))
    # print(y_test)
    # print(y_pred)
    show_sequence = False
    if show_sequence and mse > 30:
        check_results(feature_list, y_pred, y_test)
    return mse


if __name__ == '__main__':
    all_classes = pd.read_csv(
        'processed_2.csv', header=None, sep=',', encoding='gbk')

    appeared_mid_class = all_classes[0].unique()  # 在训练数据出现过的中类
    groups = all_classes.groupby([0])  # 按中类分组

    # 模型表
    models = {
        'sep regression': SepRegression(smooth=False),
        'Weighted Regression': WeightRegression(smooth=False),
        'Weekday Average': NaiveRegression(),
        'Last Week Average': WeekRegression(),
        'Smooth Weighted Regression': WeightRegression(smooth=True),
        'Mode': Mode(),
        'Linear Regression': LinearRegression(),
        'KNN 3': KNeighborsRegressor(n_neighbors=3),
        # 'Adaboost LR': AdaBoostRegressor(
        #     base_estimator=LinearRegression(), loss='linear',
        # ),
        #     # n_estimators=10, learning_rate=0.1,),
        #
        # 'Adaboost DTR': AdaBoostRegressor(
        #     loss='linear',
        # ),
        #
        # 'Adaboost SVR': AdaBoostRegressor(
        #     base_estimator=SVR(), loss='linear',
        # ),
        #     # n_estimators=10, learning_rate=0.1,),
        #
        # 'Random Forest': RandomForestRegressor(
        #     max_features=4, max_depth=3),
        #
        # 'GBR': GradientBoostingRegressor(
        #     # n_estimators=100, learning_rate=0.1,
        #     max_depth=1, loss='ls'),
        'SVR': SVR(),
        # 'ARIMA': ARIMA_((14, 0, 1)),
        # 'GaussianHMM': GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
    }

    def is_time_related_model(mod):  # 时间序列模型与普通模型的输入格式不同
        time_related_models = [
            'ARIMA',
            'sep regression',
        ]
        if mod in time_related_models:
            return True
        return False

    large_class_dict = {}
    template = pd.read_csv('template.csv', sep=',', header=0, encoding='gbk')
    class_template = template['编码'].unique()  # 读入要求预测的中类和大类
    mid_class_template = class_template[class_template >= 100]  # 去除大类
    large_class_template = class_template[class_template < 100]  # 去除中类

    model_usage_count = {}
    for model in models:
        model_usage_count[model] = 0

    for large_class in large_class_template:
        for date in range(20150501, 20150531):
            large_class_dict[(large_class, date)] = 0  # 因为需要索引，大类用dict，key是二元tuple

    mid_class_record = []

    for mid_class in mid_class_template:
        start_time = time.clock()
        if mid_class not in appeared_mid_class:  # 迷之预测，初始化为0，可能可以根据大类预测。。
            for date in range(20150501, 20150531):
                mid_class_record.append((mid_class, date, 0))  # 中类record用三元tuple
            continue

        group = groups.get_group(mid_class)
        large_class = mid_class // 100

        print('Current mid-class: {} {}'.format(mid_class, '-'*80))
        matrix = group.drop([0], axis=1).values.astype(np.float32)  # 扔掉中类标签、转化为浮点数
        spliter = int(len(group) - min(28, int(len(group) * 0.33)))  # 划分训练集与验证集
        train, validation = matrix[0: spliter], matrix[spliter:]

        mse_dict = {}
        for name in models:  # type=str
            print(name.ljust(20), ':', end='')
            model = models[name]
            # 在验证集上测试
            mse_dict[name] = evaluate_on(model, train, validation,
                                         is_time_related_model(name))
        # 找出在验证集上工作最好的模型
        best_model = min(mse_dict.items(), key=operator.itemgetter(1))[0]
        model_usage_count[best_model] += 1
        print('Best model: {}'.format(best_model))

        # 用验证集上的最优模型来预测（比较naive的方式）
        results, no_use = predict(models[best_model], matrix,
                                  is_time_related=is_time_related_model(best_model),
                                  predict_one_week=True)
        arima_error = False
        for value in results:
            if value > 999:
                arima_error = True
        if arima_error:
            assert best_model == 'ARIMA'
            mse_dict.pop(best_model, None)
            seconde_model = min(mse_dict.items(), key=operator.itemgetter(1))[0]
            print('ARIMA Error! Fall back to second best model {}'.format(seconde_model))
            results, no_use = predict(models[seconde_model], matrix,
                                      is_time_related=is_time_related_model(seconde_model),
                                      predict_one_week=True)
            model_usage_count[best_model] -= 1
            model_usage_count[seconde_model] += 1
        print(matrix[:, -1])
        print(results)

        date = 20150501
        for result in results:
            large_class_dict[(large_class, date)] += result  # 大类预测 = 中类之和
            mid_class_record.append((mid_class, date, result))  # append到中类record中
            date += 1

        print('Predicting class {} takes {}s'.format(mid_class, time.clock()-start_time))

    mid_class_df = pd.DataFrame.from_records(mid_class_record)
    large_class_tuple = [(*k, v) for k, v in large_class_dict.items()]
    large_class_df = pd.DataFrame.from_records(large_class_tuple)
    out = mid_class_df.append(large_class_df)  # 拼接中类和大类的data frame
    out.columns = ['编码', '日期', '销量']

    out.to_csv('results.csv', sep=',', index=None, encoding='gbk')
    print(model_usage_count)

