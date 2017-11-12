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
#from hmmlearn.hmm import GaussianHMM
#from arima import ARIMA_
from print_util import check_results
from xgboost_predict import xgboost_
from naive_select import  NaiveSelect
from naive_select import  NaiveSelectWeight


def select_feature(x):
    short_term_features = [-4, -3, -2, -1]
    long_term_features = [-28, -21, -14, -7]
    selected = x[short_term_features + long_term_features]
    return selected


def push(x, y):  # 将y插入x尾部，将x头部与y等长的序列丢弃
    push_len = len(y)
    assert len(x) >= push_len
    x[:-push_len] = x[push_len:]
    x[-push_len:] = y
    return x


def predict(model, matrix, predict_len=30, is_time_related=False,
            history_len=28):
    X, y = matrix[:, :-1], matrix[:, -1]
    feature_list = []
    if not is_time_related:
        model.fit(X, y)
        y_pred = deepcopy(y[- max(history_len, predict_len):])
        for i in range(0, predict_len):
            features = deepcopy(y_pred[-history_len:])
            feature_list.append(features)
            pred = model.predict(features.reshape(1, -1))
            push(y_pred, pred)

    else:
        y_pred = deepcopy(np.concatenate((matrix[0][:-1], matrix[:, -1])))
        for i in range(0, predict_len):
            model.fit(y_pred)
            features = deepcopy(y_pred)
            feature_list.append(features)
            pred = model.predict(features.reshape(1, -1))
            push(y_pred, pred)

    y_pred = y_pred.clip(min=0.0)
    return y_pred[-predict_len:], feature_list


def evaluate_on(model, train, test, is_time_related):
    X_test, y_test = test[:, :-1], test[:, -1]
    y_pred, feature_list  = predict(model, train,
                     predict_len=len(y_test),
                     is_time_related=is_time_related)

    mse = ((y_pred - y_test) ** 2).mean()
    print('MSE \t\t {:4.2f}'.format(mse))
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
        'xgboost': xgboost_(),
        'naive_select' : NaiveSelect(),
        'naive_selectWeight' : NaiveSelectWeight(),

        # 'Weekday Average': NaiveRegression(),
        # 'Last Week Average': WeekRegression(),
        # 'Mode': Mode(),
        # 'Linear Regression': LinearRegression(),
        # 'KNN 3': KNeighborsRegressor(n_neighbors=3),
        # 'Adaboost LR': AdaBoostRegressor(
        #     base_estimator=LinearRegression(), loss='linear',
        # ),
        #     # n_estimators=10, learning_rate=0.1,),
        #
        # 'Adaboost DTR': AdaBoostRegressor(
        #     loss='linear',
        # ),
        #     # n_estimators=10, learning_rate=0.1,),
        #
        # 'Adaboost SVR': AdaBoostRegressor(
        #     base_estimator=SVR(), loss='linear',
        # ),
        #     # n_estimators=10, learning_rate=0.1,),
        #
        # # 'Random Forest': RandomForestRegressor(
        # #     max_features=4, max_depth=3),

        # 'GBR': GradientBoostingRegressor(
        #     # n_estimators=100, learning_rate=0.1,
        #     max_depth=1, loss='ls'),
        'SVR': SVR(),
        # 'ARIMA 1, 0, 1': ARIMA_((1, 0, 1)),
        # 'ARIMA 0, 0, 1': ARIMA_((0, 0, 1)),
        # 'ARIMA 1, 0, 0': ARIMA_((1, 0, 0)),
        # 'ARIMA 1, 1, 1': ARIMA_((1, 1, 1)),
        # 'GaussianHMM': GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
    }

    def is_time_related_model(mod):  # 时间序列模型与普通模型的输入格式不同
        time_related_models = [
            'ARIMA 1, 0, 1',
            'ARIMA 0, 0, 1',
            'ARIMA 1, 0, 0',
            'ARIMA 1, 1, 1',
        ]
        if mod in time_related_models:
            return True
        return False

    large_class_dict = {}
    template = pd.read_csv('template.csv', sep=',', header=0, encoding='gbk')
    class_template = template['编码'].unique()  # 读入要求预测的中类和大类
    mid_class_template = class_template[class_template >= 100]  # 去除大类
    large_class_template = class_template[class_template < 100]  # 去除中类

    for large_class in large_class_template:
        for date in range(20150501, 20150531):
            large_class_dict[(large_class, date)] = 0  # 因为需要索引，大类用dict，key是二元tuple

    mid_class_record = []

    for mid_class in mid_class_template:
        if mid_class not in appeared_mid_class:  # 迷之预测，初始化为0，可能可以根据大类预测。。
            for date in range(20150501, 20150531):
                mid_class_record.append((mid_class, date, 0))  # 中类record用三元tuple
            continue

        group = groups.get_group(mid_class)
        large_class = mid_class // 100
        small_groups = group.groupby([1])
        results = np.zeros((30,), dtype=np.float32)
        for small_class, small_group in small_groups:    #每个中类下 训练所有小类
            print('Current small-class: {} {}'.format(small_class, '-'*80))
            small_group = small_group.drop([0], axis=1)      #扔掉中类标签
            matrix = small_group.drop([1], axis=1).values.astype(np.float32)  # 扔掉小类标签、转化为浮点数
            spliter = int(len(small_group) - min(28, int(len(small_group) * 0.33)))  # 划分训练集与验证集
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
            print('Best model: {}'.format(best_model))

            # 用验证集上的最优模型来预测（比较naive的方式）
            small_results, no_use = predict(models[best_model], matrix,
                                    is_time_related=is_time_related_model(best_model))
           # print(small_results)
            results = np.add(results, small_results)   #累加小类预测结果到中类

        date = 20150501
        for result in results:
            large_class_dict[(large_class, date)] += result  # 大类预测 = 中类之和
            mid_class_record.append((mid_class, date, result))  # append到中类record中
            date += 1

    mid_class_df = pd.DataFrame.from_records(mid_class_record)
    large_class_tuple = [(*k, v) for k, v in large_class_dict.items()]
    large_class_df = pd.DataFrame.from_records(large_class_tuple)
    out = mid_class_df.append(large_class_df)  # 拼接中类和大类的data frame
    out.columns = ['编码', '日期', '销量']

    out.to_csv('results.csv', sep=',', index=None, encoding='gbk')

