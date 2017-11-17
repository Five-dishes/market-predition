import pandas as pd
import numpy as np
import operator
from copy import deepcopy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from naive_regression import NaiveRegression
from hmmlearn.hmm import GaussianHMM
from weighted_regression import WeightRegression
from sep_regression import SepRegression
from arima import ARIMA_
from print_util import check_results
from pprint import pprint
import time
from xgboost_predict import XGboost


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
    y_pred, feature_list = predict(model, train,
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
    return mse, y_pred


# 模型表
models = {
    'sep regression': SepRegression(smooth=False),
    # 'Weighted Regression': WeightRegression(smooth=False),
    # 'Weekday Average': NaiveRegression(),
    # 'Last Week Average': WeekRegression(),
    # 'Smooth Weighted Regression': WeightRegression(smooth=True),
    # 'Linear Regression': LinearRegression(),
    # 'KNN 3': KNeighborsRegressor(n_neighbors=3),
    # 'SVR': SVR(),
    # 'ARIMA7': ARIMA_((7, 0, 1)),
    # 'ARIMA14': ARIMA_((14, 0, 1)),
    'XGboost': XGboost(),
    # 'naive_select': NaiveSelect(),
    # 'naive_selectWeight': NaiveSelectWeight(),
}


def is_time_related_model(mod):  # 时间序列模型与普通模型的输入格式不同
    time_related_models = [
        'ARIMA7',
        'ARIMA14',
        'sep regression',
    ]
    if mod in time_related_models:
        return True
    return False


def full_service(matrix: np.array):
    # print(len(mid_group))
    # spliter = int(len(matrix) - min(7, int(len(matrix) * 0.33)))  # validation set 设为7
    spliter = int(len(matrix) - 7)  # 划分训练集与验证集
    train_set, validation_set = matrix[0: spliter], matrix[spliter:]

    mse_dict = {}
    prediction_on_validation_dict = {}
    for name in models:  # type=str
        if (name == 'ARIMA7' or name == 'ARIMA14') and matrix[:, -1].mean() < 8:
            continue
        print(name.ljust(30), ':', end='')
        model = models[name]
        # 在验证集上测试
        mse_dict[name], prediction_on_validation = evaluate_on(
            model, train_set, validation_set, is_time_related_model(name))
        prediction_on_validation_dict[name] = prediction_on_validation
    # 找出在验证集上工作最好的模型
    best_model_on_validation = min(mse_dict.items(), key=operator.itemgetter(1))[0]
    best_prediction_on_validation = prediction_on_validation_dict[best_model_on_validation]

    print('Best model: {}'.format(best_model_on_validation))

    # 用验证集上的最优模型来预测（比较naive的方式）
    prediction_on_test, no_use = predict(
        models[best_model_on_validation], matrix,
        is_time_related=is_time_related_model(best_model_on_validation),
        predict_one_week=True)

    return best_model_on_validation, best_prediction_on_validation, \
            prediction_on_test, mse_dict[best_model_on_validation], validation_set[:, -1]


if __name__ == '__main__':
    all_classes_small = pd.read_csv(
        'small_vector.csv', header=None, sep=',', encoding='gbk')
    all_classes_mid = pd.read_csv(
        'mid_vector.csv', header=None, sep=',', encoding='gbk')

    appeared_mid_class = all_classes_small[0].unique()  # 在训练数据出现过的中类
    mid_class_groups_small = all_classes_small.groupby([0])  # 按中类分组
    mid_class_groups_mid = all_classes_mid.groupby([0])  # 按中类分组

    large_class_dict = {}
    template = pd.read_csv('template.csv', sep=',', header=0, encoding='gbk')
    class_template = template['编码'].unique()  # 读入要求预测的中类和大类
    mid_class_template = class_template[class_template >= 100]  # 去除大类
    large_class_template = class_template[class_template < 100]  # 去除中类

    model_usage_count = {}
    for model_name in models:
        model_usage_count[model_name] = 0

    class_mse_dict = {}

    for large_class in large_class_template:
        for date in range(20150501, 20150531):
            large_class_dict[(large_class, date)] = 0  # 因为需要索引，大类用dict，key是二元tuple

    mid_class_record = []

    for mid_class in mid_class_template:
        start_time = time.time()
        if mid_class not in appeared_mid_class:  # 迷之预测，初始化为0，可能可以根据大类预测。。
            for date in range(20150501, 20150531):
                mid_class_record.append((mid_class, date, 0))  # 中类record用三元tuple
            continue

        group = mid_class_groups_small.get_group(mid_class)
        mid_group = mid_class_groups_mid.get_group(mid_class)
        large_class = mid_class // 100
        small_groups = group.groupby([1])
        accumulated_pred_on_test_small = np.zeros((30,), dtype=np.float32)

        # 中类直接预测
        mid_group = mid_group.drop([0], axis=1)  # 扔掉中类标签
        matrix_mid = mid_group.values.astype(np.float32)  # 转化为浮点数

        if matrix_mid[:, -1].mean() < 10:
            continue

        print('Current mid-class: {} {}'.format(mid_class, '=' * 80))

        best_validated_model_mid, best_validated_prediction_mid, \
            pred_on_test_mid, mse_mid, used_valid_set_mid = \
            full_service(matrix_mid)

        overall_validated_pred_small = np.zeros_like(used_valid_set_mid)
        accumulated_valid_set_small = np.zeros_like(best_validated_prediction_mid)

        for small_class, small_group in small_groups:  # 每个中类下 训练所有小类
            print('Current small-class: {} {}'.format(small_class, '-' * 80))
            small_group = small_group.drop([0], axis=1)  # 扔掉中类标签
            matrix_small = small_group.drop([1], axis=1).values.astype(np.float32)  # 扔掉小类标签、转化为浮点数

            best_validated_model_small, best_validated_pred_small, \
                pred_on_test_small, best_mse_small, used_valid_set_small = \
                full_service(matrix_small)

            overall_validated_pred_small += best_validated_pred_small
            accumulated_valid_set_small += used_valid_set_small
            accumulated_pred_on_test_small += pred_on_test_small

        # print("small class accumulated validation:", used_valid_set_mid)
        # print("mid class accumulated validation:", accumulated_valid_set_small)
        assert used_valid_set_mid.all() == accumulated_valid_set_small.all()

        mse_small = np.mean((overall_validated_pred_small - accumulated_valid_set_small) ** 2)
        print("## MSE on mid class {} by mid class prediction: {}".format(
            mid_class, mse_mid))
        print("## MSE on mid class {} by small class accumulation prediction: {}".format(
            mid_class, mse_small))

        date = 20150501
        # print(small_class_results)
        # print(mid_class_results)
        if mse_mid > mse_small:
            class_mse_dict[mid_class] = mse_small
            chosen_results = accumulated_pred_on_test_small
            print("## Result by small class accumulation prediction is chosen")
        else:
            class_mse_dict[mid_class] = mse_mid
            chosen_results = pred_on_test_mid
            print("## Result by mid class prediction is chosen")

        print(matrix_mid[:, -1])
        print(chosen_results)

        for result in chosen_results:
            large_class_dict[(large_class, date)] += result  # 大类预测 = 中类之和
            mid_class_record.append((mid_class, date, result))  # append到中类record中
            date += 1

        print('Predicting mid class {} takes {}s'.format(mid_class, time.time() - start_time))

    mid_class_df = pd.DataFrame.from_records(mid_class_record)
    large_class_tuple = [(*k, v) for k, v in large_class_dict.items()]
    large_class_df = pd.DataFrame.from_records(large_class_tuple)
    out = mid_class_df.append(large_class_df)  # 拼接中类和大类的data frame
    out.columns = ['编码', '日期', '销量']

    out.to_csv('results.csv', sep=',', index=None, encoding='gbk')
    pprint(class_mse_dict)
