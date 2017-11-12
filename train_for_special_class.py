from os.path import join
import codecs
import json
import pandas as pd
from date_util import DateUtil
import datetime
from arima import ARIMA_
from sklearn.metrics import mean_squared_error
import numpy as np



important_classes = [1201, 1203, 1505, 2201, 2202, 2203]

df = pd.read_csv('processed_1.csv', sep=',', header=0, encoding='gbk')
groups = df.groupby(['中类编码'])

pred_start_date = 20150501
steps = 7


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def set_start_date():
    train_start_date_list = []
    train_dates = df['销售日期'].unique()
    for d in train_dates:
        offset = (DateUtil.int_to_date(pred_start_date) - DateUtil.int_to_date(d)).days
        if offset % 7 == 0 and offset > 28:
            print(d)
            train_start_date_list.append(d)
    return train_start_date_list


def make_train_data():
    for mid_class, group in groups:
        model = ARIMA_((7, 0, 1))
        if mid_class not in important_classes:
            continue
        cur_sale_logs = {}
        # print(mid_class)
        for i, row in group.iterrows():
            cur_sale_logs[row['销售日期']] = float(row['销售数量'])
        arr = [0] * (DateUtil.int_to_date(20150501) - DateUtil.int_to_date(20150101)).days
        for d in cur_sale_logs:
            cur_sale_num = cur_sale_logs[d]
            idx = (DateUtil.int_to_date(d) - DateUtil.int_to_date(20150101)).days
            arr[idx] = cur_sale_num
        differenced = difference(arr[: -7], steps)
        # print(len(differenced), differenced)

        # model = ARIMA(differenced, order=(7, 0, 1))
        model_fit = model.fit(differenced)
        # multi-step out-of-sample forecast
        forecast = model_fit.forecast(steps=7)[0]
        # invert the differenced forecast to something usable
        # print(forecast)
        # invert the differenced forecast to something usable
        history = [x for x in arr[: -7]]
        day = 1
        pred_list = []
        true_list = arr[-7:]
        for yhat in forecast:
            inverted = inverse_difference(history, yhat, steps)
            # print('Day %d: %f' % (day, inverted))
            history.append(inverted)
            pred_list.append(inverted)
            day += 1
        print('pred', pred_list)
        print('true', true_list)
        err = mean_squared_error(true_list, pred_list)
        print('class', mid_class, 'err', err)
        # break


def make_train_data_old(train_start_date_list):
    for mid_class, group in groups:
        model = ARIMA_((7, 0, 1))
        if mid_class not in important_classes:
            continue
        cur_sale_logs = {}
        # print(mid_class)
        for i, row in group.iterrows():
            cur_sale_logs[row['销售日期']] = row['销售数量']
        # print(cur_sale_logs)
        cur_class_X = []
        cur_class_Y = []
        cur_class_out = []
        for start_d in train_start_date_list:
            x = []
            y = []
            start_dt = DateUtil.int_to_date(start_d)
            for i in range(28):
                cur_dt = start_dt + datetime.timedelta(days=i)
                cur_d = DateUtil.date_to_int(cur_dt)
                cur_sale_num = cur_sale_logs.get(cur_d, 0)
                x.append(float(cur_sale_num))
                # print(cur_d)
            for i in range(28, 35):
                cur_dt = start_dt + datetime.timedelta(days=i)
                cur_d = DateUtil.date_to_int(cur_dt)
                cur_sale_num = cur_sale_logs.get(cur_d, 0)
                y.append(float(cur_sale_num))
            # cur_class_X.append(x)
            # cur_class_Y.append(y)
            model.fit(x)
            # print(x)
            out = model.predict_batch()
            # print(out.shape)
            if len(out) == 7:
                print('pred', out)
                print('true', y)
                cur_class_Y.extend(y)
                cur_class_out.extend(out)
        if len(cur_class_out):
            err = mean_squared_error(cur_class_Y, cur_class_out)
            print(mid_class, 'err', err)


if __name__ == '__main__':
    train_start_date_list = set_start_date()
    print(len(train_start_date_list))
    make_train_data()


