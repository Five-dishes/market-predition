# -*- coding: UTF-8 -*-

"""
Feature selection:

Input file: processed_1.csv
Headers in processed_1.csv: 大类编码	中类编码	销售日期
day of week 	day of week 2

Output file: preocessed_2.csv
Every row is corresponding to each mid-class item in every day
(from 1st Jan. to 30th April).
The last item in each row is # of unique orders of a mid-class item on that day.
For prediction, 18 features are selected: 14 of them are # of unique orders
in past 2 weeks;
While other 4 are # of unique orders on the same day of week in past 4 weeks.
"""

import pandas as pd
import numpy as np
import datetime
import sys


num_past_days = 4
num_past_weeks = 4
start_date = datetime.datetime.strptime('20150101', '%Y%m%d')


def get_feature_date_in_int(day: datetime.datetime) -> [int]:
    """

    :param day
    :return: list of dates of past 2 weeks and the corresponding days in past 4 weeks
    """
    past_days = [day - datetime.timedelta(x) for x in range(
        num_past_days, 0, -1)]
    past_weekdays = [day - datetime.timedelta(x*7) for x in range(
        num_past_weeks, 0, -1)]
    dates = past_days + past_weekdays
    dates.append(day)
    dates_int = [np.int64(date.strftime("%Y%m%d")) for date in dates]
    return dates_int


def get_features(df: pd.DataFrame, day: datetime.datetime) -> pd.Series:
    """

    :param df: the Dataframe to grab # of sold items
    :param day: the date to generate example
    :return: one example

    输入是某个(中类)商品的销售数量历史 和 一个日期
    返回这个日期 过去2周的销售数量、过去4周对应星期X的销售数量 和 当天的销售数量
    """
    # df2 = df.drop(['中类编码'], axis=1, inplace=False)
    d = df.set_index('销售日期').T.to_dict()
    feature_dates = get_feature_date_in_int(day)

    features = []
    mid_class = df.iloc[0]['中类编码']
    features.append(mid_class)
    for date in feature_dates:
        if date in d:
            features.append(d[date]['销售数量'])
        else:
            features.append(np.int64(0))
    return pd.Series(features)


df = pd.read_csv('processed_1.csv', sep=',', header=0, encoding='gbk')
df.drop('大类编码', axis=1, inplace=True)
groups = df.groupby(['中类编码'])
matrix = []

for code, df in groups:  # 每一个group是一个中类
    for period in range(31 + 28 + 31 + 30):  # 4个月
        matrix.append(get_features(df, start_date + datetime.timedelta(period)))
    break

out = pd.concat(matrix, axis=1)
out = out.transpose()
# print(out)
out.to_csv('processed_2.csv', sep=',', header=None, index=None)
