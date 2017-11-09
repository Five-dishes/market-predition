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
from date_util import DateUtil


du = DateUtil()
num_past_days = 28
num_past_weeks = 0
start_date_int = 20150101
end_date_int = 20150430


def make_dates() -> [np.int64]:
    date = end_date_int
    sequence = [date]
    while date > start_date_int:
        date = du.minus(1, date)
        sequence.append(date)
    sequence.reverse()
    return sequence


def get_sold(df: pd.DataFrame, dates: [np.int64]) -> [np.int64]:
    """
    :param df: the Dataframe to grab # of sold items
    :param dates: the dates to get # sold
    :return: one example
    输入是某个(中类)商品的销售数量历史 和 一个日期
    返回这个日期 过去2周的销售数量、过去4周对应星期X的销售数量 和 当天的销售数量
    """
    d = df.set_index('销售日期').T.to_dict()
    features = []
    for date in dates:
        if date in d:
            features.append(d[date]['销售数量'])
        else:
            features.append(np.int64(0))
    return features


def split_to_examples(sold_items: [np.int64]) -> [[np.int64]]:
    period_len = len(sold_items)
    sold_items = np.array(sold_items)
    ret = []
    for i in range(period_len - num_past_days):
        ret.append(sold_items[:num_past_days + 1])
        sold_items = np.roll(sold_items, -1)
    return ret


def day_of_week_smooth(x: [np.int64]):  # handle missing 31st...
    i = len(x) - 31  # 31st March
    left = i - 7
    right = i + 7
    x[i] = (x[left] + x[right]) // 2


if __name__ == '__main__':
    dates_all = make_dates()

    df = pd.read_csv('processed_1.csv', sep=',', header=0, encoding='gbk')
    # df.drop('大类编码', axis=1, inplace=True)
    groups = df.groupby(['中类编码'])
    matrix = []

    for mid_class, group in groups:  # 每一个group是一个中类
        large_class = group.iloc[0]['大类编码']
        sold_items_all = get_sold(group, dates_all)
        day_of_week_smooth(sold_items_all)
        examples = np.array(split_to_examples(sold_items_all))
        large_class_col = np.array([large_class] * len(examples)).reshape(len(examples), 1)
        mid_class_col = np.array([mid_class] * len(examples)).reshape(len(examples), 1)
        # examples = np.concatenate((large_class_col, mid_class_col, examples), axis=1)
        examples = np.concatenate((mid_class_col, examples), axis=1)
        matrix.append(examples)

    out = np.concatenate(matrix, axis=0)
    out = pd.DataFrame(out)
    out.to_csv('processed_2.csv', sep=',', header=None, index=None)

