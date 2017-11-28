# -*- coding: UTF-8 -*-

"""
Count # of unique orders for each mid-class item per day.

Input file: data.csv;
Headers in data.csv: custid	大类编码	中类编码	销售日期	销售数量	是否促销;
Other columns has been deleted manually.

Output file: processed_1.csv
Headers in processed_1.csv: 大类编码	中类编码	销售日期	销售数量    	day of week 	day of week 2
Although last 2 columns will never be used...

 """

import pandas as pd
import datetime


group_indices = ['大类编码', '中类编码','小类编码', '销售日期']

template = pd.read_csv('template-semi.csv', sep=',', header=0, encoding='gbk')
mid_class = template['bianma'].unique()


def add_up(rows: pd.DataFrame) -> pd.Series:
    """

    :param rows: groupby function divide DataFrame into multiple DataFrames according to given
    headers. rows is one DataFrame, aka. one group.
    :return: reduced row

    This function count number of orders each mid-class per day.
    """
    ret = pd.Series()
    sales = len(rows)  # count unique orders instead of #sold
    ret['销售数量'] = sales
    # ret['custid'] = 0
    # day = datetime.datetime.strptime(str(ret['销售日期']), '%Y%m%d')
    # ret['day of week'] = datetime.datetime.weekday(day)
    # ret['day of week 2'] = day.strftime('%a')
    # ret.drop(group_indices, inplace=True)

    return ret


def df_reduce(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df: pd.DataFrame
    :return: pd.DataFrame

    This function roll up the data frame by custid and '销售数量';
    Define an one-line function for type hint
    """
    return df.groupby(group_indices).apply(add_up)


df = pd.read_csv('train-data-semi.csv', sep=',', header=0, encoding='gbk')
reduced = df_reduce(df)

for index in reduced.index:
    if index[1] not in mid_class:
        reduced.drop(index, inplace=True)

# reduced.drop('custid', axis=1, inplace=True)
# reduced.drop('是否促销', axis=1, inplace=True)
reduced.to_csv('processed_small.csv', sep=',', encoding='gbk')

