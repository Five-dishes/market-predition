from os.path import join
import codecs
import json
import pandas as pd
from date_util import DateUtil
import matplotlib.pyplot as plt
from pylab import *


important_classes = [1201, 1203, 1505, 2201, 2202, 2203]

df = pd.read_csv('processed_1.csv', sep=',', header=0, encoding='gbk')
groups = df.groupby(['中类编码'])

for mid_class, group in groups:
    if mid_class not in important_classes:
        continue
    cur_sale_logs = {}
    # print(mid_class)
    for i, row in group.iterrows():
        cur_sale_logs[row['销售日期']] = row['销售数量']
    x = []
    y = []
    for k in cur_sale_logs:
        sale_num = cur_sale_logs[k]
        time_delta = (DateUtil.int_to_date(k) - DateUtil.int_to_date(20141231)).days
        x.append(time_delta)
        y.append(sale_num)
        # print(time_delta)
    print(len(x))

    plot(x, y)
    show()

