from os.path import join
import codecs
import json
import pandas as pd
from collections import defaultdict as dd

df = pd.read_csv('processed_1.csv', sep=',', header=0, encoding='gbk')
groups = df.groupby(['中类编码'])

difficult_classes = []
for mid_class, group in groups:
    cur_sale_logs = {}
    # print(mid_class)
    for i, row in group.iterrows():
        cur_sale_logs[row['销售日期']] = row['销售数量']
    mean_sales = sum(cur_sale_logs.values())/len(cur_sale_logs.keys())
    if mean_sales > 10:
        print('{}: {}'.format(mid_class, mean_sales))
        difficult_classes.append(mid_class)

print(difficult_classes)
# difficult_classes = [1201, 1203, 1505, 2201, 2202, 2203]
