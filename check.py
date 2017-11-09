import pandas as pd

df = pd.read_csv('results.csv', sep=',', header=0, encoding='gbk')

for index, row in df.iterrows():
    if float(row['销量']) > 999:
        print(row['销量'])
        print(index)
        print(row[['编码', '日期']])
