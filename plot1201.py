import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from pprint import pprint
from convolution import convolution
from convolution import log_smooth
from arima import ARIMA_


df = pd.read_csv('mid_vector.csv', sep=',', header=None, index_col=None)
groups = df.groupby([0])
colors = ['b', 'c', 'y', 'm', 'r', 'pink', 'green']
markers = ['.', 'o', 'x', '*', '+', '^', '_']

model = ARIMA_((14, 0, 1))

for class_name, df in groups:
    matrix = df.values[:, 2:]
    ts = np.concatenate((matrix[0][: -1], matrix[:, -1].reshape((-1))))
    if ts.mean() < 10:
        continue
    ts = log_smooth(ts)
    ts = convolution(ts)
    # for start_day in range(0, 7):
        # targets = []
        # for week_start in range(start_day, len(ts), 7):
        #     targets.append(ts[week_start: week_start + 7].mean())
        # print(class_name)
    targets = ts
    ax = plt.gca()
    # for i in range(0, 7):
    #     plt.scatter(x[x % 7 == i], ts[x % 7 == i], marker=markers[i], color=colors[i])
    train_index = np.arange(0, len(targets))
    train = list(targets)

    predictions = []
    for i in range(59):
        # pprint(train)
        model.fit(train)
        pred = model.predict(train_index)
        predictions.append(pred)
        train.append(pred)
        print('{} iteration--------------------------------------'.format(i))

    test_index = np.array(range(0, 59)) + len(train_index)

    plt.scatter(train_index, targets, marker=markers[0], color=colors[0])
    plt.scatter(test_index, predictions, marker=markers[1], color=colors[1])
    ax.set_title(str(class_name))
    # ax.set_xlim([0, 35])
    # ax.set_ylim([70, 140])
    # plt.legend(handles, models, scatterpoints=1)
    # plt.show()
    plt.savefig(pjoin('figure', str(class_name) + '.png'))
    plt.clf()

