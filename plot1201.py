import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from pprint import pprint


df = pd.read_csv('mid_vector.csv', sep=',', header=None, index_col=None)
groups = df.groupby([0])
colors = ['b', 'c', 'y', 'm', 'r', 'pink', 'green']
markers = ['.', 'x', 'o', '*', '+', '^', '_']

for class_name, df in groups:
    matrix = df.values[:, 2:]
    ts = np.concatenate((matrix[0][: -1], matrix[:, -1].reshape((-1))))
    if ts.mean() < 28:
        continue
    for start_day in range(0, 7):
        weeks = []
        for week_start in range(start_day, len(ts), 7):
            weeks.append(ts[week_start: week_start+7].mean())
        print(class_name)
        ax = plt.gca()
        # for i in range(0, 7):
        #     plt.scatter(x[x % 7 == i], ts[x % 7 == i], marker=markers[i], color=colors[i])
        x = np.arange(0, len(weeks))
        plt.scatter(x, weeks, marker=markers[0], color=colors[0])
        ax.set_title(str(class_name)+'_{}'.format(start_day))
        ax.set_xlim([0, 16])
        ax.set_ylim([8, 130])
        # plt.legend(handles, models, scatterpoints=1)
        # plt.show()
        plt.savefig(pjoin('figure', str(class_name) + '_{}.png'.format(start_day)))
        plt.clf()
    break

