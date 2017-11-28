import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
from pprint import pprint


df = pd.read_csv('valid2.csv', sep=',', header=0, index_col=None)
groups = df.groupby(['0'])
x = np.arange(0, 7)
colors = ['b', 'c', 'y', 'm', 'r']
markers = ['x', 'o', '*', '+', 't']

for class_name, df in groups:
    print(class_name)
    matrix = df.values[:, 1:]
    if matrix[:, 1:].mean() < 8:
        continue
    ax = plt.gca()
    index = 0
    handles = []
    models = []
    for row in matrix:
        model, y = row[0], row[1:]
        handle = plt.scatter(x, y, marker=markers[index], color=colors[index])
        handles.append(handle)
        models.append(model)
        index += 1
    ax.set_title(class_name)
    plt.legend(handles, models, scatterpoints=1)
    plt.savefig(pjoin('figure', str(class_name) + '.png'))
    plt.clf()


