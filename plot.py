import pandas as pd
import numpy as np
import matplotlib as plt


df = pd.read_csv('valid2.csv', sep=',', header=0, index_col=None)
groups = df.groupby([0])
for 


