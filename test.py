import pandas as pd
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

print(x_train.shape)
print(y_train.shape)
