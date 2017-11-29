import numpy as np


def convolution(x: np.ndarray, window_size: int = 7) -> np.ndarray:
    x = x.astype(np.float)
    start = window_size // 2
    end = len(x) - window_size // 2
    y = np.empty_like(x)
    for i in range(start, end):
        y[i] = np.mean(x[i - window_size//2: i + window_size//2 + 1])
    return y[start: end]


def log_smooth(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float) + 1.0
    return np.log2(x)
