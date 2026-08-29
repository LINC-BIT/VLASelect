import numpy as np
from functools import reduce


def min_max_normalize(data):
    data = np.asarray(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def flatten_2d_arr(arr):
    # if not isinstance(arr[0], list):
    #     return arr
    res = []
    for n in arr:
        if isinstance(n, list):
            res += n 
        else:
            res += [n]
    return res


def smoothing(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed