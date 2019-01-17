import numpy as np


def prepare_shape(feature, config):
    tmp = feature
    N = config.shape[1]
    while tmp.shape[1] < N:
        tmp = np.hstack((tmp, tmp))
    r_offset = np.random.randint(tmp.shape[1] - N + 1)
    tmp = tmp[:, r_offset: r_offset + N]
    return tmp
