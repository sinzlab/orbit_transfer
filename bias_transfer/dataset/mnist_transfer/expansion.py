import numpy as np


def apply_expansion(source, target):
    orig_shape = source.shape
    expanded_batch = np.zeros((orig_shape[0], 1, 40, 40))
    expanded_batch[:, :, 6:-6, 6:-6] = source
    return expanded_batch, target