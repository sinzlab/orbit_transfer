import numpy as np


def apply_label_shuffle(source, target):
    np.random.shuffle(target)  # to make this dataset random
    return source, target
