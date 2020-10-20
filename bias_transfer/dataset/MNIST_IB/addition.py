import numpy as np


def apply_additon(source, target):
    second_summand = np.arange(source.shape[0])
    np.random.shuffle(second_summand)
    concat_source = np.concatenate([source, source[second_summand]], axis=3)
    summed_targets = target + target[second_summand]
    return concat_source, summed_targets