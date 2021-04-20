import numpy as np


class Subsample:
    active = True
    def __init__(self, total_per_class: int):
        self.per_class_left = {i: total_per_class for i in range(10)}

    def __call__(self, source, target):
        if not self.active:
            return source, target
        new_source = []
        new_target = []
        for i in range(target.shape[0]):
            if self.per_class_left[target[i]] > 0:
                self.per_class_left[target[i]] -= 1
                new_source.append(source[i].copy())
                new_target.append(target[i].copy())
        if new_source:
            return np.concatenate(new_source), np.array(new_target)
        else:
            return None, None
