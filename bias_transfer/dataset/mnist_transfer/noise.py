import numpy as np


def apply_gaussian_noise(batch, targets, severity=1):
    if severity == -1:
        severity = np.random.randint(1,6)
    # adapted from https://github.com/google-research/mnist-c
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    return (
        np.clip(batch + np.random.normal(size=batch.shape, scale=c), 0, 1),
        targets,
    )
