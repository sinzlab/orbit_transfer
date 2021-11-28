import numpy as np


def apply_translation(source, target, std=5, positive_translation = False, negative_translation=False):
    # 40x40 to follow https://www.cs.toronto.edu/~tijmen/affNIST/ for translation
    offsets = np.clip(
        np.random.normal(scale=std, size=source.shape[0] * 2), a_min=-6, a_max=6
    )
    if positive_translation:
        offsets = np.abs(offsets)
    if negative_translation:
        offsets = -1 * np.abs(offsets)
    offsets = offsets.astype(np.int)
    x_offset, y_offset = offsets[: source.shape[0]], offsets[source.shape[0]:]
    for b in range(source.shape[0]):
        source[b, 0, :, :] = np.roll(
            source[b, 0, :, :], (y_offset[b], x_offset[b]), axis=(0, 1)
        )
    return source, target
