import numpy as np

from bias_transfer.dataset.mnist_transfer.scale import scale_image


def apply_additon(source, target):
    orig_shape = source.shape  # 40 x 40 (one image)
    scaled_batch = np.zeros([orig_shape[0], orig_shape[1], 20, 20])
    source = source[:, :, 6:-6, 6:-6]  # undo expansion -> 28 x 28
    for i in range(source.shape[0]):
        scaled_array = scale_image(source[i], 20)
        scaled_batch[i, 0, :, :] = scaled_array
    second_summand = np.arange(orig_shape[0])
    np.random.shuffle(second_summand)
    concat_source = np.concatenate(
        [scaled_batch, scaled_batch[second_summand]], axis=3
    )  # 20 x 40 (two images)
    summed_targets = target + target[second_summand]
    expanded_concat = np.zeros(orig_shape)
    expanded_concat[:, :, 10:-10, :] = concat_source  # 40 x 40 (two images)
    return expanded_concat, summed_targets
