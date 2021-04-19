import numpy as np
from PIL import Image


def apply_scale(source, target):
    orig_shape = source.shape
    expanded_batch = np.zeros(orig_shape)
    source = source[:, :, 6:-6, 6:-6]  # undo expansion
    scale = np.random.uniform(0.1, 1.4, size=source.shape[0])
    for i in range(source.shape[0]):
        scaled_size = int(28.0 * scale[i])
        scaled_array = scale_image(source[i], scaled_size)
        padding = 20 - scaled_size // 2
        expanded_batch[
            i, 0, padding : padding + scaled_size, padding : padding + scaled_size
        ] = scaled_array
    return expanded_batch, target


def scale_image(image, size):
    array = (np.reshape(image, (28, 28)) * 255).astype(np.uint8)
    img = Image.fromarray(array, "L")
    scaled_img = img.resize((size, size), Image.ANTIALIAS)
    scaled_array = np.array(scaled_img) / 255
    return scaled_array
