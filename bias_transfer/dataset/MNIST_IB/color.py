import numpy as np


# code adapted from https://github.com/salesforce/corr_based_prediction/blob/master/gen_color_mnist.py
# procedure following https://arxiv.org/pdf/1812.10352.pdf
# they variaed color_variance between 0.05 and 0.02 (in 0.005 steps)

class_color_means = [
    [60, 180, 75],  # green
    [255, 255, 25],  # yellow
    [0, 130, 200],  # blue
    [245, 130, 48],  # orange
    [70, 240, 240],  # cyan
    [240, 50, 230],  # magenta
    [230, 25, 75],  # red
    [0, 0, 128],  # navy
    [220, 190, 255],  # lavender
    [255, 250, 200],  # beige
]
nb_classes = 10


def get_color_codes():
    #     C = np.random.rand(nb_classes,3)
    C = np.asarray(class_color_means)
    C = C / np.max(C, axis=1)[:, None]
    return C


def get_std_color(means, targets, var):
    mean = means[targets].reshape((-1))
    cov = var * np.eye(mean.shape[0])
    c = np.random.multivariate_normal(mean=mean, cov=cov)
    c = c.reshape(targets.shape[0], 3, 1, 1)
    return c


def apply_color(
    x,
    targets,
    cfg_means=None,
    cbg_means=None,
    fg=True,
    bg=False,
    color_variance=0.0,
    shuffle=False,
):
    assert (
        len(x.shape) == 4
    ), "Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)"
    xs = x.shape
    x = (((x * 255) > 10) * 255).astype(np.float)  # thresholding to separate fg and bg
    x_rgb = np.ones((xs[0], 3, xs[2], xs[3])).astype(np.float)
    x_rgb = x_rgb * x
    targets_ = np.copy(targets)
    if shuffle:
        np.random.shuffle(targets)  # to generate cue-conflict by assigning wrong colors
    if fg:
        x_rgb_fg = 1.0 * x_rgb
        x_rgb_fg *= get_std_color(cfg_means, targets, color_variance)
    else:
        x_rgb_fg = np.zeros_like(x_rgb)
    if bg:
        x_rgb_bg = 255 - x_rgb
        x_rgb_bg *= get_std_color(cbg_means, targets, color_variance)
    else:
        x_rgb_bg = np.zeros_like(x_rgb)
    x_rgb = x_rgb_fg + x_rgb_bg
    x_rgb = np.clip(x_rgb, a_min=0.0, a_max=255.0)
    color_data_x = x_rgb / 255.0
    return color_data_x, targets_
