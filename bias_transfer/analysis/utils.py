import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from nnfabrik.utility.dj_helpers import make_hash


def set_size(width, ratio=None, fraction=1, subplots=(1, 1), gridspec_kw=None):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 426.79135
    elif width == "beamer":
        width_pt = 307.28987
    elif width == "pnas":
        width_pt = 246.09686
    elif width == "nips":
        width_pt = 397.48499
    elif "paper" in width or "talk" in width:
        width_pt = 1000
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if ratio:
        ratio = ratio[0] / ratio[1]
    else:
        # Golden ratio to set aesthetic figure height
        # https://disq.us/p/2940ij3
        ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if gridspec_kw and "height_ratios" in gridspec_kw:
        fig_height_in = 0
        max_height = max(gridspec_kw["height_ratios"])
        for h_ratio in gridspec_kw["height_ratios"]:
            fig_height_in += fig_width_in * ratio * (h_ratio/max_height)
    else:
        fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def plot_preparation(
    style,
    nrows=1,
    ncols=1,
    ratio=None,
    fraction=1.0,
    sharex=False,
    sharey=False,
    gridspec_kw=None,
):
    fs = set_size(width=style, ratio=ratio, fraction=fraction, subplots=(ncols, nrows), gridspec_kw=gridspec_kw)
    sns.set()
    if "nips" in style:
        sns.set_style(
            "whitegrid",
            {
                "axes.edgecolor": "0.1",
                "xtick.bottom": True,
                "xtick.top": False,
                "ytick.left": True,
                "ytick.right": False,
            },
        )
        nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
        mpl.rcParams.update(nice_fonts)
        mpl.use("pgf")
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )
    if "light" in style:
        sns.set_style("whitegrid")
    if "ticks" in style:
        sns.set_style("ticks")
    if "dark" in style:
        plt.style.use("dark_background")
    if "talk" in style:
        sns.set_context("talk")
    else:
        sns.set_context("paper")
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=fs,
        # dpi=dpi,
        sharex=sharex,
        sharey=sharey,
        gridspec_kw=gridspec_kw,
    )
    return fig, ax

def save_plot(fig, name, types=("pgf","pdf","png")):
    for file_type in types:
        fig.savefig(
            name + "." + file_type,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            bbox_inches="tight",
        )
    plt.close(fig)


def generate_gif(filenames, out_name):
    import imageio

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(
        "{}.gif".format(out_name),
        images,
        format="GIF",
        duration=2,
    )

