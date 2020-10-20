import matplotlib.pyplot as plt


def plot_batch(batch, targets, n_rows, n_cols, name="", file_type="png"):
    batch = batch.transpose(0, 2, 3, 1)
    fig, axs = plt.subplots(n_rows, n_cols)
    if n_rows == 1:
        axs = [axs]
    for r in range(n_rows):
        for c in range(n_cols):
            axs[r][c].imshow(batch[r * n_cols + c].squeeze())
            axs[r][c].set_title(targets[r * n_cols + c])
            axs[r][c].set_axis_off()
    plt.show()
    if name:
        fig.savefig(
            name + "." + file_type,
            facecolor=fig.get_facecolor(),
            edgecolor=fig.get_edgecolor(),
            bbox_inches="tight",
        )
