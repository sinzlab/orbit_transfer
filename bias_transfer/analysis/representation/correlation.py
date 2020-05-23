import copy

import torch

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from bias_transfer.analysis.representation.analyzer import RepresentationAnalyzer

#TODO!!!

class CorrelationAnalyzer(RepresentationAnalyzer):
    def _plot_corr_matrix(
            self, mat, title="", file_name="", n_clusters=10, indices=None, acc=None
    ):
        fig, ax = self._plot_preparation(1, 1)
        if indices is None:
            clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(1 - mat)
            indices = np.argsort(clusters.labels_)
        sns.heatmap(
            mat[indices][:, indices],
            cmap="YlGnBu",
            xticklabels=400,
            yticklabels=400,
            vmin=0.0,
            vmax=1.0,
        )
        # sns.heatmap(mat[indices][:, indices], cmap="YlGnBu", xticklabels=400, yticklabels=400)
        sns.despine(offset=10, trim=True)
        if title:
            fig.suptitle(title, fontsize=16)
        if acc:
            ax.text(
                0.82, 0.93, "Accuracy: {:02.2f}".format(acc), transform=ax.transAxes
            )
        if file_name:
            fig.savefig(
                os.path.join(self.base_path, file_name),
                facecolor=fig.get_facecolor(),
                edgecolor=fig.get_edgecolor(),
                bbox_inches="tight",
            )
            plt.close(fig)
        return indices


    def _compute_corr_matrix(self, x, mode, noise_level):
        result = self._load_representation("corr", mode, noise_level)
        if result is None:
            x_flat = x.flatten(1, -1)
            # centered = (x_flat - x_flat.mean()) / x_flat.std()
            # result = (centered @ centered.transpose(0, 1)) / x_flat.size()[1]
            centered = x_flat - x_flat.mean(dim=1).view(-1, 1)
            result = (centered @ centered.transpose(0, 1)) / torch.ger(
                torch.norm(centered, 2, dim=1), torch.norm(centered, 2, dim=1)
            )  # see https://de.mathworks.com/help/images/ref/corr2.html
            print(torch.max(result))
            result = result.detach().cpu()
            self._save_representation(result, "corr", mode, noise_level)
        return result


    def corr_matrix(
            self, mode="clean", noise_level=0.0, clean_rep=None, *args, **kwargs
    ):
        self.clean_vs_noisy(noise_level=noise_level)
        title = "Correlation matrix for rep from {} data ".format(mode)
        if mode == "noisy":
            corr_matrix = self._compute_corr_matrix(
                self.noisy_rep[0], mode, noise_level
            )
            title += "(std = {:01.2f})".format(noise_level)
            acc = self.noisy_rep[1]
        else:
            corr_matrix = self._compute_corr_matrix(self.clean_rep[0], mode, 0.0)
            acc = self.clean_rep[1]

        clean_rep = self._plot_corr_matrix(
            corr_matrix,
            title=title + "\n" + "Model: " + self.experiment.comment,
            file_name=self._get_name("corr", mode, noise_level) + "_plot",
            indices=clean_rep,
            acc=acc,
        )
        return clean_rep
