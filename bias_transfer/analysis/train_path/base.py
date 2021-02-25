import tempfile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bias_transfer.analysis.plot import save_plot, plot
from bias_transfer.tables.transfer import Checkpoint
from neuralpredictors.tracking import AdvancedMultipleObjectiveTracker as Tracker


class Analyzer:
    def __init__(self):
        self.data = {}

    def load_data(self, configs):
        # Select data:
        with tempfile.TemporaryDirectory() as temp_dir:
            for description, config in configs.items():
                level = 0
                while True:
                    restriction = config.get_restrictions(level)
                    if not restriction:
                        break
                    restricted = Checkpoint() & restriction
                    if restricted:  # could be empty if entry is not computed yet
                        fetch_res = restricted.fetch("state", "epoch", as_dict=True, download_path=temp_dir)
                        if description not in self.data:
                            self.data[description] = {}
                        for res in fetch_res:
                            data = self.data.get(description, {}).get(level, {})
                            data[res["epoch"]] = torch.load(res["state"])["net"]
                            self.data[description][level] = data
                    level += 1

    def _compute_pca(self, tensors):
        pca = PCA(n_components=2)
        pca.fit(tensors)
        pca_result = pca.transform(tensors)
        print(
            "Explained variation per principal component: {}".format(
                pca.explained_variance_ratio_
            ),
            flush=True,
        )
        return pca_result

    def _compute_tsne(self, tensors):
        tsne = TSNE(
            n_components=2, verbose=1, perplexity=40, n_iter=250, init="pca"
        )
        return tsne.fit_transform(tensors)


    def _flatten_state_dict(self, state):
        parameters = []
        for param in state.values():
            parameters.append(torch.flatten(param).cpu().numpy())
        return np.concatenate(parameters)

    @plot
    def plot_paths(self, fig, ax, level=0, method="pca"):
        parameters = []
        labels = []
        for descr, states in self.data.items():
            states = states[level]
            for epoch, state in states.items():
                parameters.append(self._flatten_state_dict(state))
                labels.append(f"{descr.name}, Seed {descr.seed}")
        parameters = np.stack(parameters)
        if method == "tsne":
            result = self._compute_tsne(parameters)
        else:
            result = self._compute_pca(parameters)
        sns.scatterplot(x=result[:,0], y=result[:,1], hue=labels, ax=ax)

