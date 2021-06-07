import math
import tempfile

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import nnfabrik as nnf
from nnfabrik.utility.nn_helpers import load_state_dict

from nntransfer.tables.transfer import TransferredTrainedModel
from nntransfer.tables.nnfabrik import Dataset, Seed, Trainer, Model
from nntransfer.analysis.results.base import Analyzer
from nntransfer.analysis.plot import plot, save_plot


class CIFARExampleAnalyzer(Analyzer):
    def plot_everything(
        self,
        configs,
        save="",
        style="light_talk",
        plot_transition=True,
        data_transfer=False,
        plot_model=0,
        **kwargs,
    ):
        colors = {
            "light_blue": "#A6CEE3",
            "dark_blue": "#2578B3",
            "light_red": "#FB9A99",
            "dark_red": "#E31E1B",
            "turquoise1": "#137155",
            "turquoise2": "#25D9A4",
            "turquoise3": "#B0D9CD",
            "orange1": "#924001",
            "orange2": "#F16A02",
            "orange3": "#F19854",
            "violet1": "#646099",
            "violet2": "#746BEB",
            "violet3": "#A59EF6",
            "pink1": "#CA0067",
            "pink2": "#F13897",
            "pink3": "#E767A8",
            "green1": "#1F4500",
            "green2": "#3D8600",
            "green3": "#90BC5E",
            "brown1": "#7D5916",
            "brown2": "#C98F23",
            "brown3": "#A38B5F",
            "grey1": "#353535",
            "grey2": "#666666",
            "grey3": "#9C9C9C",
        }
        model_colors = [
            ("grey2", "grey2"),
            ("orange2", "orange2"),
            ("pink2", "pink2"),
            ("violet3", "violet3"),
            ("green2", "green2"),
            ("brown2", "brown2"),
        ]

        @plot
        def dummy_plot(fig, ax):
            pass
            # ax[0][0].set_ylim(-10, 10)
            # ax[0][0].set_xlim(-10, 10)

        fig, ax = dummy_plot(ncols=1, style=style, **kwargs)
        dataset_plotted = False

        for i, (description, config) in enumerate(configs.items()):
            if i != plot_model:
                continue
            restr_0 = config.get_restrictions(level=0)
            restr_1 = config.get_restrictions(level=len(config)-1)

            # model, data = self.retrieve(restr_0)

            if plot_transition:
                transfer_cov = self.retrieve_transfer_covariance(restr_1)
                if transfer_cov is not None:
                    cov = ax[0][0].imshow(transfer_cov)
                    plt.colorbar(cov, ax=ax[0][0])

        legend_args = {
            "fontsize": 12,
            "title_fontsize": "13",
            "frameon": False,
            "borderaxespad": 0.0,
            "bbox_to_anchor": (0.1, 1.1),
            "loc": "lower left",
            "ncol": 3,
        }
        ax[0][0].legend(**legend_args)
        if save:
            save_plot(
                fig,
                save,
                types=("png", "pdf", "pgf") if "tex" in style else ("png", "pdf"),
            )
        return transfer_cov

    def retrieve(self, restr, data_transfer=False):
        seed = (Seed & restr).fetch1("seed")
        if data_transfer:
            restr["data_transfer"] = True
        data_loaders, model = TransferredTrainedModel().load_model(
            restr, include_trainer=False, include_state_dict=False, seed=seed
        )
        train_points, train_labels = data_loaders["train"]["img_classification"].dataset.tensors
        data = (train_points.numpy(), train_labels.numpy().squeeze())
        with tempfile.TemporaryDirectory() as temp_dir:
            file = (TransferredTrainedModel.ModelStorage & restr).fetch1(
                "model_state", download_path=temp_dir
            )
            state_dict = torch.load(file)
            load_state_dict(model, state_dict, ignore_unused=True)
        return model, data

    def plot_kernel(self, kernel, x):
        K_plot = kernel(x, x)
        plt.imshow(K_plot)
        plt.colorbar()

    def retrieve_transfer_covariance(self, restr):
        seed = (Seed & restr).fetch1("seed")
        restr["data_transfer"] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            file = ((TransferredTrainedModel.DataStorage) & restr).fetch1("transfer_data", download_path=temp_dir)
            transfer_data = np.load(file)

        V = transfer_data.get("fc2_cov_V", transfer_data.get("fc3_cov_V"))

        inputs = transfer_data["source"]
        # idx = np.argsort(inputs, axis=0)
        idx = np.arange(0,500)

        n = inputs.shape[0]
        # inputs = inputs[idx].reshape(n)
        # transfer_covariance = np.zeros((n, n))
        # for i in range(n):
        #     # transfer_covariance[i, :] = np.exp(-2 * np.sin(math.pi * (inputs[i] - inputs))**2  )
        #     transfer_covariance[i, :] = np.cos(inputs[i]- inputs)
        # transfer_covariance += np.eye(n) * 0.1
        # importance = np.linalg.inv(transfer_covariance)

        config = (Trainer() & restr).fetch1("trainer_config")

        print(idx.shape)
        if config.get("regularization",{}).get("marginalize_over_hidden"):
            V = V[idx].reshape(V.shape[0],-1)  # What we want to show
        else:
            V = V[idx].reshape(-1, V.shape[2])

        print(V.shape)


        V = (V - np.mean(V, axis=1, keepdims=True)) / math.sqrt(V.shape[1])

        transfer_covariance = V @ V.T

        transfer_covariance += np.eye(V.shape[0]) * config.get("regularization",{}).get("cov_eps",0.0)
        print(transfer_covariance)
        return transfer_covariance
