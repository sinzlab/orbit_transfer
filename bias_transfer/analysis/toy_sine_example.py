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


class ToyExampleAnalyzer(Analyzer):
    def plot_everything(
        self,
        configs,
        save="",
        style="light_talk",
        plot_source_ds=True,
        plot_target_ds=True,
        plot_eval_ds=True,
        plot_source_model=True,
        plot_target_model=True,
        plot_transition=True,
        plot_model=0,
        data_transfer=False,
        legend_on_both_ds=False,
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

        fig, ax = dummy_plot(ncols=2, style=style, **kwargs)
        dataset_plotted = False

        for i, (description, config) in enumerate(configs.items()):
            if i != plot_model:
                continue
            restr_0 = config.get_restrictions(level=0)
            if len(config) == 3:
                restr_1 = config.get_restrictions(level=1)
                restr_2 = config.get_restrictions(level=2)
            elif len(config) == 4:
                restr_1 = config.get_restrictions(level=2)
                restr_2 = config.get_restrictions(level=3)
            else:
                ValueError("TransferExperiment has wrong length/format!")

            model, data = self.retrieve(restr_0)
            if not dataset_plotted:
                if plot_source_model:
                    self.plot_model(
                        fig=fig,
                        ax=ax,
                        model=model,
                        color=colors[model_colors[0][0]],
                        label="Source Env. Solution",
                    )
                if plot_source_ds:
                    fig, ax = self.plot_dataset(
                        fig=fig,
                        ax=ax,
                        data=data,
                        color=colors["dark_red"],
                        label="Source Environment",
                        # legend_on_both=legend_on_both_ds,
                    )

            if plot_transition and plot_source_model:
                transfer_cov = self.retrieve_transfer_covariance(restr_1)
                if transfer_cov is not None:
                    cov = ax[0][1].imshow(transfer_cov)
                    plt.colorbar(cov, ax=ax[0][1])

            if not dataset_plotted and plot_eval_ds:
                _, data = self.retrieve(restr_2)
                fig, ax = self.plot_dataset(
                    fig=fig,
                    ax=ax,
                    data=data,
                    color=colors["light_blue"],
                    label="Evaluation Environment",
                )

            model, data = self.retrieve(restr_1, data_transfer=data_transfer)
            if plot_target_model:
                self.plot_model(
                    fig=fig,
                    ax=ax,
                    model=model,
                    color=colors[model_colors[i + 1][0]],
                    label=description.name,
                )
            if not dataset_plotted and plot_target_ds:
                fig, ax = self.plot_dataset(
                    fig=fig,
                    ax=ax,
                    data=data,
                    color=colors["dark_blue"],
                    label="Target Environment",
                )
            dataset_plotted = True
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
        train_points, train_labels = data_loaders["train"]["regression"].dataset.tensors
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

    @plot
    def plot_dataset(self, fig, ax, data, color, label):
        if len(data[1].shape) == 1:
            ax[0][0].scatter(data[0], data[1], color=color, label=label, marker=".")
        return fig, ax

    def plot_model(self, fig, ax, model, color, label):
        model.eval()
        X_plot = np.linspace(-5, 10, 1000).reshape(-1, 1)
        X_plot_torch = torch.from_numpy(X_plot).type(torch.float32)
        Y_pred = model(X_plot_torch)
        if isinstance(Y_pred, tuple):
            Y_pred = Y_pred[1]
        Y_pred = Y_pred.detach().numpy()

        ax[0][0].plot(X_plot, Y_pred, color=color, label=label)

    def retrieve_transfer_covariance(self, restr):
        seed = (Seed & restr).fetch1("seed")
        restr["data_transfer"] = True
        data_loaders, model, _ = TransferredTrainedModel().load_model(
            restr, include_trainer=True, include_state_dict=True, seed=seed
        )
        ds = data_loaders["train"]["transfer"].dataset.target_datasets
        V = ds.get("layers.6_cov_V", ds.get("layers.9_cov_V")).tensors[0].numpy()

        inputs = (
            data_loaders["train"]["transfer"]
            .dataset.source_datasets["img"]
            .tensors[0]
            .numpy()
        )
        idx = np.argsort(inputs, axis=0)

        n = inputs.shape[0]
        # inputs = inputs[idx].reshape(n)
        # transfer_covariance = np.zeros((n, n))
        # for i in range(n):
        #     # transfer_covariance[i, :] = np.exp(-2 * np.sin(math.pi * (inputs[i] - inputs))**2  )
        #     transfer_covariance[i, :] = np.cos(inputs[i]- inputs)
        # transfer_covariance += np.eye(n) * 0.1
        # importance = np.linalg.inv(transfer_covariance)

        V = V[idx[:, 0]].reshape(V.shape[0], -1)
        transfer_covariance = V @ V.T
        print(transfer_covariance)
        return transfer_covariance
