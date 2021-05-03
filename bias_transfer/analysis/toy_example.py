import tempfile

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import nnfabrik as nnf

from nntransfer.tables.transfer import TransferredTrainedModel
from nntransfer.tables.nnfabrik import Dataset, Seed, Trainer
from nntransfer.analysis.results.base import Analyzer
from nntransfer.analysis.plot import plot, save_plot


class ToyExampleAnalyzer(Analyzer):
    def retrieve(self, restr):
        # model
        with tempfile.TemporaryDirectory() as temp_dir:
            file = (TransferredTrainedModel.ModelStorage & restr).fetch1(
                "model_state", download_path=temp_dir
            )
            state_dict = torch.load(file)
        #     w0 = state_dict["bias"].item()
        w0 = 0
        if "w_posterior_mean" in state_dict:
            w1 = state_dict["w_posterior_mean"][:, 0].item()
            w2 = state_dict["w_posterior_mean"][:, 1].item()
        else:
            w1 = state_dict["weight"][:, 0].item()
            w2 = state_dict["weight"][:, 1].item()
        # dataset
        dataset_fn, dataset_config = (Dataset & restr).fetch1(
            "dataset_fn", "dataset_config"
        )
        dataset_config["seed"] = (Seed & restr).fetch1("seed")
        data_loaders = nnf.builder.get_data(dataset_fn, dataset_config)
        test_points, test_labels = next(
            iter(data_loaders["test"]["img_classification"])
        )
        test_points, test_labels = test_points.numpy(), test_labels.numpy().squeeze()
        data_a = test_points[test_labels == 1]
        data_b = test_points[test_labels == 0]
        mu_a = np.asarray(dataset_config["mu_a"])
        cov_a = np.asarray(dataset_config["cov_a"])
        mu_b = np.asarray(dataset_config["mu_b"])
        cov_b = np.asarray(dataset_config["cov_b"])
        return w0, w1, w2, data_a, data_b, mu_a, cov_a, mu_b, cov_b

    def retrieve_transfer_covariance(self, restr):
        # model
        with tempfile.TemporaryDirectory() as temp_dir:
            file = (TransferredTrainedModel.ModelStorage & restr).fetch1(
                "model_state", download_path=temp_dir
            )
            state_dict = torch.load(file)
        if "weight_importance" in state_dict:
            importance = state_dict["weight_importance"].numpy()
            if "weight_importance_v" in state_dict:
                v = state_dict["weight_importance_v"].numpy()
                v = v.reshape(v.shape[0], -1)
                elrg_alpha = 1 / v.shape[0]
                print("alpha", elrg_alpha)
                print("V", v)
                print(v.transpose() @ v)
                print("diag", importance)
                covariance = elrg_alpha * v.transpose() @ v + np.diag(
                    1 / importance.squeeze()
                )
                print("covariance", covariance)
            elif len(importance.squeeze().shape) == 1:
                importance = np.diag(importance.squeeze())
                covariance = np.linalg.inv(importance)
        else:
            covariance = np.array([[1, 0], [0, 1]])
        gamma = (
            (Trainer & restr)
            .fetch1("trainer_config")
            .get("regularization")
            .get("gamma")
        )
        if gamma:
            return covariance / gamma
        else:
            return None

    def find_confidence_interval(self, x, pdf, confidence_level):
        return pdf[pdf > x].sum() - confidence_level

    def density_contour(
        self, xdata, ydata, nbins_x, nbins_y, ax=None, **contour_kwargs
    ):
        """Create a density contour plot.
        Parameters
        ----------
        xdata : numpy.ndarray
        ydata : numpy.ndarray
        nbins_x : int
            Number of bins along x dimension
        nbins_y : int
            Number of bins along y dimension
        ax : matplotlib.Axes (optional)
            If supplied, plot the contour to this axis. Otherwise, open a new figure
        contour_kwargs : dict
            kwargs to be passed to pyplot.contour()
        """

        H, xedges, yedges = np.histogram2d(
            xdata, ydata, bins=(nbins_x, nbins_y), normed=True
        )
        x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins_x))
        y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y, 1))

        pdf = H * (x_bin_sizes * y_bin_sizes)

        one_sigma = so.brentq(self.find_confidence_interval, 0.0, 1.0, args=(pdf, 0.68))
        two_sigma = so.brentq(self.find_confidence_interval, 0.0, 1.0, args=(pdf, 0.95))
        three_sigma = so.brentq(
            self.find_confidence_interval, 0.0, 1.0, args=(pdf, 0.99)
        )
        levels = [
            one_sigma,
            #               two_sigma,
            #               three_sigma
        ]

        X, Y = 0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1])
        Z = pdf.T

        if ax == None:
            contour = plt.contour(
                X, Y, Z, levels=levels, origin="lower", linewidths=0.5, **contour_kwargs
            )
        else:
            print("contour")
            contour = ax.contour(
                X, Y, Z, levels=levels, origin="lower", linewidths=0.5, **contour_kwargs
            )

        return contour

    def plot_gauss_contour(self, mu, cov, ax, color):
        norm = np.random.multivariate_normal(mu, cov, size=200000000)
        self.density_contour(norm[:, 0], norm[:, 1], 400, 400, ax=ax, colors=color)

    def plot_decision_line(self, w0, w1, w2, x_lim, ax, color, color_light, label):
        if w0 == 0:
            m = -w1 / w2
            b = 0
        else:
            m = -(w0 / w2) / (w0 / w1)
            b = -w0 / w2
        x = np.linspace(-1 * x_lim, x_lim, 100)
        y = m * x + b
        #     origin = np.array([-w0/w1],[0]) # origin point
        #     ax[0][0].quiver(*origin, w1, w2, color=color, scale=1)
        ax[0][0].arrow(
            -w0 / w1, 0, w1, w2, color=color_light, linestyle=":", linewidth=0.8
        )

        plt.plot(x, y, color=color, label=label, linewidth=1)
        ax[0][0].plot(
            w1,
            w2,
            "d",
            color=color_light,
            label=label + ": Weight",
            linewidth=0.8,
            ms=2,
        )

    @plot
    def plot_dataset(
        self,
        fig,
        ax,
        mu_a=None,
        cov_a=None,
        mu_b=None,
        cov_b=None,
        data_a=None,
        data_b=None,
        color_a=None,
        color_b=None,
        label="",
        legend_on_both=False,
    ):
        if legend_on_both:
            ax[0][0].scatter(
                data_a[:, 0],
                data_a[:, 1],
                color=color_a,
                marker="+",
                linewidths=0.8,
                s=4,
                label=f"{label}: Class A",
            )
            ax[0][0].scatter(
                data_b[:, 0],
                data_b[:, 1],
                color=color_b,
                marker="_",
                linewidths=0.8,
                s=4,
                label=f"{label}: Class B",
            )
        else:
            ax[0][0].scatter(
                data_a[:, 0],
                data_a[:, 1],
                color=color_a,
                marker="+",
                linewidths=0.8,
                s=4,
                label=label,
            )
            ax[0][0].scatter(
                data_b[:, 0],
                data_b[:, 1],
                color=color_b,
                marker="_",
                linewidths=0.8,
                s=4,
            )
        # self.plot_gauss_contour(mu_a, cov_a, ax[0][0], color=color_a)
        # self.plot_gauss_contour(mu_b, cov_b, ax[0][0], color=color_b)

    @plot
    def plot_model(
        self,
        fig,
        ax,
        w0=None,
        w1=None,
        w2=None,
        line_color=None,
        arrow_color=None,
        line_label="",
    ):
        if line_color and line_label:
            self.plot_decision_line(
                w0, w1, w2, 10, ax, line_color, arrow_color, line_label
            )

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
        legend_on_both_ds=False,
            ** kwargs,
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
            ax[0][0].set_ylim(-10, 10)
            ax[0][0].set_xlim(-10, 10)

        fig, ax = dummy_plot(ratio=(1, 1), style=style, **kwargs)
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

            (w0, w1, w2, data_a, data_b, mu_a, cov_a, mu_b, cov_b) = self.retrieve(
                restr_0
            )
            if not dataset_plotted:
                if plot_source_model:
                    self.plot_model(
                        fig=fig,
                        ax=ax,
                        w0=w0,
                        w1=w1,
                        w2=w2,
                        line_color=colors[model_colors[0][0]],
                        arrow_color=colors[model_colors[0][1]],
                        line_label="Source Env. Solution",
                    )
                if plot_source_ds:
                    fig, ax = self.plot_dataset(
                        fig=fig,
                        ax=ax,
                        data_a=data_a,
                        data_b=data_b,
                        mu_a=mu_a,
                        cov_a=cov_a,
                        mu_b=mu_b,
                        cov_b=cov_b,
                        color_a=colors["dark_blue"],
                        color_b=colors["dark_red"],
                        label="Source Environment",
                        legend_on_both=legend_on_both_ds,
                    )

            if plot_transition and plot_source_model:
                transfer_cov = self.retrieve_transfer_covariance(restr_1)
                print(description.name, transfer_cov)
                if transfer_cov is not None:
                    self.plot_gauss_contour(
                        mu=np.array([w1, w2]),
                        cov=transfer_cov,
                        ax=ax[0][0],
                        color=colors[model_colors[i + 1][0]],
                    )

            if not dataset_plotted and plot_eval_ds:
                (w0, w1, w2, data_a, data_b, mu_a, cov_a, mu_b, cov_b) = self.retrieve(
                    restr_2
                )
                fig, ax = self.plot_dataset(
                    fig=fig,
                    ax=ax,
                    data_a=data_a,
                    data_b=data_b,
                    mu_a=mu_a,
                    cov_a=cov_a,
                    mu_b=mu_b,
                    cov_b=cov_b,
                    color_a=colors["light_blue"],
                    color_b=colors["light_red"],
                    label="Evaluation Environment",
                )

            (w0, w1, w2, data_a, data_b, mu_a, cov_a, mu_b, cov_b) = self.retrieve(
                restr_1
            )
            if plot_target_model:
                self.plot_model(
                    fig=fig,
                    ax=ax,
                    w0=w0,
                    w1=w1,
                    w2=w2,
                    line_color=colors[model_colors[i + 1][0]],
                    arrow_color=colors[model_colors[i + 1][1]],
                    line_label=description.name,
                )
            if not dataset_plotted and plot_target_ds:
                fig, ax = self.plot_dataset(
                    fig=fig,
                    ax=ax,
                    data_a=data_a,
                    data_b=data_b,
                    mu_a=mu_a,
                    cov_a=cov_a,
                    mu_b=mu_b,
                    cov_b=cov_b,
                    color_a=colors["dark_blue"],
                    color_b=colors["dark_red"],
                    label="Target Environment",
                )

            dataset_plotted = True
        legend_args = {
            "fontsize": 12,
            "title_fontsize": "13",
            "frameon": False,
            "borderaxespad": 0.0,
            "bbox_to_anchor": (1.05, 1, 1.5, 0.05),
            "loc": 2,
        }
        plt.legend(**legend_args)
        if save:
            save_plot(
                fig,
                save,
                types=("png", "pdf", "pgf") if "tex" in style else ("png", "pdf"),
            )
