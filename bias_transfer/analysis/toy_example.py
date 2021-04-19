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
            if len(importance.squeeze().shape) == 1:
                importance = np.diag(importance.squeeze())
            covariance = np.linalg.inv(importance)
        else:
            covariance = np.array([[1, 0], [0, 1]])
        alpha = (
            (Trainer & restr)
            .fetch1("trainer_config")
            .get("regularization")
            .get("alpha")
        )
        if alpha:
            return covariance / alpha
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
            contour = ax.contour(
                X, Y, Z, levels=levels, origin="lower", linewidths=0.5, **contour_kwargs
            )

        return contour

    def plot_gauss_contour(self, mu, cov, ax, color):
        norm = np.random.multivariate_normal(mu, cov, size=10000000)
        self.density_contour(norm[:, 0], norm[:, 1], 100, 100, ax=ax, colors=color)

    def plot_decision_line(self, w0, w1, w2, x_lim, ax, color, label):
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
        ax[0][0].arrow(-w0 / w1, 0, w1, w2, color=color)
        #     print(w0)
        plt.plot(x, y, color=color, label=label)

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
    ):

        ax[0][0].scatter(data_a[:, 0], data_a[:, 1], color=color_a, marker="+")
        ax[0][0].scatter(data_b[:, 0], data_b[:, 1], color=color_b, marker="_")
        self.plot_gauss_contour(mu_a, cov_a, ax[0][0], color=color_a)
        self.plot_gauss_contour(mu_b, cov_b, ax[0][0], color=color_b)

    @plot
    def plot_model(
        self,
        fig,
        ax,
        w0=None,
        w1=None,
        w2=None,
        line_color=None,
        line_label="",
    ):
        if line_color and line_label:
            self.plot_decision_line(w0, w1, w2, 10, ax, line_color, line_label)

    def plot_everything(self, configs, save=""):
        colors = {
            "light_blue": "#A6CEE3",
            "dark_blue": "#2578B3",
            "light_red": "#FB9A99",
            "dark_red": "#E31E1B",
            "light_green": "#B2DF8A",
            "dark_green": "#33A02C",
            "light_grey": "#D3D3D3",
            "dark_grey": "#696969",
            "dark_orange": "#FA7E01",
            "dark_violet": "#6A3D9A",
        }
        model_colors = ["dark_green", "dark_orange", "dark_violet", "light_green"]
        # fig_weight = plt.figure()
        # ax_weight = fig_weight.add_subplot(projection="3d")
        fig, ax = None, None
        dataset_plotted = False

        for i, (description, config) in enumerate(configs.items()):
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
                    ratio=(1, 1),
                    legend_outside=False,
                )
                ax[0][0].set_ylim(-10, 10)
                ax[0][0].set_xlim(-10, 10)
                self.plot_model(
                    fig=fig,
                    ax=ax,
                    w0=w0,
                    w1=w1,
                    w2=w2,
                    line_color=colors[model_colors[0]],
                    line_label="Source Task",
                )
            # ax_weight.scatter(w0, w1, w2, color=colors[4])

            transfer_cov = self.retrieve_transfer_covariance(restr_1)
            if transfer_cov is not None:
                self.plot_gauss_contour(
                    mu=np.array([w1, w2]),
                    cov=transfer_cov,
                    ax=ax[0][0],
                    color=colors[model_colors[i + 1]],
                )

            (w0, w1, w2, data_a, data_b, mu_a, cov_a, mu_b, cov_b) = self.retrieve(
                restr_1
            )
            self.plot_model(
                fig=fig,
                ax=ax,
                w0=w0,
                w1=w1,
                w2=w2,
                line_color=colors[model_colors[i + 1]],
                line_label=description.name,
            )
            if not dataset_plotted:
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
                )
            # ax_weight.scatter(w0, w1, w2, color=colors[5])

            (w0, w1, w2, data_a, data_b, mu_a, cov_a, mu_b, cov_b) = self.retrieve(
                restr_2
            )
            if not dataset_plotted:
                fig, ax = self.plot_dataset(
                    fig=fig,
                    ax=ax,
                    data_a=data_a,
                    data_b=data_b,
                    mu_a=mu_a,
                    cov_a=cov_a,
                    mu_b=mu_b,
                    cov_b=cov_b,
                    color_a=colors["light_grey"],
                    color_b=colors["dark_grey"],
                )
            dataset_plotted = True
        if save:
            save_plot(fig, save, ("png",))
