import os
import copy
import math
import shutil

import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable

from bias_transfer.analysis.plot import plot_preparation, save_plot
from bias_transfer.models import IntermediateLayerGetter
from bias_transfer.trainer.main_loop_modules import NoiseAugmentation
from nnfabrik.utility.dj_helpers import make_hash
from .analyzer import RepresentationAnalyzer

ALL_REPRESENTATIONS = {
    "conv1": "layer0.conv1",
    # "relu": "layer0.relu",
    # layer1
    "layer1.0.conv1": "layer1.0.conv1",
    "layer1.0.conv2": "layer1.0.conv2",
    "layer1.0.conv3": "layer1.0.conv3",
    # "layer1.0.relu": "layer1.0.relu",
    "layer1.1.conv1": "layer1.1.conv1",
    "layer1.1.conv2": "layer1.1.conv2",
    "layer1.1.conv3": "layer1.1.conv3",
    # "layer1.1.relu": "layer1.1.relu",
    "layer1.2.conv1": "layer1.2.conv1",
    "layer1.2.conv2": "layer1.2.conv2",
    "layer1.2.conv3": "layer1.2.conv3",
    # "layer1.2.relu": "layer1.2.relu",
    # layer2
    "layer2.0.conv1": "layer2.0.conv1",
    "layer2.0.conv2": "layer2.0.conv2",
    "layer2.0.conv3": "layer2.0.conv3",
    # "layer2.0.relu": "layer2.0.relu",
    "layer2.1.conv1": "layer2.1.conv1",
    "layer2.1.conv2": "layer2.1.conv2",
    "layer2.1.conv3": "layer2.1.conv3",
    # "layer2.1.relu": "layer2.1.relu",
    "layer2.2.conv1": "layer2.2.conv1",
    "layer2.2.conv2": "layer2.2.conv2",
    "layer2.2.conv3": "layer2.2.conv3",
    # "layer2.2.relu": "layer2.2.relu",
    "layer2.3.conv1": "layer2.3.conv1",
    "layer2.3.conv2": "layer2.3.conv2",
    "layer2.3.conv3": "layer2.3.conv3",
    # "layer2.3.relu": "layer2.3.relu",
    # layer3
    "layer3.0.conv1": "layer3.0.conv1",
    "layer3.0.conv2": "layer3.0.conv2",
    "layer3.0.conv3": "layer3.0.conv3",
    # "layer3.0.relu": "layer3.0.relu",
    "layer3.1.conv1": "layer3.1.conv1",
    "layer3.1.conv2": "layer3.1.conv2",
    "layer3.1.conv3": "layer3.1.conv3",
    # "layer3.1.relu": "layer3.1.relu",
    "layer3.2.conv1": "layer3.2.conv1",
    "layer3.2.conv2": "layer3.2.conv2",
    "layer3.2.conv3": "layer3.2.conv3",
    # "layer3.2.relu": "layer3.2.relu",
    "layer3.3.conv1": "layer3.3.conv1",
    "layer3.3.conv2": "layer3.3.conv2",
    "layer3.3.conv3": "layer3.3.conv3",
    # "layer3.3.relu": "layer3.3.relu",
    "layer3.4.conv1": "layer3.4.conv1",
    "layer3.4.conv2": "layer3.4.conv2",
    "layer3.4.conv3": "layer3.4.conv3",
    # "layer3.4.relu": "layer3.4.relu",
    "layer3.5.conv1": "layer3.5.conv1",
    "layer3.5.conv2": "layer3.5.conv2",
    "layer3.5.conv3": "layer3.5.conv3",
    # "layer3.5.relu": "layer3.5.relu",
    # layer4
    "layer4.0.conv1": "layer4.0.conv1",
    "layer4.0.conv2": "layer4.0.conv2",
    "layer4.0.conv3": "layer4.0.conv3",
    # "layer4.0.relu": "layer4.0.relu",
    "layer4.1.conv1": "layer4.1.conv1",
    "layer4.1.conv2": "layer4.1.conv2",
    "layer4.1.conv3": "layer4.1.conv3",
    # "layer4.1.relu": "layer4.1.relu",
    "layer4.2.conv1": "layer4.2.conv1",
    "layer4.2.conv2": "layer4.2.conv2",
    "layer4.2.conv3": "layer4.2.conv3",
    # "layer4.2.relu": "layer4.2.relu",
    # core output
    "flatten": "core",
    "fc": "readout",
}


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device)
    I = torch.eye(n, device=K.device)
    H = I - unit / n

    return torch.mm(
        torch.mm(H, K), H
    )  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = torch.dot(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = torch.mm(X, X.T)
    L_Y = torch.mm(Y, Y.T)
    return torch.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


def pairwise_l2_distances(x, y=None):
    """
    see: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def RDM(X, dist_measure="corr"):
    X = X - X.mean(dim=-1).unsqueeze(-1)
    if dist_measure == "corr":
        result = (X @ torch.transpose(X, 0, 1)) / torch.ger(
            torch.norm(X, 2, dim=1), torch.norm(X, 2, dim=1)
        )
    elif dist_measure == "l2":
        result = pairwise_l2_distances(X)
    return result


def RDM_comparison(X, Y, dist_measure="corr"):
    RDM_X = RDM(X, dist_measure).flatten()
    RDM_Y = RDM(Y, dist_measure).flatten()
    result = RDM_X @ RDM_Y.T
    result /= (X.shape[0]) ** 2
    return result


def similarity(X, Y, dist_measure="CKA"):
    if dist_measure == "CKA":
        return linear_CKA(X, Y)
    else:
        return RDM_comparison(X, Y, dist_measure)


class NoiseStabilityAnalyzer(RepresentationAnalyzer):
    def __init__(
        self,
        num_samples=0,
        num_repeats=4,
        noise_std_max=0.51,
        noise_std_step=0.01,
        rep_names=None,
        dist_measures=("CKA",),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if rep_names is None:
            rep_names = ALL_REPRESENTATIONS.values()
        self.rep_names = rep_names
        if not num_samples:
            self.num_samples = len(self.sample_loader.sampler)
        else:
            self.num_samples = num_samples
        self.num_repeats = num_repeats
        self.dist_measures = dist_measures
        self.noise_stds = np.arange(0, noise_std_max, noise_std_step)
        if isinstance(self.model, IntermediateLayerGetter):
            self.model = self.model._model
        self.model = IntermediateLayerGetter(self.model, ALL_REPRESENTATIONS)
        self.accuracy = None
        self.tmp_path = os.path.join(
            self.base_path, "tmp" + make_hash(self.name)
        )
        self.num_batches = math.ceil(self.num_samples / self.sample_loader.batch_size)

    def run(self):
        noise_stabilities = {d: [] for d in self.dist_measures}
        for rep_name in self.rep_names:
            shutil.rmtree(self.tmp_path, ignore_errors=True, onerror=None)
            os.makedirs(self.tmp_path)
            self._reset_seed()
            self._compute_representation(rep_name)
            torch.cuda.empty_cache()
            for dist_measure in self.dist_measures:
                stability_matrix = self.compute_stability_matrix(dist_measure, rep_name)
                # compute stability measure
                noise_stabilities[dist_measure].append(
                    np.average(stability_matrix[0, :])
                )
                # prepare plotting
                stability_df = pd.DataFrame(stability_matrix)
                stability_df.astype(float)
                stability_df.columns = ["{:01.2f}".format(n) for n in self.noise_stds]
                stability_df.index = ["{:01.2f}".format(n) for n in self.noise_stds]
                fig, axs = plot_preparation(
                    nrows=2,
                    ncols=1,
                    fraction=0.5,
                    sharex=True,
                    ratio=(0.9, 1),  # to make it quadratic
                    gridspec_kw={"height_ratios": [1, 4]},
                    style="nips",
                )
                self.plot_acc_over_noise(axs[0])
                self.plot_matrix(
                    matrix_df=stability_df,
                    title=self.name + ": " + rep_name + "(" + dist_measure + ")",
                    min=0 if dist_measure == "CKA" else None,
                    max=1 if dist_measure == "CKA" else None,
                    save=None,
                    fig=fig,
                    axs=axs[1],
                    cbar_outside=True,
                )
                levels = np.arange(0, 1.0, 0.1)
                contours = axs[1].contour(
                    stability_matrix, colors="white", levels=levels
                )
                axs[1].clabel(contours, inline=True, fontsize=8)
                save_plot(
                    fig, self.get_file_name(dist_measure, rep_name.replace(".", "_"))
                )
                print("Finished {} {}-Analysis".format(rep_name, dist_measure))
            shutil.rmtree(self.tmp_path, ignore_errors=True, onerror=None)
        for dist_measure, stability in noise_stabilities.items():
            fig = self.plot_noise_stability(stability)
            save_plot(fig, self.get_file_name(dist_measure, "stability"))

    def _compute_representation(self, rep_name, *args, **kwargs):
        test_input = next(iter(self.sample_loader))[0][:1].to(self.device)
        test_out, _ = self.model(test_input)
        test_out = test_out[rep_name]
        if isinstance(test_out, list):
            print(rep_name, len(test_out))
            test_out = test_out[0]
        rep_size = test_out.flatten(1, -1).shape[-1]

        correct = torch.zeros((self.num_repeats, len(self.noise_stds)))
        for batch_idx, (inputs, targets) in enumerate(self.sample_loader):
            inputs, targets = (
                inputs.to(self.device, dtype=torch.float),
                targets.to(self.device),
            )
            self.batch_size = inputs.shape[0]
            reps = torch.zeros(
                (
                    # self.batch_size,
                    min(
                        self.batch_size, self.num_samples - batch_idx * self.batch_size,
                    ),
                    self.num_repeats,
                    len(self.noise_stds),
                    rep_size,
                )
            )
            for repeat in range(self.num_repeats):
                for noise_idx, noise_std in enumerate(self.noise_stds):
                    # apply noise
                    trainer = copy.deepcopy(self.experiment.trainer)
                    trainer.noise_std = {noise_std: 1.0}
                    module = NoiseAugmentation(
                        self.model,
                        config=trainer,
                        device=self.device,
                        data_loader={"img_classification": self.sample_loader},
                        seed=42,
                    )
                    inputs_ = inputs.clone()
                    model, inputs_ = module.pre_forward(
                        self.model, inputs_, {}, train_mode=False
                    )
                    # Forward
                    outputs = model(inputs_)
                    samples_start = 0
                    samples_end = min(
                        self.batch_size, self.num_samples - batch_idx * self.batch_size
                    )
                    # for rep_name in self.rep_names:
                    rep = outputs[0][rep_name].flatten(1, -1).detach().cpu()
                    reps[samples_start:samples_end, repeat, noise_idx] = rep.view(
                        (self.batch_size, -1)
                    )[: (samples_end - samples_start)]

                    # track accuracy
                    _, predicted = outputs[1].max(1)
                    targets_ = targets[: (samples_end - samples_start)]
                    predicted_ = predicted[: (samples_end - samples_start)]
                    correct[repeat, noise_idx] += predicted_.eq(targets_).sum().item()
            torch.save(
                reps,
                os.path.join(self.tmp_path, "reps_{}_{}".format(rep_name, batch_idx)),
            )

            if (batch_idx + 1) * self.batch_size >= self.num_samples:
                break

        self.accuracy = ((correct / self.num_samples) * 100).numpy()

    def plot_noise_stability(self, stability):
        df = pd.DataFrame(
            {"Layer": range(1, len(stability) + 1), "Stability": stability}
        )
        fig, axs = plot_preparation(nrows=1, ncols=1, fraction=0.5, style="nips",)
        g = sns.lineplot(x="Layer", y="Stability", data=df, ax=axs)
        xlabels = [str(int(x)) for x in g.get_xticks()]
        g.set_xticklabels(xlabels)
        return fig

    def plot_acc_over_noise(self, ax):
        df = pd.DataFrame(self.accuracy)
        df.astype(float)
        df.columns = ["{:01.2f}".format(n) for n in self.noise_stds]
        df = df.stack().reset_index()
        df.columns = ["repeat", "noise", "Accuracy"]
        sns.lineplot(x="noise", y="Accuracy", data=df, ax=ax)
        ax.set_ylim(ymin=0, ymax=100)

    def compute_stability_matrix(self, dist_measure, rep_name):
        result = self.load_matrix(dist_measure, rep_name.replace(".", "_"))
        if result is None:
            rep_pieces = [
                torch.load(
                    os.path.join(
                        self.tmp_path, "reps_{}_{}".format(rep_name, batch_idx)
                    )
                )
                for batch_idx in range(self.num_batches)
            ]
            rep = torch.cat(rep_pieces)
            result = torch.zeros((len(self.noise_stds), len(self.noise_stds)))
            first_loop = (
                range(self.num_repeats - 1) if self.num_repeats > 1 else range(1)
            )
            for r in first_loop:
                second_loop = (
                    range(r + 1, self.num_repeats) if self.num_repeats > 1 else range(1)
                )
                for r2 in second_loop:
                    reps1 = rep[:, r].to(self.device)
                    reps2 = rep[:, r2].to(self.device)
                    for i in range(len(self.noise_stds)):
                        for j in range(i, len(self.noise_stds)):
                            res = similarity(reps1[:, i], reps2[:, j], dist_measure)
                            result[i, j] += res.detach().cpu()
                            if j != i:
                                result[j, i] += res.detach().cpu()
                            res = similarity(reps2[:, i], reps1[:, j], dist_measure)
                            result[i, j] += res.detach().cpu()
                            if j != i:
                                result[j, i] += res.detach().cpu()
                    del reps1
                    del reps2
            result = (
                (
                    result / (2 * (self.num_repeats * (self.num_repeats - 1) / 2))
                    if self.num_repeats > 1
                    else result
                )
                .detach()
                .numpy()
            )
            self.save_matrix(result, dist_measure, rep_name.replace(".", "_"))
        return result
