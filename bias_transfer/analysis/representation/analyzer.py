import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from torch.backends import cudnn

import bias_transfer.trainer.trainer
from bias_transfer.analysis.utils import plot_preparation, save_plot


class RepresentationAnalyzer:
    def __init__(
        self,
        experiment,
        table,
        name: str,
        dataset: str = "val",
        base_path: str = "/work/analysis/",
    ):
        self.experiment = experiment
        self.dataset = dataset
        # data_loaders, self.model, self.trainer = (
        #                                                  table & experiment.get_restrictions()
        #                                          ).restore_saved_state(,,
        #                                          self.num_samples = -1
        # self.sample_loader = torch.utils.data.DataLoader(
        #     data_loaders[dataset]["img_classification"].dataset,
        #     sampler=data_loaders[dataset]["img_classification"].sampler,
        #     batch_size=64,
        #     shuffle=False,
        #     num_workers=1,
        #     pin_memory=False,
        # )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self._reset_seed()
        self.criterion = nn.CrossEntropyLoss()
        self.base_path = base_path
        self.name = name

    def _reset_seed(self):
        torch.manual_seed(42)
        np.random.seed(42)
        if self.device == "cuda":
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.cuda.manual_seed(42)

    def _compute_representation(self, main_loop_modules):
        (
            acc,
            loss,
            module_losses,
            collected_outputs,
        ) = bias_transfer.trainer.main_loop.main_loop(
            self.model,
            self.criterion,
            self.device,
            None,
            self.sample_loader,
            0,
            main_loop_modules,
            train_mode=False,
            return_outputs=True,
        )
        outputs = [o[self.rep_name] for o in collected_outputs]
        print("Acc:", acc, "Loss:", loss, flush=True)
        return torch.cat(outputs), acc

    def get_file_name(self, method, rep_name):
        return os.path.join(self.base_path, "_".join([self.name, rep_name, method]))

    def save_matrix(self, to_save, method, rep_name):
        name = self.get_file_name(method, rep_name) + ".npy"
        if not os.path.isdir(self.base_path):
            os.mkdir(self.base_path)
        np.save(os.path.join(self.base_path, name), to_save)

    def load_matrix(self, method, rep_name):
        name = self.get_file_name(method, rep_name) + ".npy"
        file = os.path.join(self.base_path, name)
        if os.path.isfile(file):
            print("Found existing {} result that will be loaded now".format(method))
            return np.load(file)
        return None

    def plot_matrix(
        self,
        matrix_df,
        title,
        fig=None,
        axs=None,
        save="",
        min=None,
        max=None,
        cbar_outside=True,
    ):
        if not fig or not axs:
            fig, axs = plot_preparation(ratio=(4, 4), style="nips")
        fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
        if cbar_outside:
            cbar_ax = fig.add_axes([0.90, 0.2, 0.02, 0.4])  # [left, bottom, width, height]
        sns.heatmap(
            matrix_df,
            cmap="YlGnBu",
            xticklabels=10,
            yticklabels=10,
            vmin=min,
            vmax=max,
            ax=axs,
            cbar=True,
            cbar_ax=cbar_ax if cbar_outside else None,
        )
        sns.despine(offset=10, trim=True)
        if cbar_outside:
            fig.tight_layout(rect=[0, 0, 0.9, 1])
        else:
            fig.tight_layout()

        st = fig.suptitle(title, fontsize=12)
        st.set_y(1.05)
        if save:
            save_plot(fig,save)
