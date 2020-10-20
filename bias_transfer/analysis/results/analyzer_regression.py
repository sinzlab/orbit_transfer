import math
import re
from functools import partial

import torch
import torch.backends.cudnn as cudnn

from bias_transfer.analysis.utils import plot_preparation, save_plot
from bias_transfer.gp.nn_kernel import nn_kernel
from bias_transfer.utils.io import load_checkpoint
import numpy as np
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering
from nnfabrik.main import *
from mlutils.tracking import AdvancedMultipleObjectiveTracker as Tracker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self):
        self.data_loaders = {}
        self.model = None
        self.trainer = None

    def load_model(self, config, table, transfer_level):
        # Select data:
        if transfer_level < len(config.get_restrictions()):
            restricted = table & config.get_restrictions()[transfer_level]
        else:
            print("Nothing to load")
            restricted = None
        if restricted:  # could be empty if entry is not computed yet
            self.data_loaders, self.model, self.trainer = restricted.load_model(
                include_dataloader=True, include_trainer=True, include_state_dict=True
            )

    def plot_eval(self, save=""):
        self.model.eval()
        x_test, y_test = self.data_loaders["test"]["regression"].dataset.tensors
        x_train, y_train = self.data_loaders["train"]["regression"].dataset.tensors
        plt.plot(x_test, y_test, color="orange", lw=2, label="True")
        plt.plot(x_train, y_train, color="red", label="Traning data")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        prediction = self.model(x_test.to(device))  # input x and predict based on x
        if isinstance(prediction, tuple):
            prediction = prediction[1]
        plt.plot(x_test, prediction.detach().cpu().numpy(), label="Prediction")
        plt.legend()
        if save:
            fig = plt.gcf()
            fig.savefig(save, dpi=200)

    def plot_kernel(self):
        self.model.eval()
        x_test, y_test = self.data_loaders["test"]["regression"].dataset.tensors
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        K_plot = nn_kernel(x_test, x_test, net=self.model, device=device)
        plt.imshow(K_plot)
        # if np.count_nonzero(x) > 0:
        #     _ = plt.xticks(np.arange(0,x.shape[0], 15),x[::15,0].astype(np.int))
        #     _ = plt.yticks(np.arange(0,x.shape[0], 15),x[::15,0].astype(np.int))
        plt.colorbar()
