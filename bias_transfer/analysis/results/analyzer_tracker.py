import math
import re
from functools import partial

import torch
import torch.backends.cudnn as cudnn

from bias_transfer.analysis.utils import plot_preparation, save_plot
from bias_transfer.tables.transfer import TransferredTrainedModel
from bias_transfer.utils.io import load_checkpoint
import numpy as np
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering
from bias_transfer.tables.nnfabrik import *
from mlutils.tracking import AdvancedMultipleObjectiveTracker as Tracker

corruption_map = {
    "shot_noise": "Shot Noise",
    "impulse_noise": "Impulse Noise",
    "speckle_noise": "Speckle Noise",
    "gaussian_noise": "Gaussian Noise",
    "defocus_blur": "Defocus Blur",
    "gaussian_blur": "Gauss Blur",
    "motion_blur": "Motion Blur",
    "glass_blur": "Glass Blur",
    "zoom_blur": "Zoom Blur",
    "brightness": "Brightness",
    "fog": "Fog",
    "frost": "Frost",
    "snow": "Snow",
    "contrast": "Contrast",
    "elastic_transform": "Elastic Transform",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG Compression",
    "saturate": "Saturate",
    "spatter": "Spatter",
}

Res_Alex_Net_mean = dict()
Res_Alex_Net_mean["Gaussian Noise"] = 0.886
Res_Alex_Net_mean["Shot Noise"] = 0.894
Res_Alex_Net_mean["Impulse Noise"] = 0.923
Res_Alex_Net_mean["Defocus Blur"] = 0.820
Res_Alex_Net_mean["Gauss Blur"] = 0.826
Res_Alex_Net_mean["Glass Blur"] = 0.826
Res_Alex_Net_mean["Motion Blur"] = 0.786
Res_Alex_Net_mean["Zoom Blur"] = 0.798
Res_Alex_Net_mean["Snow"] = 0.867
Res_Alex_Net_mean["Frost"] = 0.827
Res_Alex_Net_mean["Fog"] = 0.819
Res_Alex_Net_mean["Brightness"] = 0.565
Res_Alex_Net_mean["Contrast"] = 0.853
Res_Alex_Net_mean["Elastic Transform"] = 0.646
Res_Alex_Net_mean["Pixelate"] = 0.718
Res_Alex_Net_mean["JPEG Compression"] = 0.607
Res_Alex_Net_mean["Speckle Noise"] = 0.845
Res_Alex_Net_mean["Spatter"] = 0.718
Res_Alex_Net_mean["Saturate"] = 0.658

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Analyzer:
    def __init__(self):
        self.data = {}

    def load_data(self, configs, transfer_levels=(0,)):
        # Select data:
        for description, config in configs.items():
            for level in transfer_levels:
                restriction = config.get_restrictions(level)
                if restriction:
                    restricted = TransferredTrainedModel() & restriction
                else:
                    restricted = None
                if restricted:  # could be empty if entry is not computed yet
                    fetch_res = restricted.fetch1("output")
                    if fetch_res:  # could be a data generation step (no output)
                        if description not in self.data:
                            self.data[description] = {
                                level: Tracker.from_dict(fetch_res)
                            }
                        else:
                            self.data[description][level] = Tracker.from_dict(fetch_res)

    def plot_training_progress(self, dataset="Validation"):
        self.plot(
            to_plot=((dataset, "img_classification", "accuracy")), plot_method="line"
        )

    def plot_noise_eval(self, std=True, bn_train=False, **kwargs):
        bn_train = " BN=Train" if bn_train else ""
        if std:
            to_plot = (
                (
                    "Noise noise_std 0.0_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.05_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.1_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.2_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.3_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.5_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 1.0_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
            )
        else:
            to_plot = (
                (
                    "Noise noise_std 0.0_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.05_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.1_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.2_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.3_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 0.5_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
                (
                    "Noise noise_std 1.0_1.0" + bn_train,
                    "img_classification",
                    "accuracy",
                ),
            )

        def rename(name):
            number_idx = re.search(r"\d", name)
            name = name[number_idx.start() :]
            underscore_idx = name.find("_")
            name = name[:underscore_idx]
            return float(name)

        self.plot(to_plot, plot_method="bar", rename=rename, **kwargs)

    def generate_table(
        self,
        objective=("Test", "img_classification", "accuracy"),
        last_n=0,
        label_steps=False,
    ):
        row_list = []
        for desc, results in self.data.items():
            if label_steps:
                name_split = desc.name.split(" ")
                name = " ".join(name_split[:-1])
                labels = name_split[-1][1:-1].split(";")
            else:
                name, labels = (desc.name, None)
            row = {"name": name}
            levels = sorted(list(results.keys()))
            if last_n:
                levels = levels[(-1) * last_n :]
            for level, tracker in results.items():
                try:
                    if level in levels:
                        l = levels.index(level)
                        if labels:
                            l = labels[l]
                        row[l] = tracker.get_current_objective(*objective)
                except:
                    pass  # no valid entry for this objective
            row_list.append(row)
        df = pd.DataFrame(row_list).groupby("name").first()
        return df

    def generate_normalized_table(self):
        df = self.generate_table(last_n=2,label_steps=True)
        for i, c in enumerate(df.columns):
            offset = "A" if i % 2 == 0 else "B"
            baseline = df.at[f"Direct Training {offset}", c]
            df.insert(2 * i + 1, c + " normalized", df[c].divide(baseline).multiply(100))
        return df

    def plot_frontier(self, save="", style="lighttalk", legend_outside=True):
        df = self.generate_table(last_n=2, label_steps=True)
        columns = df.columns[1:]
        fig, ax = plot_preparation(style, nrows=2, ncols=2)
        for i, c in enumerate(columns):
            if i % 2 == 1:
                sns.scatterplot(
                    data=df,
                    x=columns[i - 1],
                    y=c,
                    hue="name",
                    ax=ax[(i-1)//4][((i-1)%4)//2],
                    legend="brief" if i==7 else False,
                )

        sns.despine(offset=10, trim=True)
        plt.subplots_adjust(hspace=0.3)
        if "talk" in style:
            if legend_outside:
                plt.legend(
                    fontsize=14,
                    title_fontsize="14",
                    bbox_to_anchor=(1.05, 1),
                    loc=2,
                    borderaxespad=0.0,
                )
            else:
                plt.legend(fontsize=14, title_fontsize="14")
        elif legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if save:
            save_plot(
                fig,
                save + "_" + style,
                types=("png", "pdf", "pgf") if "nips" in style else ("png",),
            )

    def plot(
        self,
        to_plot=(),
        plot_method="bar",
        save="",
        style="lighttalk",
        legend_outside=True,
        rename=lambda x: x,
    ):
        if not to_plot in ("c_test_eval", "c_test_loss"):
            fig, ax = plot_preparation(style)
        # Plot
        if plot_method == "bar" and not isinstance(to_plot[0], tuple):
            row_list = []
            for desc, tracker in self.data.items():
                row_list.append(
                    {
                        "name": desc.name,
                        to_plot[-1]: tracker.get_current_objective(*to_plot),
                    }
                )
            df = pd.DataFrame(row_list)
            sns.barplot(x="name", y=to_plot[-1], data=df, ax=ax)
        elif plot_method == "bar":  # compare a number of datasets
            row_list = []
            for desc, tracker in self.data.items():
                row = {"name": desc.name}
                for key in to_plot:
                    row[rename(key[0])] = tracker.get_current_objective(*key)
                row_list.append(row)
            df = pd.DataFrame(row_list)
            df.index = df.name
            del df["name"]
            df = df.stack().reset_index()
            df.columns = ["Training", "Level", to_plot[0][-1]]
            sns.lineplot(x="Level", data=df, y=to_plot[0][-1], hue="Training", ax=ax)
        elif plot_method == "grid":
            data_to_plot = self.extract_c_test_results()
            g = sns.FacetGrid(
                data=data_to_plot,
                col="Corruption",
                col_wrap=4,
                sharey=True,
                sharex=True,
                # height=4
            )

            def draw_heatmap(data, *args, **kwargs):
                del data["Corruption"]
                # print(data)
                sns.heatmap(data, annot=True, cbar=False)

            g.map_dataframe(draw_heatmap)
            fig = g.fig
        elif plot_method == "line":
            row_list = []
            for desc, tracker in self.data.items():
                row = {"name": desc.name, to_plot[-1]: tracker.get_objective(*to_plot)}
                row_list.append(row)
            df = pd.DataFrame(row_list)
            df.index = df.name
            del df["name"]
            df = df[to_plot[-1]].apply(pd.Series)
            df = df.stack().reset_index()
            df.columns = ["Training", "Epoch", to_plot[-1]]
            sns.lineplot(x="Epoch", y=to_plot[-1], hue="Training", data=df, ax=ax)
        else:
            print("Unknown plot option!")

        sns.despine(offset=10, trim=True)
        if to_plot in ("c_test_eval", "c_test_loss"):
            # remove ticks again (see: https://stackoverflow.com/questions/37860163/seaborn-despine-brings-back-the-ytick-labels)
            # loop over the non-left axes:
            for i, ax in enumerate(g.axes.flat):
                if i % 4 != 0:
                    # get the yticklabels from the axis and set visibility to False
                    for label in ax.get_yticklabels():
                        label.set_visible(False)
                    ax.yaxis.offsetText.set_visible(False)
                if i < len(g.axes) - 4:
                    # get the xticklabels from the axis and set visibility to False
                    for label in ax.get_xticklabels():
                        label.set_visible(False)
                    ax.xaxis.offsetText.set_visible(False)
        if "talk" in style:
            if legend_outside:
                plt.legend(
                    fontsize=14,
                    title_fontsize="14",
                    bbox_to_anchor=(1.05, 1),
                    loc=2,
                    borderaxespad=0.0,
                )
            else:
                plt.legend(fontsize=14, title_fontsize="14")
        elif legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if save:
            save_plot(
                fig,
                save + "_" + style,
                types=("png", "pdf", "pgf") if "nips" in style else ("png",),
            )

    def extract_c_test_results(self):
        corruptions = (
            "shot_noise",
            "impulse_noise",
            # "speckle_noise",
            "gaussian_noise",
            "defocus_blur",
            # "gaussian_blur",
            "motion_blur",
            "glass_blur",
            "zoom_blur",
            "brightness",
            "fog",
            "frost",
            "snow",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            # "saturate",
            # "spatter",
        )
        data_to_plot = pd.DataFrame()
        for corruption in corruptions:
            row_list = []
            for desc, tracker in self.data.items():
                row = {
                    severity: tracker.get_current_objective(
                        corruption, str(severity), "accuracy"
                    )
                    for severity in range(1, 6)
                }
                row[0] = tracker.get_current_objective(
                    "Test", "img_classification", "accuracy"
                )
                row["name"] = desc.name
                row_list.append(row)
            df = pd.DataFrame(row_list)
            df = df.groupby("name").mean()
            df["Corruption"] = corruption
            data_to_plot = pd.concat([data_to_plot, df], axis=0, sort=True)
        return data_to_plot

    def calculate_c_scores(self):
        c_data = self.extract_c_test_results()
        df = c_data[c_data.columns[0:6]].apply(lambda x: 100 - x)
        df_mean = df[df.columns[0:6]].mean(axis=1)
        c_data = pd.concat([c_data, df_mean], axis=1)
        c_data.columns = [1, 2, 3, 4, 5, 0, "Corruption", "Mean"]

        def normalize_alexnet(row):
            mean_error = row["Mean"]
            corruption = row["Corruption"]
            ce = mean_error / Res_Alex_Net_mean[corruption_map[corruption]]
            return pd.concat([row, pd.Series({"mCE": ce})])

        c_data = c_data.apply(normalize_alexnet, axis=1)
        c_data = c_data.groupby("name").mean()
        return c_data


def print_table_for_excel(table):
    prior_columns = 1
    keys = []
    for key in table.fetch(as_dict=True)[0].keys():
        if "comment" in key:
            keys.append(key)
        if key == "transfer_trainer_hash" or key == "transfer_trainer_config":
            keys.append(key)
            prior_columns = 3
    # heading
    row = table.fetch("output", as_dict=True)[0]["output"][1]["dev_noise_acc"]
    print("," * prior_columns, end="")
    for key in row.keys():
        print(key + ("," * (len(row[key]))), end="")
    print()
    print("," * prior_columns, end="")
    for key in row.keys():
        for sub_key in row[key].keys():
            print(sub_key.split("_")[0], end=",")
    print()
    # content
    for row in table.fetch("output", *keys, as_dict=True):
        comment = []
        extra = []
        for k, v in row.items():
            if k == "output" or "hash" in k:
                continue
            elif "config" in k:
                extra = [
                    "freeze_{}".format(v["freeze"]),
                    "reset_{}".format(v["reset_linear"]),
                ]
            else:
                comment.append(v)
        print(".".join(comment), end=", ")
        if extra:
            print(",".join(extra), end=", ")
        output = row["output"]
        final_results = output[1]["dev_noise_acc"]
        for key in final_results.keys():
            for res in final_results[key].items():
                print(res[1], end=",")
        print()
