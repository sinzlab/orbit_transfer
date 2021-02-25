import string

from .plot import plot
import json
import pickle as pkl
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_robustness(
    models,
    folder_name,
    test_set,
    levels,
    std=False,
    name_map={},
    noises=[],
    plot_overview=False,
    plot_correlation=False,
    plot_individual=True,
    **kwargs,
):

    if plot_individual:
        means = {}
        stds = {}
        for model in models:
            if std:
                with open(
                    "./{}/{}_stds_{}.json".format(folder_name, model, test_set), "r"
                ) as fp:
                    data = json.load(fp)
                stds[model] = data
            else:
                with open(
                    "./{}/{}_stderrs_{}.json".format(folder_name, model, test_set), "r"
                ) as fp:
                    data = json.load(fp)
                stds[model] = data
            with open(
                "./{}/{}_means_{}.json".format(folder_name, model, test_set), "r"
            ) as fp:
                data = json.load(fp)
            means[model] = data
        for k in stds:
            stds[k] = stds[k]["model"]
            means[k] = means[k]["model"]
    else:
        stds = None
        means = None

    if plot_correlation:
        with open(f"./{folder_name}/corruption_scores" + ".pkl", "rb") as f:
            corrupt_scores_loaded = pkl.load(f)
        with open(f"./{folder_name}/neural_performances" + ".pkl", "rb") as f:
            neural_performances = pkl.load(f)

        br_convert_key = {
            "tin_mtl": 1,
            "tin_mtl_ratio_2": 2,
            "tin_mtl_ratio_1_2": -2,
            "tin_mtl_ratio_1_3": -3,
            "tin_mtl_ratio_1_4": -4,
            "tin_mtl_ratio_1_5": -5,
            "tin_mtl_ratio_1_7": -7,
            "tin_mtl_ratio_1_10": -10,
            "tin_mtl_ratio_1_15": -15,
        }
        corrupt_list = []
        neurals = []
        imgcls = []
        brs = []
        for key in corrupt_scores_loaded.keys():
            if key != "tin_baseline":
                corrupt_list.append(corrupt_scores_loaded[key])
                # brs.append(br_convert_key[key])
                neurals.append(neural_performances[key])
        corrupt_list = np.array(corrupt_list) * 100

        robustness_data = pd.DataFrame(
            {
                "neural performance": neurals,
                "robustness score %": corrupt_list,
                "category": ["MTL" for _ in neurals],
            }
        )
    else:
        robustness_data = None

    if plot_overview:
        overview_data = {
            "tin_baseline": 1.0,
            "tin_mtl": 1.1623512415052255,
            "tin_mtl_shuffled": 0.9624131504344579,
            "tin_smtl": 1.0529372552208522,
        }
        overview_data = {name_map[k]: v *100 for k, v in overview_data.items()}
    else:
        overview_data = None

    _plot(
        noises=noises,
        means=means,
        stds=stds,
        levels_list=levels,
        name_map=name_map,
        robustness_data=robustness_data,
        robustness_overview=overview_data,
        **kwargs,
    )


def name_change(old_name, prefix=""):
    name = old_name.replace("->", " → ")
    name = name.replace("_", " ")
    name = " ".join([n.capitalize() for n in name.split()])
    return prefix + name


@plot
def _plot(
    fig,
    ax,
    noises,
    means,
    stds,
    levels_list,
    name_map,
    robustness_data,
    robustness_overview,
):
    colors = {
        "Baseline": "#000000",
        "MTL Oracle": "#2578B3",
        "Oracle": "#A6CEE3",
        "MTL shuffled": "#FB9A99",
        "MTL": "#E31E1B",
    }

    row, col = 0, 0
    if means is not None:
        for i, cat in enumerate(noises):
            levels = pd.DataFrame(columns=["model", "category", "level", "mean", "std"])
            for model in means.keys():
                means_ordered = {
                    float(level): v for level, v in means[model][cat].items()
                }
                stds_ordered = {
                    float(level): v for level, v in stds[model][cat].items()
                }
                levels = levels.append(
                    pd.DataFrame(
                        {
                            "model": name_map[model],
                            "category": cat,
                            "level": list(means_ordered.keys())[:-1],
                            "mean": list(means_ordered.values())[:-1],
                            "std": list(stds_ordered.values())[:-1],
                        }
                    )
                )
            d = levels.groupby("category").get_group(cat)
            d_mean = d.pivot(index="level", columns="model", values=["mean", "std"])
            plot = d_mean["mean"].plot(
                ax=ax[row][col], legend=False, yerr=d_mean["std"], color=colors
            )
            ax[row][col].set_title(name_change(cat))
            plot.set_xticks(levels_list)
            plot.set_xticklabels(levels_list)
            plot.set_xlabel("Corruption Severity")
            if col == 0:
                plot.set_ylabel("Accuracy [%]")

            ax[row][col].grid(True, linestyle=":")

            col = (col + 1) % len(ax[row])
            if col == 0:
                row += 1

        handles, labels = ax[0][0].get_legend_handles_labels()
        new_labels = ["Baseline", "MTL", "MTL shuffled", "MTL Oracle","Oracle"]
        new_handles = []
        for label in new_labels:
            new_handles.append(handles[labels.index(label)])
        fig.legend(new_handles, new_labels, loc=(0.05, 0.90), ncol=6, frameon=False)
        fig.tight_layout()

    if robustness_data is not None:
        plot = sns.scatterplot(
            data=robustness_data,
            x="neural performance",
            y="robustness score %",
            # sizes=(300, 900),
            ax=ax[row][col],
            hue="category",
            palette=colors,
            legend=False,
        )
        plot.set_xlabel("Neural Prediction [corr]")
        plot.set_ylabel("Robustness Score [%]")

        ax[row][col].grid(True, linestyle=":")
        col = (col + 1) % len(ax[row])
        if col == 0:
            row += 1

    if robustness_overview is not None:
        plot = sns.barplot(
            x=list(robustness_overview.keys()),
            y=list(robustness_overview.values()),
            # sizes=(300, 900),
            ax=ax[row][col],
            palette=colors,
        )
        plot.set_xlabel("")
        plot.set_ylabel("Robustness Score [%]")

        ax[row][col].grid(True, linestyle=":")
        ax[row][col].set_ylim([50, 120])
        fig.tight_layout()
        sns.despine(offset=3, trim=False)
        plt.setp(ax[row][col].xaxis.get_majorticklabels(), rotation=30, ha="right")


