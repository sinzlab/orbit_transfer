import math
import string

from .plot import plot
import json
import pickle as pkl
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as stats_model
import statsmodels.formula.api as smf
import matplotlib.transforms


def plot_robustness(
    models,
    folder_name,
    test_set,
    levels,
    std=False,
    name_map={},
    noises=[],
    noise_grouping={},
    plot_overview=False,
    plot_correlation=False,
    plot_correlation_bootstrapped=False,
    plot_individual=True,
    plot_per_noise_robustness=False,
    add_overview_to_groups=False,
    **kwargs,
):

    if plot_individual:
        means = {}
        stds = {}
        for model in models:
            with open(
                f"./{folder_name}/{model}_all_seeds_{test_set}_bootstrapped_stds.json",
                "r",
            ) as fp:
                data = json.load(fp)
            stds[model] = data
            with open(
                f"./{folder_name}/{model}_all_seeds_{test_set}_bootstrapped_means.json",
                "r",
            ) as fp:
                data = json.load(fp)
            means[model] = data
        for k in stds:
            stds[k] = stds[k]["model"]
            means[k] = means[k]["model"]
    else:
        stds = None
        means = None
    models = models[1:]

    if plot_per_noise_robustness:
        with open(
            f"./{folder_name}/robust_scores_per_noise.json",
            "r",
        ) as fp:
            per_noise_data_ = json.load(fp)
        per_noise_data_ = {
            name_map[k]: {
                k2: {"mean": v2["mean"] * 100, "std": v2["std"] * 100}
                for k2, v2 in v.items()
            }
            for k, v in per_noise_data_.items()
        }
        per_noise_data = {}
        if noise_grouping:
            for model in models:
                model = name_map[model]
                results = per_noise_data_[model]
                per_noise_data[model] = {}
                for group_name, group_items in noise_grouping.items():
                    per_noise_data[model][name_change(group_name)] = {
                        "mean": 0,
                        "std": 0,
                    }
                    for noise in group_items:
                        per_noise_data[model][name_change(group_name)][
                            "mean"
                        ] += results[noise]["mean"]
                        per_noise_data[model][name_change(group_name)]["std"] += (
                            results[noise]["std"] ** 2
                        )
                    per_noise_data[model][name_change(group_name)]["mean"] /= len(
                        group_items
                    )
                    per_noise_data[model][name_change(group_name)]["std"] = math.sqrt(
                        per_noise_data[model][name_change(group_name)]["std"]
                        / len(group_items)
                    )

        else:
            for model in models:
                model = name_map[model]
                results = per_noise_data_[model]
                per_noise_data[model] = {
                    name_change(k): v
                    for k, v in sorted(
                        results.items(), key=lambda item: noises.index(item[0])
                    )
                }
    else:
        per_noise_data = {}

    if plot_correlation_bootstrapped:
        with open(f"./{folder_name}/mtl_bootstrapped" + ".pkl", "rb") as f:
            clean_and_neural = pkl.load(f)
        with open(
            f"./{folder_name}/robust_scores_scatter_bootstrapped.json", "r"
        ) as fp:
            robust_scores = json.load(fp)
        for br, scores in robust_scores.items():
            scores.update(clean_and_neural[int(br)])
            robust_scores[br] = scores
        print("ROBUST", robust_scores)
        corrupt_list = []
        corrupt_err_list = []
        neurals = []
        neurals_err_list = []
        imgcls = []
        for key in robust_scores.keys():
            corrupt_list.append(robust_scores[key]["mean_score"])
            corrupt_err_list.append(robust_scores[key]["score_standard_err"])
            neurals.append(robust_scores[key]["mean_neural"])
            neurals_err_list.append(robust_scores[key]["std_neural"])
            imgcls.append(robust_scores[key]["mean_cls"])
        corrupt_list = np.array(corrupt_list) * 100
        corrupt_err_list = np.array(corrupt_err_list) * 100

        robustness_data = pd.DataFrame(
            {
                "Neural": neurals,
                "Robustness": corrupt_list,
                "robustness err": corrupt_err_list,
                "neural err": neurals_err_list,
                "Clean": imgcls,
                "category": ["MTL" for _ in neurals],
            }
        )
    elif plot_correlation:
        with open(f"./{folder_name}/mtl" + ".pkl", "rb") as f:
            clean_and_neural = pkl.load(f)
        with open(f"./{folder_name}/robust_scores_scatter_seeds.json", "r") as fp:
            robust_scores = json.load(fp)
        for br, scores in robust_scores.items():
            scores.update(clean_and_neural[int(br)])
            robust_scores[br] = scores
        print("ROBUST", robust_scores)
        corrupt_list = []
        neurals = []
        imgcls = []
        for key in robust_scores.keys():
            corrupt_list += robust_scores[key]["scores"]
            neurals += robust_scores[key]["neural"]
            imgcls += robust_scores[key]["mean_cls"]
        corrupt_list = np.array(corrupt_list) * 100

        robustness_data = pd.DataFrame(
            {
                "Neural": neurals,
                "Robustness": corrupt_list,
                "Clean": imgcls,
                "category": ["MTL" for _ in neurals],
            }
        )
    else:
        robustness_data = None

    if plot_overview or add_overview_to_groups:
        overview_data = {
            "tin_baseline": {"mean": 1.0, "std": 0.0},
            "tin_mtl": {
                "mean": 1.141074196151724,
                "std": 0.04030340549401311,
            },
            "tin_mtl_shuffled": {
                "mean": 0.9632811867144085,
                "std": 0.027942464073949062,
            },
            "tin_mtl_simulated": {
                "mean": 1.2174372498105903,
                "std": 0.044304298544084894,
            },
            "tin_oracle": {
                "mean": 1.2374721551434498,
                "std": 0.037550730118767965,
            },
        }
        overview_data = {
            name_map[k]: {k2: v2 * 100 for k2, v2 in v.items()}
            for k, v in overview_data.items()
        }
        if add_overview_to_groups and per_noise_data:
            for model, results in overview_data.items():
                per_noise_data[model]["Mean"] = results
            overview_data = {}
    else:
        overview_data = {}

    _plot(
        noises=noises,
        means=means,
        stds=stds,
        levels_list=levels,
        name_map=name_map,
        robustness_data=robustness_data,
        robustness_overview=overview_data,
        robustness_per_noise=per_noise_data,
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
    robustness_per_noise,
):
    colors = {
        "Baseline": "#000000",
        "MTL-Oracle": "#2578B3",
        "Oracle": "#A6CEE3",
        "MTL-Shuffled": "#FB9A99",
        "MTL-Monkey": "#E31E1B",
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
                            "level": list(means_ordered.keys()),
                            "mean": list(means_ordered.values()),
                            "std": list(stds_ordered.values()),
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
            if row == len(ax)-1:
                plot.set_xlabel("Corruption Severity")
            else:
                plot.set_xlabel(None)
            if col == 0:
                plot.set_ylabel("Accuracy [%]")

            # ax[row][col].set_ylim([0, 50])
            ax[row][col].grid(True, linestyle=":")

            col = (col + 1) % len(ax[row])
            if col == 0:
                row += 1

        ax[-1][-1].axis("off")

        handles, labels = ax[0][0].get_legend_handles_labels()
        new_labels = ["Baseline", "MTL-Monkey", "MTL-Shuffled", "MTL-Oracle", "Oracle"]
        new_handles = []
        for label in new_labels:
            new_handles.append(handles[labels.index(label)])
        fig.legend(new_handles, new_labels, loc=(0.05, 0.96), ncol=6, frameon=False)
        fig.tight_layout()

    if robustness_per_noise:
        plot_per_category(
            ax,
            col,
            colors,
            fig,
            robustness_per_noise,
            row,
            despine=True,
        )
        ax[row][col].set_ylabel("Robustness Score [%]")
        box = ax[row][col].get_position()
        box.x0 = box.x0 - 0.035
        box.x1 = box.x1 - 0.035
        ax[row][col].set_position(box)
        col = (col + 1) % len(ax[row])
        if col == 0:
            row += 1

    if robustness_overview:
        if robustness_per_noise:
            overview_data = {
                m: {"Total": result} for m, result in robustness_overview.items()
            }
            del overview_data["Baseline"]
            plot_per_category(ax, col, colors, fig, overview_data, row, despine=False)
            ax[row][col].set_ylabel("")
            ax[row][col].axes.get_yaxis().set_visible(False)
            box = ax[row][col].get_position()
            box.x0 = box.x0 - 0.065
            box.x1 = box.x1 - 0.065
            ax[row][col].set_position(box)
            col = (col + 1) % len(ax[row])
            if col == 0:
                row += 1
        else:
            ax[row][col].axhline(y=100, color=colors["Baseline"], label="Baseline")
            plot = sns.barplot(
                x=list(robustness_overview.keys())[1:],
                y=[y["mean"] for y in robustness_overview.values()][1:],
                yerr=[y["std"] for y in robustness_overview.values()][1:],
                ax=ax[row][col],
                palette=colors,
            )
            plot.set_xlabel("")
            plot.set_ylabel("Robustness Score [%]")

            ax[row][col].grid(True, linestyle=":")
            # ax[row][col].set_ylim([50, 140])
            fig.tight_layout()
            sns.despine(offset=3, trim=False)
            # plt.setp(ax[row][col].xaxis.get_majorticklabels(), rotation=30, ha="right")
            plt.setp(ax[row][col].xaxis.get_majorticklabels(), rotation=-40, ha="left")
            # Create offset transform by 5 points in x direction
            dx = -1 / 72.0
            dy = 0 / 72.0
            offset = matplotlib.transforms.ScaledTranslation(
                dx, dy, fig.dpi_scale_trans
            )
            # apply offset transform to all x ticklabels.
            for label in ax[row][col].xaxis.get_majorticklabels():
                label.set_transform(label.get_transform() + offset)
            col = (col + 1) % len(ax[row])
            if col == 0:
                row += 1

    if robustness_data is not None:
        if "robustness err" in robustness_data.columns:
            markers, caps, bars = ax[row][col].errorbar(
                robustness_data["Neural"],
                robustness_data["Robustness"],
                yerr=robustness_data["robustness err"],
                xerr=robustness_data["neural err"],
                linestyle="None",
                zorder=-32,
            )
            # loop through bars and caps and set the alpha value
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        m, b = np.polyfit(robustness_data["Neural"], robustness_data["Robustness"], 1)
        ax[row][col].plot(robustness_data["Neural"], m * robustness_data["Neural"] + b, color="grey")
        # Get significance values for correlation:
        mod = smf.ols(formula="Robustness ~ Clean * Neural", data=robustness_data)
        res = mod.fit()
        print("Robustness ~ Clean * Neural")
        print(res.summary())

        mod = smf.ols(formula="Robustness ~ Clean + Neural", data=robustness_data)
        res = mod.fit()
        print("Robustness ~ Clean + Neural")
        print(res.summary())
        for i in range(3):
            print(res.pvalues[i])

        plot = sns.scatterplot(
            data=robustness_data,
            x="Neural",
            y="Robustness",
            # sizes=(300, 900),
            ax=ax[row][col],
            hue="Clean",
            palette="rocket_r",
            legend=False,
        )

        norm = plt.Normalize(
            robustness_data["Clean"].min(),
            robustness_data["Clean"].max(),
        )
        sm = plt.cm.ScalarMappable(cmap="rocket_r", norm=norm)
        sm.set_array([])
        cbar = ax[row][col].figure.colorbar(sm)
        cbar.set_label("Clean Accuracy [%]", rotation=270, labelpad=10)

        plot.set_xlabel("Neural Prediction [corr]")
        plot.set_ylabel("Robustness Score [%]")

        ax[row][col].grid(True, linestyle=":")

        # fig.tight_layout()


def plot_per_category(ax, col, colors, fig, robustness_per_noise, row, despine=False):
    df = pd.concat(
        {k: pd.DataFrame(v).T for k, v in robustness_per_noise.items()}, axis=0
    )
    df.reset_index(inplace=True)
    df.columns = ["Model", "Corruption", "Robustness", "std"]
    data_up = df.copy()
    data_down = df.copy()
    data_up["Robustness"] = data_up["Robustness"] + data_up["std"]
    data_down["Robustness"] = data_down["Robustness"] - data_down["std"]
    df = pd.concat([data_up, data_down])
    ax[row][col].axhline(y=100, color=colors["Baseline"], label="Baseline")
    plot = sns.barplot(
        x="Corruption",
        y="Robustness",
        hue="Model",
        data=df,
        ax=ax[row][col],
        # yerr=df["std"],
        palette=colors,
    )
    # patches = sorted(plot.patches, key=lambda patch: patch.get_x())
    # for i, bar in enumerate(patches[-4:]):
    #     if i == 0:
    #         plt.axvline(x=bar.get_x(), color="grey", linestyle=":")
    #     bar.set_x(bar.get_x()+ bar.get_width())
    plot.set_xlabel("")
    ax[row][col].grid(True, linestyle=":")
    ax[row][col].set_ylim([50, 150])
    handles, labels = ax[row][col].get_legend_handles_labels()
    ax[row][col].get_legend().remove()
    new_labels = ["Baseline", "MTL-Monkey", "MTL-Shuffled", "MTL-Oracle", "Oracle"]
    new_handles = []
    for label in new_labels:
        new_handles.append(handles[labels.index(label)])
    fig.legend(new_handles, new_labels, loc=(0.01, 0.92), ncol=6, frameon=False)
    # fig.tight_layout()
    if despine:
        sns.despine(offset=3, trim=False)
    # plt.setp(ax[row][col].xaxis.get_majorticklabels(), rotation=30)
    # for label in ax[row][col].get_xticklabels():
    #     label.set_horizontalalignment('center')
    # ax[row][col].setp(ax[row][col].xaxis.get_majorticklabels(), rotation=-45)
    # ax[row][col].set_xticklabels(ax[row][col].get_xticks(), rotation=-45)

    # for tick in ax[row][col].get_xticklabels():
    #     tick.set_rotation(-45)
    # dx = 1 / 72.0
    # dy = 5 / 72.0
    # offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # for label in ax[row][col].xaxis.get_majorticklabels():
    #     label.set_transform(label.get_transform() + offset)

    for tick in ax[row][col].get_xticklabels():
        tick.set_rotation(-45)
    # Create offset transform by 5 points in x direction
    dx = 2 / 72.0
    dy = 4 / 72.0
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for label in ax[row][col].xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
