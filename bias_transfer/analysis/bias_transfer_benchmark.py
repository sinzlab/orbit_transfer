import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nntransfer.analysis.results.base import Analyzer
from nntransfer.analysis.plot import plot


class BiasTransferAnalyzer(Analyzer):
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
        df = pd.DataFrame(row_list)
        if not df.empty:
            df = df.groupby("name").first()
            # Split off alpha from name
            df = df.reset_index()
            new = df["name"].str.split(":", n=1, expand=True)
            if len(new.columns) > 1:
                df.drop(columns=["name"], inplace=True)
                df["name"] = new[0]
                df["alpha"] = new[1]
            df = df.set_index("name")
        return df

    def generate_normalized_table(self):
        df = self.generate_table(last_n=2, label_steps=True)
        for i, c in enumerate(df.columns):
            offset = "A" if i % 2 == 0 else "B"
            baseline = df.at[f"Direct Training {offset}", c]
            df.insert(
                2 * i + 1, c + " normalized", df[c].divide(baseline).multiply(100)
            )
        return df

    @plot
    def plot_frontier(
        self, fig, ax, columns_range=(), title=False, hide_lines=False,
    ):
        df = self.generate_table(last_n=2, label_steps=True)
        direct_a = (
            df.loc["Direct Training on Target"]
            if "Direct Training on Target" in df.index
            else None
        )
        direct_b = (
            df.loc["Direct Training on Eval"]
            if "Direct Training on Eval" in df.index
            else None
        )
        max_x, min_x, max_y, min_y = 0, 100, 0, 100
        for i, c in enumerate(df.columns):
            if not columns_range[0] <= i <= columns_range[1]:
                continue
            if i % 2 == 1:
                if True:
                    a = ax[i - 1 - columns_range[0]][i - 1 - columns_range[0]]
                else:
                    a = ax[(i - 1) // 4][((i - 1) % 4) // 2]
                colors = [
                    "#a6cee3",
                    "#1f78b4",
                    "#b2df8a",
                    "#33a02c",
                    "#fb9a99",
                    "#e31a1c",
                    "#fdbf6f",
                    "#ff7f00",
                    "#cab2d6",
                    "#6a3d9a",
                    "#ffff99",
                ]
                models = sorted(list(set(df.index)))
                print(models)
                colors = dict(zip(models, colors[: len(models)]))
                print(colors)
                plot_res = sns.lineplot(
                    data=df,
                    x=df.columns[i - 1],
                    y=c,
                    hue="name",
                    ax=a,
                    legend="brief",
                    style="name",
                    markers=True,
                    palette=colors,
                )
                for line in plot_res.lines[2:]:
                    line.set_visible(not hide_lines)
                # if i == 5 and legend_outside:
                #     a.legend(
                #         fontsize=14,
                #         title_fontsize="14",
                #         bbox_to_anchor=(1.05, 1),
                #         loc="upper left",
                #         borderaxespad=0.0,
                #     )
                if direct_b is not None:
                    a.axhline(
                        y=direct_b[c], lw=0.7, color=colors["Direct Training on Eval"]
                    )
                if direct_a is not None:
                    a.axvline(
                        x=direct_a[df.columns[i - 1]],
                        lw=0.7,
                        color=colors["Direct Training on Target"],
                    )
                min_x = min(min_x, a.get_xlim()[0])
                min_y = min(min_y, a.get_ylim()[0])
                max_x = max(max_x, a.get_xlim()[1])
                max_y = max(max_y, a.get_ylim()[1])
                if title:
                    a.set_title(self.name_map(a.get_xlabel()), fontweight="bold")
                a.set_xlabel(
                    self.name_map(a.get_xlabel().split("->")[1], "Target Task: ")
                )
                a.set_ylabel(self.name_map(a.get_ylabel(), "Evaluation: "))


        for i in range(len(ax)):
            for j in range(len(ax[i])):
                # axs[i][j].set_xlim([min_x,max_x])
                ax[i][j].set_ylim([min_y, max_y])

        # sns.despine(offset=5, trim=False)
        # plt.subplots_adjust(hspace=0.4)
        # if "talk" in style:
        #     if legend_outside:
        #         pass
        #         # ax.legend(
        #         #     fontsize=14,
        #         #     title_fontsize="14",
        #         #     bbox_to_anchor=(1.05, 1),
        #         #     loc="upper left",
        #         #     borderaxespad=0.0,
        #         # )
        #     else:
        #         plt.legend(fontsize=14, title_fontsize="14")
        # elif legend_outside:
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        # if save:
        #     save_plot(
        #         fig,
        #         save + "_" + style,
        #         types=("png", "pdf", "pgf") if "nips" in style else ("png",),
        #     )
