import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .base import Analyzer


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

    def plot_frontier(self, save="", style="lighttalk", legend_outside=True):
        df = self.generate_table(last_n=2, label_steps=True)
        direct_a = df.loc["Direct Training A"]
        direct_b = df.loc["Direct Training B"]
        fig, axs = plot_preparation(style, nrows=2, ncols=2)
        max_x, min_x, max_y, min_y = 0, 100, 0, 100
        for i, c in enumerate(df.columns):
            if i % 2 == 1:
                ax = axs[(i - 1) // 4][((i - 1) % 4) // 2]
                sns.lineplot(
                    data=df,
                    x=df.columns[i - 1],
                    y=c,
                    hue="name",
                    ax=ax,
                    legend="brief" if i == 5 else False,
                    style="name",
                    markers=True,
                )
                if i == 5 and legend_outside:
                    ax.legend(
                        fontsize=14,
                        title_fontsize="14",
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                        borderaxespad=0.0,
                    )
                ax.axhline(y=direct_b[c], lw=0.7, color="brown")
                ax.axvline(x=direct_a[df.columns[i - 1]], lw=0.8, color="red")
                min_x = min(min_x, ax.get_xlim()[0])
                min_y = min(min_y, ax.get_ylim()[0])
                max_x = max(max_x, ax.get_xlim()[1])
                max_y = max(max_y, ax.get_ylim()[1])
                ax.set_title(self.name_map(ax.get_xlabel()), fontweight="bold")
                ax.set_xlabel(self.name_map(ax.get_xlabel().split("->")[1], "A: "))
                ax.set_ylabel(self.name_map(ax.get_ylabel(), "B: "))

        axs[-1][-1].axis("off")

        for i in range(len(axs)):
            for j in range(len(axs[i])):
                # axs[i][j].set_xlim([min_x,max_x])
                axs[i][j].set_ylim([min_y, max_y])

        sns.despine(offset=5, trim=False)
        plt.subplots_adjust(hspace=0.4)
        if "talk" in style:
            if legend_outside:
                pass
                # ax.legend(
                #     fontsize=14,
                #     title_fontsize="14",
                #     bbox_to_anchor=(1.05, 1),
                #     loc="upper left",
                #     borderaxespad=0.0,
                # )
            else:
                plt.legend(fontsize=14, title_fontsize="14")
        elif legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
        if save:
            save_plot(
                fig,
                save + "_" + style,
                types=("png", "pdf", "pgf") if "nips" in style else ("png",),
            )
