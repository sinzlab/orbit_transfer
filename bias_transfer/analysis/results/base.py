import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bias_transfer.analysis.utils import plot_preparation, save_plot
from bias_transfer.tables.transfer import TransferredTrainedModel
from neuralpredictors.tracking import AdvancedMultipleObjectiveTracker as Tracker


class Analyzer:
    def __init__(self):
        self.data = {}

    def load_data(self, configs):
        # Select data:
        for description, config in configs.items():
            level = 0
            while True:
                restriction = config.get_restrictions(level)
                if not restriction:
                    break
                restricted = TransferredTrainedModel() & restriction
                if restricted:  # could be empty if entry is not computed yet
                    fetch_res = restricted.fetch1("output")
                    if fetch_res:  # could be a data generation step (no output)
                        if description not in self.data:
                            self.data[description] = {
                                level: Tracker.from_dict(fetch_res)
                            }
                        else:
                            self.data[description][level] = Tracker.from_dict(fetch_res)
                level += 1

    def _post_plot_operations(self, fig, style="lighttalk", legend_outside=True, save=""):
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

    def plot_progress_line(self, to_plot, level=0, style="lighttalk", **kwargs):
        fig, ax = plot_preparation(style)
        row_list = []
        for desc, tracker in self.data.items():
            if len(tracker.keys()) > level:
                l = list(tracker.keys())[level]
                try:
                    row = {
                        "name": desc.name,
                        to_plot[-1]: tracker[l].get_objective(*to_plot),
                    }
                    row_list.append(row)
                    print(desc.name, row)
                except:
                    print("skipping", desc.name)
        df = pd.DataFrame(row_list)
        df.index = df.name
        del df["name"]
        df = df[to_plot[-1]].apply(pd.Series)
        df = df.stack().reset_index()
        df.columns = ["Training", "Epoch", to_plot[-1]]
        sns.lineplot(x="Epoch", y=to_plot[-1], hue="Training", data=df, ax=ax)
        sns.despine(offset=10, trim=True)
        self._post_plot_operations(fig, style, **kwargs)

    def plot_comparison_line(
        self, to_plot, style="lighttalk", rename=lambda x: x, **kwargs
    ):
        fig, ax = plot_preparation(style)
        row_list = []
        for desc, tracker in self.data.items():
            level = max(tracker.keys())
            row = {"name": self.name_map(desc.name)}
            for key in to_plot:
                row[rename(key[0])] = tracker[level].get_current_objective(*key)
            row_list.append(row)
        df = pd.DataFrame(row_list)
        df.index = df.name
        del df["name"]
        df = df.stack().reset_index()
        df.columns = ["Training", "Level", to_plot[0][-1]]
        sns.lineplot(x="Level", data=df, y=to_plot[0][-1], hue="Training", ax=ax)
        sns.despine(offset=10, trim=True)
        self._post_plot_operations(fig, style, **kwargs)

    def plot_bar(self, to_plot, style="lighttalk", **kwargs):
        fig, ax = plot_preparation(style)
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
        sns.despine(offset=10, trim=True)
        self._post_plot_operations(fig, style, **kwargs)

    def name_map(self, old_name, prefix=""):
        name = old_name.replace("->", " → ")
        name = name.replace("_", " ")
        name = " ".join([n.capitalize() for n in name.split()])
        return prefix + name
