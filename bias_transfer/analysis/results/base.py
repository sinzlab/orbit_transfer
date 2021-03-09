import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from bias_transfer.analysis.plot import plot
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

    @plot
    def plot_progress_line(self, to_plot, fig=None, ax=None, level=0, **kwargs):
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

    @plot
    def plot_comparison_line(
        self, to_plot, fig=None, ax=None, rename=lambda x: x, **kwargs
    ):
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
        sns.lineplot(x="Level", data=df, y=to_plot[0][-1], hue="Training", ax=ax[0][0])

    @plot
    def plot_bar(self, to_plot, fig=None, ax=None, **kwargs):
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

    def name_map(self, old_name, prefix=""):
        name = old_name.replace("->", " → ")
        name = name.replace("_", " ")
        name = " ".join([n.capitalize() for n in name.split()])
        return prefix + name
