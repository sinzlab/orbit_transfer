import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from nntransfer.analysis.plot import plot
from nntransfer.analysis.results.base import Analyzer
from nntransfer.tables.transfer import TransferredTrainedModel


class SimpleAnalyzer(Analyzer):
    def __init__(self):
        self.data = {}

    def load_data(self, configs):
        # Select data:
        for description, config in configs.items():
            level = 0
            entry_complete = True
            entry = {}
            while True:
                restriction = config.get_restrictions(level)
                if not restriction:
                    break
                restricted = TransferredTrainedModel() & restriction
                if restricted:  # could be empty if entry is not computed yet
                    fetch_res = restricted.fetch1("output")
                    if fetch_res:  # could be a data generation step (no output)
                        entry[level] = fetch_res
                else:
                    entry_complete = False
                level += 1
            if entry_complete:
                self.data[description] = entry

    def generate_table(
        self,
        objectives=(
            ("final", "train", "acc"),
            ("final", "test", "acc"),
            ("final", "validation", "acc"),
            # ("final", "validation_shift", "acc"),
            ("final", "test_shift", "acc"),
            # ("final", "validation_all", "acc"),
            ("final", "test_all", "acc"),
        ),
        last_n=1,
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
            row = {"name": name.split(":")[0]}
            for hyp in name.split(":")[1].split(" "):
                if len(hyp.split("=")) != 2:
                    continue
                hyp_name, hyp_value = hyp.split("=")
                try:
                    hyp_value = float(hyp_value)
                except:
                    pass
                row[hyp_name] = hyp_value
            levels = sorted(list(results.keys()))
            if last_n:
                levels = levels[(-1) * last_n :]
            for level, tracker in results.items():
                try:
                    if level in levels:
                        l = levels.index(level)
                        if labels:
                            l = labels[l]
                        for obj in objectives:
                            res = tracker
                            for key in obj:
                                res = res[key]
                            row[f"{l} {obj[-2]}"] = res
                except:
                    pass  # no valid entry for this objective
            row_list.append(row)
        df = pd.DataFrame(row_list)
        # if not df.empty:
        #     df = df.groupby("name").first()
        #     # Split off alpha from name
        #     df = df.reset_index()
        #     new = df["name"].str.split(":", n=1, expand=True)
        #     if len(new.columns) > 1:
        #         df.drop(columns=["name"], inplace=True)
        #         df["name"] = new[0]
        #         df["hyps"] = new[1]
        # return df
        # new = df["hyps"].str.split(" ", expand=True)
        # if len(new.columns) > 1:
        #     df.drop(columns=["hyps"], inplace=True)
        #     for c in new.columns[1:]:
        #         split = new[c].str.split("=", n=2, expand=True)
        #         df[split[0][0]] = pd.to_numeric(split[1])
        df = df.set_index("name")
        return df

    @plot
    def plot_gamma_line(
        self,
        to_plot,
        fig=None,
        ax=None,
        rename=lambda x: x,
        names={},
        x_var="gamma",
        x_var_rename="$\gamma$",
        objectives={
            # "0 train": "Train",
            "0 test": "Seen Shifts",
            # "0 validation": "Validation (Seen Shift)",
            # "0 validation_shift": "Validation (Unseen Shift)",
            "0 test_shift": "Unseen Shifts",
            # "0 validation_all": "Validation (All Shift)",
            "0 test_all": "All Shifts",
        },
        all_objectives=[
            "0 test_all",
            "0 validation_all",
            "0 validation_shift",
            "0 validation",
            "0 test",
            "0 train",
            "0 test_shift",
        ],
        drop=[],
        xticks = (),
        **kwargs,
    ):
        df = self.generate_table()
        id_vars = [c for c in df.columns if c not in all_objectives]
        df = df.rename(columns=objectives)
        for d in drop:
            df = df.drop(df[df.index == d].index)
        df = df.rename(index=names)

        value_vars = list(objectives.values())

        cols = len(ax[0])
        for i, name in enumerate(names.values()):
            r = i // cols
            c = i % cols
            sub_df = df.loc[df.index == name]
            self.single_plot(
                sub_df,
                ax[r][c],
                legend=(i == 0),
                value_vars=value_vars,
                id_vars=id_vars,
                x_var=x_var,
                x_var_rename=x_var_rename,
                name=name,
            )
            if xticks:
                ax[r][c].set_xticks(xticks)
            if c > 0:
                ax[r][c].set_ylabel("")
                ax[r][c].set_xlabel("")
            if r < len(ax) - 1:
                ax[r][c].set_xlabel("")
        # ax[-1][-1].axis('off')
        ax[0][0].get_legend().remove()
        # fig.legend(bbox_to_anchor=(1.1,0.4))
        fig.legend(loc=(0.28, 0.92), ncol=3, frameon=False)

    def single_plot(
        self, df, ax, legend, value_vars, id_vars, x_var, x_var_rename, name
    ):
        max_df = df
        max_row = df.iloc[[df["Seen Shifts"].argmax()]]
        print(max_row)
        for var in id_vars:
            if var == x_var:
                continue
            if max_df[var].isnull().values.any() or max_row[var].isnull().values.any():
                continue
            max_df = max_df.loc[max_df[var] == max_row[var].iat[0]]

        def plot_helper(df, alpha, legend, style):
            df = pd.melt(
                df,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name="Set",
                value_name="Acc",
                ignore_index=False,
            )
            df = df.rename(columns={x_var: x_var_rename, "Acc": "Accuracy [%]"})
            hyps = [c for c in id_vars if c != x_var]
            df["hyps"] = df[hyps[0]].astype(str)
            del df[hyps[0]]
            for hyp in hyps[1:]:
                df["hyps"] = df["hyps"] + df[hyp].astype(str)
                del df[hyp]

            sns.lineplot(
                x=x_var_rename,
                y="Accuracy [%]",
                data=df,
                hue="Set",
                ax=ax,
                style=style,
                markers=False,
                dashes=False,
                alpha=alpha,
                legend=legend,
            )
            ax.set_ylim(0, 100)

        plot_helper(max_df, alpha=1.0, legend=legend, style="Set")
        ax.set_title(name)
        plot_helper(df, alpha=0.05, legend=False, style="hyps")
