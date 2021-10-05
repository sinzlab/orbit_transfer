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
                        row[l] = tracker.get_current_objective(objective)
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
