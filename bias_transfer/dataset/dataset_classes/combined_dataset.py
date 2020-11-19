from typing import Dict

from torch.utils.data import Dataset


class ParallelDataset(Dataset):
    """
    This class implements a wrapper that encapsulates multiple datasets on both source and target side.
    Assuming those datasets have the same length, they will be iterated through in parallel.
    """

    def __init__(self, source_datasets: Dict, target_datasets: Dict):
        self.source_datasets = source_datasets
        self.target_datasets = target_datasets

        # for ds in source_datasets.values():
        #     assert len(ds) == self.__len__()
        # for ds in target_datasets.values():
        #     assert len(ds) == self.__len__()

    def __getitem__(self, index):
        sources = {}
        for key, source_ds in self.source_datasets.items():
            s = source_ds[index]
            if isinstance(s, tuple):
                s = s[0]  # if this ds returns (source,target)-pair
            sources[key] = s
        targets = {}
        for key, target_ds in self.target_datasets.items():
            t = target_ds[index]
            if isinstance(t, tuple):
                t = t[-1]  # if this ds returns (source,target)-pair
            targets[key] = t
        return sources, targets

    def __len__(self):
        return len(next(iter(self.target_datasets)))
