from itertools import cycle

import torch
from torch import nn


def get_subdict(dictionary: dict, keys: list = None):
    """
    Args:
        dictionary: dictionary of all keys
        keys: list of strings representing the keys to be extracted from dictionary
    Return:
        dict: subdictionary containing only input keys
    """

    if keys:
        return {k: v for k, v in dictionary.items() if k in keys}
    return dictionary


class SchedulerWrapper:
    def __init__(self, lr_scheduler, warmup_scheduler):
        self.lr_scheduler = lr_scheduler
        self.warmup_scheduler = warmup_scheduler

    def __getattr__(self, item):
        if item != "warmup_scheduler":
            return getattr(self.lr_scheduler, item)

    def step(self, *args, **kwargs):
        self.lr_scheduler.step(*args, **kwargs)
        self.warmup_scheduler.dampen()


class StopClosureWrapper:
    def __init__(self, stop_closures):
        self.stop_closures = stop_closures

    def __call__(self):
        results = {task: {} for task in self.stop_closures.keys()}
        for task in self.stop_closures.keys():
            if task == "img_classification" or task == "regression":
                res = self.stop_closures[task]()
            else:
                for objective in self.stop_closures[task].keys():
                    results[task][objective] = self.stop_closures[task][objective]()
        return res


class MTL_Cycler:
    def __init__(self, loaders, main_key="img_classification", ratio=1):
        self.main_key = (
            main_key  # data_key of the dataset whose batch ratio is always 1
        )
        self.main_loader = loaders[main_key]
        self.other_loaders = {k: loaders[k] for k in loaders.keys() if k != main_key}
        self.ratio = ratio  # number of neural batches vs. one batch from TIN
        self.num_batches = len(self.main_loader) * (ratio + 1)

    def generate_batch(self, main_cycle, other_cycles_dict):
        for i in range(len(self.main_loader)):
            yield self.main_key, main_cycle
            for _ in range(self.ratio):
                key, loader = next(other_cycles_dict)
                yield key, loader

    def __iter__(self):
        other_cycles = {k: cycle(v) for k, v in self.other_loaders.items()}
        other_cycles_dict = cycle(other_cycles.items())
        main_cycle = cycle(self.main_loader)
        for k, loader in self.generate_batch(main_cycle, other_cycles_dict):
            yield k, next(loader)

    def __len__(self):
        return self.num_batches


class LongCycler:
    """
    Cycles through trainloaders until the loader with largest size is exhausted.
        Needed for dataloaders of unequal size (as in the monkey data).
    """

    def __init__(self, loaders):
        self.loaders = loaders
        self.max_batches = max([len(loader) for loader in self.loaders.values()])

    def __iter__(self):
        cycles = [cycle(loader) for loader in self.loaders.values()]
        for k, loader, _ in zip(
            cycle(self.loaders.keys()),
            (cycle(cycles)),
            range(len(self.loaders) * self.max_batches),
        ):
            yield k, next(loader)

    def __len__(self):
        return len(self.loaders) * self.max_batches


class XEntropyLossWrapper(nn.Module):
    def __init__(self, criterion):
        super(XEntropyLossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1))  # std
        self.criterion = criterion  # it is nn.CrossEntropyLoss

    def forward(self, preds, targets):
        precision = torch.exp(-self.log_w)
        loss = precision * self.criterion(preds, targets) + self.log_w
        return loss


class NBLossWrapper(nn.Module):
    def __init__(self):
        super(NBLossWrapper, self).__init__()
        self.log_w = nn.Parameter(torch.zeros(1))  # r: number of successes

    def forward(self, preds, targets):
        r = torch.exp(self.log_w)
        loss = (
            (targets + r) * torch.log(preds + r)
            - (targets * torch.log(preds))
            - (r * self.log_w)
            + torch.lgamma(r)
            - torch.lgamma(targets + r)
            + torch.lgamma(targets + 1)
            + 1e-5
        )
        return loss.mean()


def stringify(x):
    if type(x) is dict:
        x = ".".join(["{}_{}".format(k, v) for k, v in x.items()])
    return str(x)