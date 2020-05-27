import os
import numpy as np
import torch
import pickle as pkl
from torchvision.datasets import VisionDataset


class PklDataset(VisionDataset):
    def __init__(
        self, pkl_file, root, transforms=None, transform=None, target_transform=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        with open(pkl_file, "rb") as f:
            dataset = pkl.load(f)
            self.samples = dataset["data"]
            self.targets = dataset["extrapolated_targets"]

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
