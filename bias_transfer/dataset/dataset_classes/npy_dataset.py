import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset


class NpyDataset(VisionDataset):
    def __init__(
        self,
        sample_file,
        target_file,
        root,
        start=0,
        end=-1,
        transforms=None,
        transform=None,
        target_transform=None,
        source_type=torch.float32,
        target_type=torch.long,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.sample_file = os.path.join(self.root, sample_file)
        self.target_file = os.path.join(self.root, target_file)
        self.start = start
        self.end = end
        self.samples = None
        self.targets = None
        self.load(source_type, target_type)
        self.end = self.end if self.end > 0 else self.targets.shape[0]

    def load(self, source_type, target_type):
        if self.samples is None:
            self.samples = torch.from_numpy(
                np.load(self.sample_file)[self.start : self.end]
            ).type(source_type)
        if self.targets is None:
            self.targets = torch.from_numpy(
                np.load(self.target_file)[self.start : self.end]
            ).type(target_type)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return self.end - self.start
