import os
import numpy as np
import torch
from torchvision.datasets import VisionDataset


class NpyDataset(VisionDataset):
    def __init__(
        self,
        samples,
        targets,
        root="",
        transforms=None,
        transform=None,
        target_transform=None,
        source_type=torch.float32,
        target_type=torch.long,
    ):
        super().__init__(root, transforms, transform, target_transform)
        if not isinstance(samples, np.ndarray):
            self.samples = np.load(os.path.join(self.root, samples))
        else:
            self.samples = samples
        self.samples = torch.from_numpy(self.samples).type(source_type)
        if not isinstance(targets, np.ndarray):
            self.targets = np.load(os.path.join(self.root, targets))
        else:
            self.targets = targets
        self.targets = torch.from_numpy(self.targets).type(target_type)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return self.targets.shape[0]
