import numbers
from collections import Sequence

from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, Compose, ToTensor, ToPILImage

from bias_transfer.dataset.mnist_transfer.plot import plot_batch
from nntransfer.trainer.main_loop_modules.main_loop_module import MainLoopModule


class SIMCLR(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.core_out = self.config.simclr.get("core_out", "fc2")
        self.n_views = self.config.simclr.get("n_views", 2)
        self.temperature = self.config.simclr.get("temperature", 1)
        model = self.trainer.model
        augmentations = self.config.simclr.get(
            "augmentations", ["crop"]
        )
        view_transforms = []
        if "noise" in augmentations:
            view_transforms.append(GaussianNoise())
        # if "blur" in augmentations:
        #     view_transforms.append(GaussianBlur(kernel_size=3))

        for loader in trainer.data_loaders.values():
            dataset = loader["img_classification"].dataset
            while not hasattr(dataset, "transform") and hasattr(dataset, "dataset"):
                dataset = dataset.dataset
            if isinstance(dataset.transform, ContrastiveLearningViewGenerator):
                continue
            transforms = dataset.transform.transforms
            insert_before = 0
            while not isinstance(transforms[insert_before], ToTensor):
                insert_before += 1
            transforms = (
                transforms[:insert_before]
                + [
                    RandomResizedCrop(
                        model.input_size,
                        scale=(0.08, 1.0),
                        ratio=(0.75, 1.3333333333333333),
                    )
                ]
                + transforms[insert_before:]
                + view_transforms
            )
            transform = Compose(transforms)
            transform = ContrastiveLearningViewGenerator(
                transform, n_views=self.n_views
            )
            dataset.transform = transform

        core_out_dim = getattr(model, self.core_out).out_features
        self.lin1 = nn.Linear(core_out_dim, 50, bias=False).to(self.device)
        self.lin2 = nn.Linear(50, 25, bias=False).to(self.device)

    def pre_forward(self, model, inputs, task_key, shared_memory):
        self.batch_size = inputs.shape[0]
        # disentangle duplicate inputs
        inputs = inputs.reshape(
            self.batch_size * self.n_views, -1, inputs.shape[2], inputs.shape[3]
        )  # now duplicated inputs are alternating
        # plot_batch(inputs.cpu().numpy(), None, 2, 5)
        # prin()
        return model, inputs

    def info_nce_loss(self, features):
        """
        Adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.stack(
            [torch.arange(self.batch_size) for i in range(self.n_views)], dim=1
        ).reshape(self.batch_size * self.n_views)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def post_forward(self, outputs, loss, targets, **shared_memory):
        extra_outputs, outputs = outputs[0], outputs[1]
        # Retrieve representations that were selected for rep-matching:
        features = self.lin2(
            F.relu(self.lin1(extra_outputs[self.core_out].flatten(start_dim=1)))
        )
        logits, labels = self.info_nce_loss(features)
        return (extra_outputs, logits), loss, labels


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return torch.cat([self.base_transform(x) for i in range(self.n_views)], dim=0)


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )

