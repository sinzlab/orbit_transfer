import os

import torchvision
from torchvision import datasets

from nntransfer.dataset.dataset_classes.pkl_dataset import PklDataset
from nntransfer.dataset.utils import get_dataset, create_ImageFolder_format

from bias_transfer.dataset.base_loader import ImageDatasetLoader

DATASET_URLS = {
    "TinyImageNet": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "CIFAR10-Semisupervised": "1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi",
}


class URLDatasetLoader(ImageDatasetLoader):
    def get_datasets(self, config, transform_test, transform_train, transform_val):
        dataset_dir = get_dataset(
            DATASET_URLS[config.dataset_cls],
            config.data_dir,
            dataset_cls=config.dataset_cls,
        )

        train_dir = os.path.join(dataset_dir, "train")
        if config.dataset_cls == "CIFAR10-Semisupervised":
            train_dataset = PklDataset(
                train_dir, transform=transform_train, root=config.data_dir
            )
            valid_dataset = PklDataset(
                train_dir, transform=transform_val, root=config.data_dir
            )
            dataset_cls = torchvision.datasets.CIFAR10
            test_dataset = dataset_cls(
                root=config.data_dir,
                train=False,
                transform=transform_test,
            )
        else:
            if config.dataset_cls != "ImageNet":
                create_ImageFolder_format(dataset_dir)
            val_dir = os.path.join(dataset_dir, "val", "images")
            train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
            valid_dataset = datasets.ImageFolder(train_dir, transform=transform_val)
            test_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

        st_test_dataset = self.add_stylized_test(config, transform_test)
        c_test_datasets = self.add_corrupted_test(config, transform_test)

        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            c_test_datasets,
            st_test_dataset,
        )


url_dataset_loader = URLDatasetLoader()
