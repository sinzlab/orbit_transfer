import os

import torch
from bias_transfer.dataset.mnist_transfer import generate_and_save
from nntransfer.dataset.dataset_classes.npy_dataset import NpyDataset

from bias_transfer.dataset.base_loader import ImageDatasetLoader


class MNISTTransferDatasetLoader(ImageDatasetLoader):
    def get_datasets(self, config, transform_test, transform_train, transform_val):
        if config.dataset_cls == "MNIST-Transfer":
            dataset_dir = os.path.join(config.data_dir, config.dataset_sub_cls + "-Transfer")
            generate_and_save(
                config.bias, base_path=config.data_dir, dataset=config.dataset_sub_cls
            )
            print(f"Loading: {config.bias}")
            train_dataset = NpyDataset(
                f"{config.bias}_train_source.npy",
                f"{config.bias}_train_target.npy",
                root=dataset_dir,
                transform=transform_train,
                target_type=torch.float32
                if "regression" in config.bias
                else torch.long,
            )
            valid_dataset = NpyDataset(
                f"{config.bias}_train_source.npy",
                f"{config.bias}_train_target.npy",
                root=dataset_dir,
                transform=transform_val,
                target_type=torch.float32
                if "regression" in config.bias
                else torch.long,
            )
            test_dataset = NpyDataset(
                f"{config.bias}_test_source.npy",
                f"{config.bias}_test_target.npy",
                root=dataset_dir,
                transform=transform_test,
                target_type=torch.float32
                if "regression" in config.bias
                else torch.long,
            )
        else:
            raise KeyError()

        st_test_dataset = self.add_stylized_test(config, transform_test)
        c_test_datasets = self.add_corrupted_test(config, transform_test)

        return (
            train_dataset,
            valid_dataset,
            test_dataset,
            c_test_datasets,
            st_test_dataset,
        )


mnist_transfer_dataset_loader = MNISTTransferDatasetLoader()
