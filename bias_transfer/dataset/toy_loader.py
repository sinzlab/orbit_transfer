import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from bias_transfer.configs import ToyDatasetConfig
from bias_transfer.configs.dataset.toy import ToySineDatasetConfig


class ToyDatasetLoader:
    def __call__(self, seed, **config):
        config = ToyDatasetConfig.from_dict(config)
        print("Loading dataset: {}".format(config.dataset_cls))
        torch.manual_seed(seed)
        np.random.seed(seed)

        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert (config.valid_size >= 0) and (config.valid_size <= 1), error_msg

        mu_a, cov_a = np.array(config.mu_a), np.array(config.cov_a)
        mu_b, cov_b = np.array(config.mu_b), np.array(config.cov_b)
        data_a = np.random.multivariate_normal(mean=mu_a, cov=cov_a, size=config.size)
        data_b = np.random.multivariate_normal(mean=mu_b, cov=cov_b, size=config.size)
        X = np.concatenate([data_a, data_b])
        Y = np.concatenate(
            [
                np.ones([config.size, 1]),
                np.zeros([config.size, 1]),
            ]
        )
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

        train_len = X_train.shape[0]
        valid_start = int(train_len * (1.0 - config.valid_size))
        train_dataset = TensorDataset(
            torch.tensor(X_train[:valid_start], dtype=torch.float),
            torch.tensor(Y_train[:valid_start], dtype=torch.float),
        )
        valid_dataset = TensorDataset(
            torch.tensor(X_train[valid_start:], dtype=torch.float),
            torch.tensor(Y_train[valid_start:], dtype=torch.float),
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float),
            torch.tensor(Y_test, dtype=torch.float),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=config.shuffle,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False,
        )

        data_loaders = {
            "train": {"img_classification": train_loader},
            "validation": {"img_classification": valid_loader},
            "test": {"img_classification": test_loader},
        }

        return data_loaders


toy_dataset_loader = ToyDatasetLoader()


class ToySineDatasetLoader:
    def __call__(self, seed, **config):
        config = ToySineDatasetConfig.from_dict(config)
        print("Loading dataset: {}".format(config.dataset_cls))
        torch.manual_seed(seed)
        np.random.seed(seed)

        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert (config.valid_size >= 0) and (config.valid_size <= 1), error_msg

        X, Y = self.generate_sine_data(size=config.size, **config.sine)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

        train_len = X_train.shape[0]
        valid_start = int(train_len * (1.0 - config.valid_size))
        train_dataset = TensorDataset(
            torch.tensor(X_train[:valid_start], dtype=torch.float),
            torch.tensor(Y_train[:valid_start], dtype=torch.float),
        )
        valid_dataset = TensorDataset(
            torch.tensor(X_train[valid_start:], dtype=torch.float),
            torch.tensor(Y_train[valid_start:], dtype=torch.float),
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float),
            torch.tensor(Y_test, dtype=torch.float),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=config.shuffle,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            shuffle=False,
        )

        data_loaders = {
            "train": {"regression": train_loader},
            "validation": {"regression": valid_loader},
            "test": {"regression": test_loader},
        }

        return data_loaders

    def generate_sine_data(
        self,
        amplitude,
        phase,
        freq,
        x_range,
        size,
        samples_per_function,
        multi_regression=False,
    ):
        amplitude = np.random.uniform(low=amplitude[0], high=amplitude[1], size=size)
        phase = np.random.uniform(low=phase[0], high=phase[1], size=size)
        freq = np.random.uniform(low=freq[0], high=freq[1], size=size)
        y_data = np.zeros((size, samples_per_function))
        x_data = np.zeros((size, samples_per_function))
        for i in range(size):
            if not multi_regression or i == 0:
                x_data[i, :] = np.random.uniform(
                    low=x_range[0], high=x_range[1], size=samples_per_function
                )
                x = x_data[i]
            y_data[i, :] = amplitude[i] * np.sin(x * freq[i] - phase[i])
        if size == 1:
            return x_data.reshape(samples_per_function, 1), y_data.reshape(
                samples_per_function
            )
        if multi_regression:
            return x_data[0].reshape(samples_per_function,1), y_data.transpose()
        return x_data.reshape(size, samples_per_function, 1), y_data


toy_sine_dataset_loader = ToySineDatasetLoader()
