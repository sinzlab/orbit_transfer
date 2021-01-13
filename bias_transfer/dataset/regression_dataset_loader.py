import h5py
import numpy as np
import torch
import torch.utils.data as Data
from sklearn.datasets import fetch_openml

from bias_transfer.configs.dataset import Regression


def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    # normalize:
    avg_ppmvs -= np.mean(avg_ppmvs)
    avg_ppmvs /= np.std(avg_ppmvs)
    X_plot = months
    Y_plot = avg_ppmvs
    X_train = np.concatenate((X_plot[:120], X_plot[150:300], X_plot[380:450]))
    Y_train = np.concatenate((Y_plot[:120], Y_plot[150:300], Y_plot[380:450]))

    return X_plot, Y_plot, X_train, Y_train


def load_co2():
    f = h5py.File("co2_data.h5", "r")
    data_train = np.concatenate((f["data"].value, f["label"].value), axis=1)
    f.close()
    X_train = data_train[:, 0].reshape(-1, 1)
    Y_train = data_train[:, 1].reshape(-1)

    X_plot = np.concatenate((X_train, np.arange(1.73, 3.51, 0.01).reshape(-1, 1)))
    Y_plot = np.concatenate((Y_train, np.zeros((int((3.51 - 1.73) // 0.01 + 1),))))
    X_train = np.concatenate((X_train[:120], X_train[150:]))
    Y_train = np.concatenate((Y_train[:120], Y_train[150:]))

    return X_plot, Y_plot, X_train, Y_train


def load_sinusoid_data(noisy=False, train_range=10):
    def f(x):
        return (np.sin(x)).ravel()

    rng = np.random.RandomState(0)
    X_plot = np.linspace(-10, 40, 1000).reshape(-1, 1)
    X_train = np.sort(train_range * rng.rand(10 * train_range, 1), axis=0)
    # X_train = np.concatenate((X_train, (np.sort(10 * rng.rand(100, 1) + 20, axis=0))))
    Y_train = f(X_train)
    Y_plot = f(X_plot)
    if noisy:
        Y_train = Y_train + 1 * (0.5 - rng.rand(X_train.shape[0]))
    return X_plot, Y_plot, X_train, Y_train


def regression_dataset_loader(seed, **config):
    config = Regression.from_dict(config)
    print("Loading dataset: {}".format(config.dataset_cls))
    torch.manual_seed(seed)
    np.random.seed(seed)

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (config.valid_size >= 0) and (config.valid_size <= 1), error_msg

    if config.dataset_cls == "co2":
        X_plot, Y_plot, X_train, Y_train = load_co2()
    elif config.dataset_cls == "co2_original":
        X_plot, Y_plot, X_train, Y_train = load_mauna_loa_atmospheric_co2()
    else:
        X_plot, Y_plot, X_train, Y_train = load_sinusoid_data(
            noisy=config.noisy, train_range=config.train_range
        )

    train_len = X_train.shape[0]
    valid_start = int(train_len * (1.0 - config.valid_size))
    train_dataset = Data.TensorDataset(
        torch.tensor(X_train[:valid_start], dtype=torch.float),
        torch.tensor(Y_train[:valid_start], dtype=torch.float),
    )
    valid_dataset = Data.TensorDataset(
        torch.tensor(X_train[valid_start:], dtype=torch.float),
        torch.tensor(Y_train[valid_start:], dtype=torch.float),
    )
    test_dataset = Data.TensorDataset(
        torch.tensor(X_plot, dtype=torch.float),
        torch.tensor(Y_plot, dtype=torch.float),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        shuffle=True,
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
