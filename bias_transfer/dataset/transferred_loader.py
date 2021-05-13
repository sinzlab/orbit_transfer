import torch
from torch.utils.data import TensorDataset
import nnfabrik as nnf

from bias_transfer.dataset import mnist_transfer_dataset_loader
from nntransfer.dataset.dataset_classes.combined_dataset import ParallelDataset
from nntransfer.dataset.dataset_classes.npy_dataset import NpyDataset


def load_npy(postfix, data_key, transfer_data, data_loaders, main_data_loader):
    transferred_dataset = NpyDataset(
        samples=transfer_data["source" + postfix],
        targets=transfer_data["target" + postfix],
    )
    data_loaders["train"][data_key] = torch.utils.data.DataLoader(
        dataset=transferred_dataset,
        batch_size=main_data_loader.batch_size,
        num_workers=main_data_loader.num_workers,
        pin_memory=main_data_loader.pin_memory,
        shuffle=True,
    )


def transferred_dataset_loader(
    seed,  **config
):
    print("transferred data loader")
    transfer_data_file = config.pop("transfer_data")
    transfer_data = {k: transfer_data_file[k] for k in transfer_data_file.files}

    primary_dataset_fn = nnf.builder.resolve_fn(config.pop("primary_dataset_fn"), default_base="datasets")
    data_loaders = primary_dataset_fn(seed, **config)
    main_task = next(iter(data_loaders["train"].keys()))
    main_data_loader = data_loaders["train"][main_task]
    main_dataset = main_data_loader.dataset
    if "covariance" in transfer_data:
        data_loaders["covariance"] = transfer_data.pop("covariance")
    if "preds_prev_mem_prev_model" in transfer_data:
        data_loaders["preds_prev_mem_prev_model"] = transfer_data.pop("preds_prev_mem_prev_model")
    if "kernel_inv_prev_mem_prev_model" in transfer_data:
        data_loaders["kernel_inv_prev_mem_prev_model"] = transfer_data.pop("kernel_inv_prev_mem_prev_model")

    if "source_cs" in transfer_data:  # we have a coreset
        if config.get("train_on_coreset"):
            print("We train on coreset")
            load_npy("_cs", main_task, transfer_data, data_loaders, main_data_loader)
        else:
            if config.get("train_on_reduced_data"):
                load_npy("", main_task, transfer_data, data_loaders, main_data_loader)
            if config.get("load_coreset"):
                load_npy(
                    "_cs",
                    f"{main_task}_cs",
                    transfer_data,
                    data_loaders,
                    main_data_loader,
                )
    else:
        datasets = {}
        for rep_name, rep_data in transfer_data.items():
            datasets[rep_name] = TensorDataset(torch.from_numpy(rep_data))
        if "source" in transfer_data:  # we have input data
            source_ds = datasets.pop("source")
            transfer_dataset = ParallelDataset(
                source_datasets={"img": source_ds}, target_datasets=datasets
            )
            transfer_data_loader = torch.utils.data.DataLoader(
                dataset=transfer_dataset,
                batch_size=main_data_loader.batch_size,
                sampler=main_data_loader.sampler,
                num_workers=main_data_loader.num_workers,
                pin_memory=main_data_loader.pin_memory,
                shuffle=False,
            )
            data_loaders["train"]["transfer"] = transfer_data_loader
        else:  # we don't have input data -> only targets that are presented in parallel to class-labels
            datasets["class"] = main_dataset
            combined_dataset = ParallelDataset(
                source_datasets={"img": main_dataset}, target_datasets=datasets
            )
            combined_data_loader = torch.utils.data.DataLoader(
                dataset=combined_dataset,
                batch_size=main_data_loader.batch_size,
                sampler=main_data_loader.sampler,
                num_workers=main_data_loader.num_workers,
                pin_memory=main_data_loader.pin_memory,
                shuffle=False,
            )
            data_loaders["train"][main_task] = combined_data_loader
    return data_loaders
