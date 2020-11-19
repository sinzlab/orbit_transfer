import torch
from torch.utils.data import TensorDataset

from bias_transfer.dataset import img_dataset_loader
from bias_transfer.dataset.dataset_classes.combined_dataset import ParallelDataset


def transferred_dataset_loader(seed, primary_dataset_fn=img_dataset_loader, **config):
    transfer_data_file = config.pop("transfer_data")
    transfer_data = {k: transfer_data_file[k] for k in transfer_data_file.files}

    data_loaders = primary_dataset_fn(seed, **config)
    main_data_loader = data_loaders["train"]["img_classification"]
    main_dataset = main_data_loader.dataset

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
        data_loaders["train"]["img_classification"] = combined_data_loader
    return data_loaders
