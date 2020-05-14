from bias_transfer.configs.dataset import MTLDatasetsConfig
from nnfabrik.builder import resolve_data
from .img_dataset_loader import img_dataset_loader
from .neural_dataset_loader import neural_dataset_loader

#
# def mtl_datasets_loader(seed, **config):
#     neural_dataset_config = config.pop("neural_dataset_config")
#     img_dataset_config = config.pop("img_dataset_config")
#
#     neural_dataset_config.pop("seed")
#
#     neural_dataset_loaders = neural_dataset_loader(seed, **neural_dataset_config)
#     img_dataset_loaders = img_dataset_loader(seed, **img_dataset_config)
#
#     data_loaders = neural_dataset_loaders
#     data_loaders["train"]["img_classification"] = img_dataset_loaders["train"][
#         "img_classification"
#     ]
#     data_loaders["validation"]["img_classification"] = img_dataset_loaders[
#         "validation"
#     ]["img_classification"]
#     data_loaders["test"]["img_classification"] = img_dataset_loaders["test"][
#         "img_classification"
#     ]
#     if "c_test" in img_dataset_loaders:
#         data_loaders["c_test"] = img_dataset_loaders["c_test"]
#     return data_loaders

def update(to_update, new_entries, prefix=""):
    for k,v in new_entries.items():
        if prefix:
            k = prefix + "_" + k
        to_update[k] = v

def mtl_datasets_loader(seed, **config):
    mtl_config = MTLDatasetsConfig.from_dict(config)
    mtl_data_loaders = {"train": {}, "validation": {}, "test": {}}
    for prefix, dataset_config in mtl_config.items():
        dataset_config.seed = seed
        dataset_fn = resolve_data(dataset_config.fn)
        data_loaders = dataset_fn(**dataset_config.to_dict())
        update(mtl_data_loaders["train"], data_loaders["train"], prefix)
        update(mtl_data_loaders["validation"], data_loaders["validation"], prefix)
        update(mtl_data_loaders["test"], data_loaders["test"], prefix)
        if "c_test" in data_loaders:
            if "c_test" not in mtl_data_loaders:
                mtl_data_loaders["c_test"] = {}
            update(mtl_data_loaders["c_test"], data_loaders["c_test"], prefix)
        if "st_test" in data_loaders:
            if "st_test" not in mtl_data_loaders:
                mtl_data_loaders["st_test"] = {}
            update(mtl_data_loaders["st_test"], data_loaders["st_test"], prefix)
    return mtl_data_loaders
