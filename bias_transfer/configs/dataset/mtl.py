from typing import Dict

from bias_transfer.configs.dataset.base import DatasetConfig
from bias_transfer.tables.nnfabrik import Dataset


class MTLDatasetsConfig(DatasetConfig):
    config_name = "dataset"
    table = Dataset()
    fn = "bias_transfer.dataset.mtl_datasets_loader"

    def __init__(self, sub_configs, **kwargs):
        self.load_kwargs(**kwargs)
        self.sub_configs = sub_configs
        super().__init__(**kwargs)

        # super().__init__(**kwargs)
        # self.neural_dataset_dict = kwargs.pop("neural_dataset_dict", {})
        # self.neural_dataset_config = NeuralDatasetConfig(
        #     **self.neural_dataset_dict
        # ).to_dict()
        # self.img_dataset_dict = kwargs.pop("img_dataset_dict", {})
        # self.img_dataset_config = ImageDatasetConfig(**self.img_dataset_dict).to_dict()
        #
        # self.update(**kwargs)

    def items(self):
        return self.sub_configs.items()

    def values(self):
        return self.sub_configs.values()

    def keys(self):
        return self.sub_configs.keys()

    def __getitem__(self, item):
        return self.sub_configs[item]

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MTLDatasetsConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
        Returns:
            :class:`MTLDatasetConfig`: An instance of a configuration object
        """
        sub_configs = {}
        for name, conf in config_dict.items():
            dataset_cls = next(iter(conf.keys()))
            sub_configs[name] = globals()[dataset_cls].from_dict(conf[dataset_cls])
        return cls(sub_configs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = {}
        for name, conf in self.sub_configs.items():
            output[name] = {conf.__class__.__name__: conf.to_dict()}
        return output