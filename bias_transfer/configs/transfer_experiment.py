import copy
from typing import List

from nnfabrik.utility.dj_helpers import make_hash

from .experiment import Experiment


class TransferExperiment(Experiment):
    r""" Collection of potentially multiple configs to define an experiment
    """

    config_name = "config"
    table = None
    fn = None

    def __init__(self, configs, update: List = []):
        self.configs = configs
        comment = []
        for c in self.configs:
            comment.append(c.comment)
            c.comment = " -> ".join(comment)
        self.comment = " -> ".join(comment)
        self.update(update)

    def update(self, settings: List):
        for i, setting in enumerate(settings):
            self.configs[i].update(setting)

    def get_restrictions(self, level: int = 0):
        if len(self.configs) <= level:
            return {}
        collapsed_history = ""
        for i, config in enumerate(self.configs[: level + 1]):
            current_key = config.get_key()
            current_key["transfer_step"] = i
            current_key["data_transfer"] = int(config.trainer.data_transfer)
            current_key["collapsed_history"] = collapsed_history
            collapsed_history = make_hash(current_key)
            del current_key["transfer_step"]  # we don't want this in the recipe (only here for hash)
            del current_key["data_transfer"]  # same as above
        return current_key

    def add_to_table(self):
        """
        Insert the config (+ fn and comment) into the dedicated table if not present already
        :return:
        """
        for config in self.configs:
            config.add_to_table()

    @classmethod
    def from_dict(cls, config_dicts: list) -> "TransferExperiment":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.PretrainedConfig.get_config_dict`
                method.
        Returns:
            :class:`TransferExperiment`: An instance of a configuration object
        """
        configs = []
        for c_dict in config_dicts:
            configs.append(Experiment.from_dict(c_dict))
        return cls(configs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        return [c.to_dict() for c in self.configs]
