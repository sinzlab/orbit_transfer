import copy

from nnfabrik.utility.dj_helpers import make_hash

from .experiment import Experiment


class TransferExperiment(Experiment):
    r""" Collection of potentially multiple configs to define an experiment
    """

    config_name = "config"
    table = None
    fn = None

    def __init__(self, configs):
        self.configs = configs
        comment = []
        for c in self.configs:
            comment.append(c.comment)
            c.comment = " -> ".join(comment)
        self.comment = " -> ".join(comment)

    def get_restrictions(self, level: int = 0):
        if len(self.configs) <= level:
            return {}
        prev_key = {
            "prev_model_fn": "",
            "prev_model_hash": "",
            "prev_dataset_fn": "",
            "prev_dataset_hash": "",
            "prev_trainer_fn": "",
            "prev_trainer_hash": "",
            "collapsed_history": "",
        }
        for i, config in enumerate(self.configs[:level+1]):
            if i > 0:
                key_for_hash = copy.deepcopy(prev_key)
                key_for_hash["prev_collapsed_history"] = key_for_hash["collapsed_history"]
                del key_for_hash["collapsed_history"]
                collapsed_history = make_hash(key_for_hash)
                prev_key = {
                    "prev_model_fn": current_key["model_fn"],
                    "prev_model_hash": current_key["model_hash"],
                    "prev_dataset_fn": current_key["dataset_fn"],
                    "prev_dataset_hash": current_key["dataset_hash"],
                    "prev_trainer_fn": current_key["trainer_fn"],
                    "prev_trainer_hash": current_key["trainer_hash"],
                    "collapsed_history": collapsed_history,
                }
            current_key = config.get_key()  # TrainedModel keys
        current_key.update(prev_key)
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
