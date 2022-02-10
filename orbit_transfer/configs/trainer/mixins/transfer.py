from typing import Dict, Tuple

from nntransfer.configs.base import BaseConfig


class DataGenerationMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.data_transfer: bool = False
        self.save_input: bool = False
        self.save_representation: bool = False
        self.compute_fisher: Dict = {
            "DEFAULT EMPTY": True,  # will turn into an empty dict
            "num_samples": 1024,
            "empirical": True,
        }
        self.compute_si_omega: Dict = {
            "DEFAULT EMPTY": True,  # will turn into an empty dict
            "damping_factor": 0.0001,
        }
        self.compute_covariance: bool = False
        self.extract_coreset: Dict = {}
        self.reset_for_new_task: bool = False

        super().__init__(**kwargs)


class TransferMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.data_transfer: bool = False
        self.scale_loss_with_arctanh: bool = False
        self.synaptic_intelligence_computation: bool = False
        self.freeze = None
        self.freeze_bn: bool = False
        self.transfer_restriction: Tuple = ()
        self.transfer_after_train: bool = False
        self.single_input_stream: bool = True
        self.readout_name: str = "fc"
        self.reset: Tuple = ()
        self.reset_linear_frequency = None
        self.regularization: Dict = {
            "DEFAULT EMPTY": True,  # will turn into an empty dict
            "regularizer": "L2SP/Mixup/RDL/KnowledgeDistillation",
            "alpha": 1.0,
            "decay_alpha": True,
        }

        super().__init__(**kwargs)

    def conditional_assignment(self):
        if (
            self.reset_linear_frequency
            and not "RandomReadoutReset" in self.main_loop_modules
        ):
            self.main_loop_modules.append("RandomReadoutReset")
        if (
            self.synaptic_intelligence_computation
            and not "SynapticIntelligence" in self.main_loop_modules
        ):
            self.main_loop_modules.append("SynapticIntelligence")
        if (
            self.regularization
            and not self.regularization["regularizer"] in self.main_loop_modules
        ):
            self.main_loop_modules.append(self.regularization["regularizer"])
        super().conditional_assignment()
