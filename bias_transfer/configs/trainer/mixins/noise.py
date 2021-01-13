from typing import Dict

from bias_transfer.configs.base import BaseConfig


class NoiseAugmentationMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.add_noise: bool = False
        self.noise_std: Dict = {
            "DEFAULT EMPTY": True,
            0.08: 0.1,
            0.12: 0.1,
            0.18: 0.1,
            0.26: 0.1,
            0.38: 0.1,
            -1: 0.5,
        }
        self.noise_snr: Dict = {}
        self.noise_test: Dict = {
            "noise_snr": [
                {5.0: 1.0},
                {4.0: 1.0},
                {3.0: 1.0},
                {2.0: 1.0},
                {1.0: 1.0},
                {0.5: 1.0},
                {0.0: 1.0},
            ],
            "noise_std": [
                {0.0: 1.0},
                {0.05: 1.0},
                {0.1: 1.0},
                {0.2: 1.0},
                {0.3: 1.0},
                {0.5: 1.0},
                {1.0: 1.0},
            ],
        }
        self.apply_noise_to_validation: bool = True

        super().__init__(**kwargs)

    def conditional_assignment(self):
        if (
                self.noise_snr or self.noise_std or self.noise_test
        ) and not self.representation_matching:  # Logit matching includes noise augmentation
            self.main_loop_modules.append("NoiseAugmentation")
        super().conditional_assignment()

class NoiseAdversarialMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.noise_adv_classification: bool = False
        self.noise_adv_regression: bool = False
        self.noise_adv_loss_factor: float = 1.0
        self.noise_adv_gamma: float = 10.0

        super().__init__(**kwargs)

    def conditional_assignment(self):
        if self.noise_adv_classification or self.noise_adv_regression:
            self.main_loop_modules.append("NoiseAdvTraining")
        super().conditional_assignment()

class RepresentationMatchingMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.representation_matching = {
            "DEFAULT EMPTY": True,
            "representation": "conv_rep",
            "criterion": "mse",
            "second_noise_std": {(0, 0.5): 1.0},
            "lambda": 1.0,
            "only_for_clean": True,
        }

        super().__init__(**kwargs)

    def conditional_assignment(self):
        if self.representation_matching:
            self.main_loop_modules.append("RepresentationMatching")
        super().conditional_assignment()
