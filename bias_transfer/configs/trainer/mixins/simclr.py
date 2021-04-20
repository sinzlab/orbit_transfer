from typing import Dict

from nntransfer.configs.base import BaseConfig


class SimclrMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.get_intermediate_rep = {"fc2": "fc2"}
        self.simclr = {
            "core_out": "fc2",
            "criterion": "mse",
        }

        super().__init__(**kwargs)

    def conditional_assignment(self):
        if (
            self.simclr
            and not "SIMCLR" in self.main_loop_modules
        ):
            self.main_loop_modules.append("SIMCLR")
        super().conditional_assignment()
