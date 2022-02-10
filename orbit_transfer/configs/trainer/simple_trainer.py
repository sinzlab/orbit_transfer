from typing import Dict, Tuple

from nntransfer.configs import TrainerConfig
from nntransfer.tables.nnfabrik import Trainer


class SimpleTrainerConfig(TrainerConfig):
    config_name = "trainer"
    fn = "orbit_transfer.trainer.simple_trainer.trainer_fn"
    table = Trainer()

    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)
        self.forward = "default"
        self.total_steps = 8000
        self.batch_size = 1000
        self.eval_every = 1000
        self.print_every = 1000
        self.checkpoint_every = 1000
        self.rep_layer = "linear1"
        self.teacher_rep_layer = "conv1"
        self.lr_decay: float = 0.8
        self.learning_rate = 1e-2
        self.restore_best = True
        self.lr_decay_steps = 3
        self.patience = 5
        self.weight_decay = 0.0
        self.device = "cuda"
        self.gamma = 0.0
        self.softmax_temp = 1.0
        self.equiv_factor = 1.0
        self.invertible_factor = 1.0
        self.identity_factor = 1.0
        self.student_model = {}

        super().__init__(**kwargs)
