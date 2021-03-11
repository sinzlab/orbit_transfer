from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

from .main_loop_module import MainLoopModule


class RepresentationMonitor(MainLoopModule):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.reps = self.config.representation_monitor.get("representations", ["core"])
        objectives = {
            "Training": {"RepresentationMonitor": {"magnitude": 0, "normalization": 0}},
            "Validation": {
                "RepresentationMonitor": {"magnitude": 0, "normalization": 0}
            },
            "Test": {"RepresentationMonitor": {"magnitude": 0, "normalization": 0}},
        }
        self.tracker.add_objectives(objectives)

    def post_forward(self, outputs, loss, targets, **shared_memory):
        extra_outputs, outputs = outputs[0], outputs[1]
        batch_size = outputs.shape[0]
        rep_magnitudes = torch.zeros(
            (
                batch_size,
                len(self.reps),
            ),
            device=self.device,
        )
        for i, rep in enumerate(self.reps):
            # Retrieve representations that were selected for monitoring:
            rep = extra_outputs[rep].cpu().detach()
            rep = torch.flatten(rep, start_dim=1)
            rep_magnitudes[i] = torch.norm(rep, dim=1).sum()  # norm everything but batch dim -> sum over batch dim
        total_magnitude = rep_magnitudes.mean()  # mean over representations
        self.tracker.log_objective(
            total_magnitude.item(),
            (self.mode, "RepresentationMonitor", "magnitude"),
        )
        self.tracker.log_objective(
            batch_size,
            (self.mode, "RepresentationMonitor", "normalization"),
        )
        # Remove rep-matching duplicates from outputs
        return (extra_outputs, outputs), loss, targets
