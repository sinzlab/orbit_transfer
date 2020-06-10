from torch import nn
import torch
import numpy as np

from .noise_augmentation import NoiseAugmentation


class RepresentationMatching(NoiseAugmentation):
    def __init__(self, model, config, device, data_loader, seed):
        super().__init__(model, config, device, data_loader, seed)
        self.reps = self.config.representation_matching.get("representations", ["core"])
        if self.config.representation_matching.get("criterion", "cosine") == "cosine":
            self.criterion = nn.CosineEmbeddingLoss()
        else:
            self.criterion = nn.MSELoss()

        self.combine_losses = self.config.representation_matching.get("combine_losses")
        self.layer_weights = torch.ones((1, len(self.reps)), device=self.device)
        if self.combine_losses in ("learned_softmax", "learned_normalized"):
            self.layer_weights = nn.Parameter(self.layer_weights, requires_grad=True)
        else:
            normalization = 0
            for i in range(len(self.reps)):
                if self.combine_losses in ("lin", "exp", "avg"):
                    if self.combine_losses == "exp":
                        weight = np.exp(i + 1)
                    elif self.combine_losses == "lin":
                        weight = i + 1
                    else:
                        weight = 1
                self.layer_weights[0,i] = weight
                normalization += weight
            self.layer_weights /= normalization

    def pre_forward(
        self,
        model,
        inputs,
        shared_memory,
        train_mode,
        data_key="img_classification",
        **kwargs
    ):
        self.batch_size = inputs.shape[0]
        if "rep_matching" not in data_key:
            # Apply noise to input and save as input1:
            model, inputs1 = super().pre_forward(
                model, inputs, shared_memory, train_mode
            )
            # Decide for which inputs to perform representation matching:
            if self.config.representation_matching.get("only_for_clean", False):
                # Only for the clean part of the input:
                self.clean_flags = (shared_memory["applied_std"] == 0.0).squeeze()
            else:
                # For everything:
                self.clean_flags = torch.ones((self.batch_size,)).type(torch.BoolTensor)
        else:
            inputs1 = inputs
            self.clean_flags = torch.ones((self.batch_size,)).type(torch.BoolTensor)
        if self.config.representation_matching.get(
            "second_noise_std", None
        ) or self.config.representation_matching.get("second_noise_snr", None):
            # Apply noise to the selected inputs:
            inputs2, _ = self.apply_noise(
                inputs[self.clean_flags],
                self.device,
                std=self.config.representation_matching.get("second_noise_std", None),
                snr=self.config.representation_matching.get("second_noise_snr", None),
                rnd_gen=self.rnd_gen if not train_mode else None,
                img_min=self.img_min,
                img_max=self.img_max,
                noise_scale=self.noise_scale,
            )
        else:
            inputs2 = inputs
        inputs = torch.cat([inputs1, inputs2])
        return model, inputs

    def post_forward(self, outputs, loss, targets, extra_losses, train_mode, **kwargs):
        extra_outputs, outputs = outputs[0], outputs[1]
        rep_match_losses = torch.zeros((len(self.reps),), device=self.device)
        for i, rep in enumerate(self.reps):
            # Retrieve representations that were selected for rep-matching:
            rep_1 = extra_outputs[rep][: self.batch_size][self.clean_flags]
            rep_2 = extra_outputs[rep][self.batch_size :]
            # Compute the loss:
            if (
                self.config.representation_matching.get("criterion", "cosine")
                == "cosine"
            ):
                o = torch.ones(
                    rep_1.shape[:1], device=self.device
                )  # ones indicating that we want to measure similarity
                sim_loss = self.criterion(rep_1, rep_2, o)
            else:
                sim_loss = self.criterion(rep_1, rep_2)
            rep_match_losses[i] = sim_loss
        # combine losses from different layers:
        if self.combine_losses == "learned_softmax":
            layer_weights = nn.functional.softmax(self.layer_weights)
        elif self.combine_losses == "learned_normalized":
            layer_weights = self.layer_weights / self.layer_weights.sum()
        else:
            layer_weights = self.layer_weights
        total_rep_match_loss = layer_weights @ rep_match_losses
        # Add to the normal loss:
        loss += (
            self.config.representation_matching.get("lambda", 1.0)
            * total_rep_match_loss
        )
        extra_losses["RepresentationMatching"] += total_rep_match_loss.item()
        # Remove rep-matching duplicates from outputs
        outputs = outputs[: self.batch_size]
        for k, v in extra_outputs.items():
            if isinstance(v, torch.Tensor):
                extra_outputs[k] = v[: self.batch_size]
        return (extra_outputs, outputs), loss, targets
