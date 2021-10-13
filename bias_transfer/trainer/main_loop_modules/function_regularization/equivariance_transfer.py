from torch import nn
import torch
import torch.nn.functional as F


from .representation import RepresentationRegularization


class EquivarianceTransfer(RepresentationRegularization):
    def __init__(self, trainer):
        super().__init__(trainer, name="Equiv")
        objectives = {
            "Training": {
                "Equiv": {
                    "distance": 0,
                    "inv_reg": 0,
                    "identity_reg": 0,
                    "normalization": 0,
                }
            },
            "Validation": {
                "Equiv": {
                    "distance": 0,
                    "inv_reg": 0,
                    "identity_reg": 0,
                    "normalization": 0,
                }
            },
            "Test": {
                "Equiv": {
                    "distance": 0,
                    "inv_reg": 0,
                    "identity_reg": 0,
                    "normalization": 0,
                }
            },
        }
        self.tracker.add_objectives(objectives, init_epoch=True)
        self.learn_equiv = self.config.regularization.get("learn_equiv", True)
        self.group_size = self.config.regularization.get("group_size", 40)
        self.equiv_factor = self.config.regularization.get("equiv_factor", 1.0)
        self.inv_factor = self.config.regularization.get("inv_factor", 1.0)
        self.id_factor = self.config.regularization.get("id_factor", 1.0)
        self.id_between_filters = self.config.regularization.get(
            "id_between_filters", False
        )
        self.max_stacked_transform = self.config.regularization.get(
            "max_stacked_transform", 1
        )

    def pre_forward(self, model, inputs, task_key, shared_memory):
        if self.mode in ("Training", "Validation", "Test"):
            b = inputs.shape[0]
            # sample group element (one per batch element)
            g = torch.randint(0, self.group_size, (b,))
            n = (
                torch.randint(1, self.max_stacked_transform, (1,))
                if self.max_stacked_transform > 1
                else 1
            )
            if self.learn_equiv and not self.id_between_filters:
                h = (g + torch.randint(1, self.group_size, (b,))) % self.group_size
                inputs = inputs.repeat(2, 1, 1, 1)
                g = torch.cat([g, h], dim=0)
            # apply group representation on input
            rho_g_inputs = self.teacher(inputs, g, 0, n)
            if self.learn_equiv:
                if not self.id_between_filters:
                    rho_g_inputs, rho_h_inputs = (
                        rho_g_inputs[:b],
                        rho_g_inputs[b:],
                    )
                    shared_memory["rho_h_inputs"] = rho_h_inputs
                    inputs = inputs[:b]
                    g = g[:b]
                shared_memory["inputs"] = inputs
                shared_memory["rho_g_inputs"] = rho_g_inputs
            shared_memory["b"] = b
            shared_memory["g"] = g
            shared_memory["n"] = n
            # pass transformed and non-transformed input through the model
            inputs = torch.cat([inputs, rho_g_inputs], dim=0)
        return model, inputs

    def post_forward(self, outputs, loss, targets, **shared_memory):
        if self.mode in ("Training", "Validation", "Test"):
            extra_outputs = outputs[0]
            layers = list(extra_outputs.keys())
            b = shared_memory["b"]
            g = shared_memory["g"]
            n = shared_memory["n"]

            # collect output that for transformed input -> [b:]
            phi_rho_g_x = torch.cat(
                [extra_outputs[l][b:].flatten(1) for l in layers],
                dim=1,
            )
            # apply group representation on output that belongs to untransformed input -> [:b]
            rho_g_phi_x = torch.cat(
                [
                    self.teacher(extra_outputs[layer][:b], g, l, n).flatten(1)
                    for l, layer in enumerate(layers)
                ],
                dim=1,
            )
            # minimize distance
            equiv_loss = F.cross_entropy(
                outputs[1].flatten(1)[b:], targets, reduction="mean"
            )
            equiv_loss += F.mse_loss(rho_g_phi_x, phi_rho_g_x) * self.equiv_factor
            self.tracker.log_objective(
                equiv_loss.item() * b, (self.mode, self.name, "distance")
            )
            if self.learn_equiv:
                equiv_loss += self.enforce_invertible(
                    shared_memory["inputs"], shared_memory["rho_g_inputs"], g, n
                )
                if self.id_between_filters:
                    equiv_loss += self.prevent_identity_between_filters(
                        batch_size=b
                    )  # maximize this one
                else:
                    equiv_loss += self.prevent_identity(
                        shared_memory["rho_g_inputs"], shared_memory["rho_h_inputs"]
                    )  # maximize this one

            loss += self.gamma * equiv_loss
            self.tracker.log_objective(b, (self.mode, self.name, "normalization"))
            outputs = (
                outputs[0],
                outputs[1][:b],
            )  # throw away outputs for transformed inputs now
        return outputs, loss, targets

    def enforce_invertible(self, inputs, rho_g_inputs, g, n):
        inv_rho_g_x = self.teacher(rho_g_inputs, -g, n=n)
        reg = F.mse_loss(inputs.squeeze(), inv_rho_g_x.squeeze())
        self.tracker.log_objective(
            reg.item() * rho_g_inputs.shape[0], (self.mode, self.name, "inv_reg")
        )
        return reg * self.inv_factor

    def prevent_identity_between_filters(self, batch_size):
        kernels = self.teacher.kernels.flatten(1)
        kernels = F.normalize(kernels, dim=1)
        similarity_matrix = torch.matmul(kernels, kernels.T)
        similarity_matrix = torch.triu(
            similarity_matrix, diagonal=1
        )  # get only entries above diag
        G = similarity_matrix.shape[0]
        reg = torch.sum(torch.abs(similarity_matrix)) / ((G * (G - 1)) / 2)
        self.tracker.log_objective(
            reg.item() * batch_size, (self.mode, self.name, "identity_reg")
        )
        return reg * self.id_factor  # minimize similarity by adding to the loss

    def prevent_identity(self, rho_g_inputs, rho_h_inputs):
        reg = torch.abs(
            F.cosine_similarity(
                rho_h_inputs.flatten(1), rho_g_inputs.flatten(1), dim=1, eps=1e-8
            )
        ).mean()  # minimize similarity by adding to the loss
        self.tracker.log_objective(
            reg.item() * rho_g_inputs.shape[0], (self.mode, self.name, "identity_reg")
        )
        return reg * self.id_factor
