from torch import nn
import torch
import torch.nn.functional as F


from .representation import RepresentationRegularization

import torchvision
import numpy as np

import matplotlib.pyplot as plt

from .. import RDL


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn(model, test_loader, device, max_g, l):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        if l != 0:
            data = data[:3].reshape(1, 1, 3, 28, 28).repeat(max_g, 1, 1, 1, 1)
        else:
            data = data[:1].reshape(1, 1, 28, 28).repeat(max_g, 1, 1, 1)
        g = torch.arange(max_g)

        input_tensor = data.cpu()
        print(input_tensor.shape)
        transformed_input_tensor = model(data, g=g, l=l)[0].cpu()
        if l != 0:
            transformed_input_tensor = transformed_input_tensor.squeeze()
            input_tensor = input_tensor.squeeze()
        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        axarr[1].imshow(out_grid)
        axarr[1].set_title("Transformed Images")


class EquivarianceTransfer(RepresentationRegularization):
    def __init__(self, trainer):
        super().__init__(trainer, name="Equiv")
        objectives = {
            "Training": {
                "Equiv": {
                    "CE": 0,
                    "distance": 0,
                    "inv_reg": 0,
                    "identity_reg": 0,
                    "normalization": 0,
                }
            },
            "Validation": {
                "Equiv": {
                    "CE": 0,
                    "distance": 0,
                    "inv_reg": 0,
                    "identity_reg": 0,
                    "normalization": 0,
                }
            },
            "Test": {
                "Equiv": {
                    "CE": 0,
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
        self._ce_factor = self.ce_factor = self.config.regularization.get(
            "ce_factor", 1.0
        )
        self._equiv_factor = self.equiv_factor = self.config.regularization.get(
            "equiv_factor", 1.0
        )
        self._inv_factor = self.inv_factor = self.config.regularization.get(
            "inv_factor", 1.0
        )
        self._id_factor = self.id_factor = self.config.regularization.get(
            "id_factor", 1.0
        )
        self.id_between_filters = self.config.regularization.get(
            "id_between_filters", False
        )
        self.id_between_transforms = self.config.regularization.get(
            "id_between_transforms", False
        )
        self.max_stacked_transform = self.config.regularization.get(
            "max_stacked_transform", 1
        )
        self.hinge_epsilon = self.config.regularization.get("hinge_epsilon", 0.5)
        self.mse_dist = self.config.regularization.get("mse_dist", False)
        self.ramp_up = self.config.regularization.get("ramp_up", {})
        self.exclude_inv_from_id_loss = self.config.regularization.get(
            "exclude_inv_from_id_loss", False
        )
        self.inv_for_all_layers = self.config.regularization.get(
            "inv_for_all_layers", False
        )
        self.visualize = self.config.regularization.get("visualize", False)

    def pre_epoch(self, model, mode, **options):
        super().pre_epoch(model, mode, **options)
        for name in ("ce", "inv", "id", "equiv"):
            final_val = getattr(self, f"_{name}_factor")
            if name in self.ramp_up and self.epoch <= self.ramp_up[name]:
                step_size = final_val / self.ramp_up[name]
                current_val = step_size * self.epoch
                setattr(self, f"{name}_factor", current_val)
            else:
                setattr(self, f"{name}_factor", final_val)

    def post_epoch(self, model):
        # Visualize the STN transformation on some input batch
        if self.visualize:
            for l in range(1):
                print(f"Layer {l}")
                visualize_stn(
                    self.teacher,
                    self.train_loader["img_classification"],
                    device="cuda",
                    max_g=self.group_size,
                    l=l,
                )
                plt.ioff()
                plt.show()

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
            if self.learn_equiv and (
                not self.id_between_filters or self.id_between_transforms
            ):
                h = (g + torch.randint(1, self.group_size, (b,))) % self.group_size
                if self.exclude_inv_from_id_loss:
                    minus_g = (-g) % self.group_size
                    h = torch.where(h == minus_g, h + 1, h)
                inputs = inputs.repeat(2, 1, 1, 1)
                g = torch.cat([g, h], dim=0)
                shared_memory["h"] = h
            # apply group representation on input
            rho_g_inputs, transform_g = self.teacher(inputs, g=g, l=0, n=n)
            if self.learn_equiv:
                if not self.id_between_filters or self.id_between_transforms:
                    rho_g_inputs, rho_h_inputs = (
                        rho_g_inputs[:b],
                        rho_g_inputs[b:],
                    )
                    shared_memory["rho_h_inputs"] = rho_h_inputs
                    inputs = inputs[:b]
                    g = g[:b]
                if self.id_between_transforms:
                    transform_g, transform_h = (
                        transform_g[:b],
                        transform_g[b:],
                    )
                    shared_memory["transform_g"] = transform_g
                    shared_memory["transform_h"] = transform_h
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
            rho_g_phi_x = [
                self.teacher(extra_outputs[layer][:b], g=g, l=l + 1, n=n)[0]
                for l, layer in enumerate(layers)
            ]
            # minimize distance
            equiv_loss = self.ce_factor * F.cross_entropy(
                outputs[1].flatten(1)[b:], targets, reduction="mean"
            )
            self.tracker.log_objective(
                equiv_loss.item() * b, (self.mode, self.name, "CE")
            )
            mse = (
                F.mse_loss(
                    torch.cat([x.flatten(1) for x in rho_g_phi_x], dim=1), phi_rho_g_x
                )
                * self.equiv_factor
            )
            self.tracker.log_objective(
                mse.item() * b, (self.mode, self.name, "distance")
            )
            equiv_loss = equiv_loss + mse
            if self.learn_equiv:
                equiv_loss += self.enforce_invertible(
                    shared_memory["inputs"],
                    shared_memory["rho_g_inputs"],
                    g=g,
                    n=n,
                    l=0,
                )
                if self.inv_for_all_layers:
                    # invertibility loss for each layer
                    for l, layer in enumerate(layers):
                        equiv_loss += self.enforce_invertible(
                            extra_outputs[layer][:b], rho_g_phi_x[l], g=g, n=n, l=l + 1
                        )
                if self.id_between_filters:
                    equiv_loss += self.prevent_identity_between_filters(batch_size=b)
                else:
                    equiv_loss += self.prevent_identity(
                        shared_memory["rho_g_inputs"], shared_memory["rho_h_inputs"]
                    )
                if self.id_between_transforms:
                    equiv_loss += self.prevent_identity(
                        shared_memory["transform_g"], shared_memory["transform_h"]
                    )

            loss += self.gamma * equiv_loss
            self.tracker.log_objective(b, (self.mode, self.name, "normalization"))
            outputs = (
                outputs[0],
                outputs[1][:b],
            )  # throw away outputs for transformed inputs now
        return outputs, loss, targets

    def enforce_invertible(self, inputs, rho_g_inputs, g, n, l=0):
        inv_rho_g_x = self.teacher(rho_g_inputs, -g, n=n, l=l)[0]
        reg = F.mse_loss(inputs.squeeze(), inv_rho_g_x.squeeze())
        self.tracker.log_objective(
            reg.item() * rho_g_inputs.shape[0], (self.mode, self.name, "inv_reg")
        )
        return reg * self.inv_factor

    def prevent_identity_between_filters(self, batch_size):
        if hasattr(self.teacher, "kernels"):
            kernels = self.teacher.kernels.flatten(1)
        else:
            kernels = self.teacher.theta[0]
        kernels = F.normalize(kernels, dim=1)
        if self.mse_dist:
            similarity_matrix = RDL.compute_mse_matrix(kernels, kernels)
        else:
            similarity_matrix = torch.matmul(kernels, kernels.T)
        G = similarity_matrix.shape[0]
        normalization = (G * (G - 1)) / 2  # upper triangle
        if self.mse_dist and self.hinge_epsilon:
            similarity_matrix = torch.max(
                self.hinge_epsilon - similarity_matrix,
                torch.zeros_like(similarity_matrix),
            )
            similarity_matrix = (similarity_matrix * 10) ** 2
            similarity_matrix = torch.triu(
                similarity_matrix, diagonal=1
            )  # get only entries above diag
        else:
            similarity_matrix = torch.triu(
                similarity_matrix, diagonal=1
            )  # get only entries above diag
            if self.exclude_inv_from_id_loss:
                idx = torch.arange(G)
                idx = (idx * -1) % G
                similarity_matrix[torch.arange(G), idx] = 0
                normalization -= G // 2  # removing -g
            similarity_matrix = torch.abs(similarity_matrix)
        reg = torch.sum(similarity_matrix) / normalization
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
