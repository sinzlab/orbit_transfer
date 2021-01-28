from torch import nn
import torch
import torch.nn.functional as F

from bias_transfer.trainer.main_loop_modules.function_regularization.fromp_utils import (
    update_input,
    logistic_hessian,
    full_softmax_hessian,
    parameters_to_matrix,
    parameter_grads_to_vector,
    vector_to_parameter_grads,
)
from bias_transfer.trainer.main_loop_modules.main_loop_module import MainLoopModule
from mlutils.training import eval_state


class FROMP(MainLoopModule):
    """
    Adapted from https://github.com/team-approx-bayes/fromp
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        self.prior_prec = self.config.regularization.get("prior_prec")
        self.grad_clip_norm = self.config.regularization.get("grad_clip_norm")
        self.alpha = self.config.regularization.get("alpha")
        if self.prior_prec < 0.0:
            raise ValueError(f"invalid prior precision: {self.prior_prec}")
        if (self.grad_clip_norm is not None) and (not self.grad_clip_norm >= 0.0):
            raise ValueError(f"invalid gradient clip norm: {self.grad_clip_norm}")
        if self.alpha < 0.0:
            raise ValueError(f"invalid alpha: {self.alpha}")
        self.covariance = torch.tensor(
            self.trainer.data_loaders.pop("covariance"), device=self.device
        )
        main_task = next(iter(self.trainer.data_loaders["train"].keys()))
        self.memorable_points_prev = (
            self.trainer.data_loaders["train"].pop(f"{main_task}_cs").dataset.samples
        )
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.train_modules = []
        self.set_train_modules(self.model, self.train_modules)

        self.init_task(self.config.regularization.get("eps", 1e-5))

    def init_task(self, eps):
        """
        Calculate values (memorable_logits, hkh_l) for regularisation term (all but the first task)

        """
        self.kernel_inv_prev_mem_prev_model = []
        covariance = 1.0 / (self.covariance + self.prior_prec)

        with eval_state(self.model):
            memorable_data_prev = self.memorable_points_prev.to(self.device)
            self.optimizer.zero_grad()
            logits_prev_mem = self.model.forward(memorable_data_prev)

        num_classes = logits_prev_mem.shape[-1]
        if num_classes == 1:
            preds_prev_mem = torch.sigmoid(logits_prev_mem)
        else:
            preds_prev_mem = torch.softmax(logits_prev_mem, dim=-1)
        self.preds_prev_mem_prev_model = preds_prev_mem.detach()

        # Calculate kernel = J \Sigma J^T for all memory points, and store via cholesky decomposition
        intermediate_outputs = []
        for module in self.train_modules:
            intermediate_outputs.append(module.output)
        for class_id in range(num_classes):
            loss_for_class = preds_prev_mem[:, class_id].sum()
            retain_graph = (
                True if class_id < num_classes - 1 else None
            )  # only clean up the graph after the last class
            grad = self.calculate_grad(
                loss_for_class,
                intermediate_outputs,
                self.train_modules,
                retain_graph=retain_graph,
            )
            kernel = (
                torch.einsum("ij,j,pj->ip", grad, covariance, grad)
                + torch.eye(grad.shape[0], dtype=grad.dtype, device=grad.device) * eps
            )
            self.kernel_inv_prev_mem_prev_model.append(
                torch.cholesky_inverse(torch.cholesky(kernel))
            )

    @classmethod
    def set_train_modules(cls, module, train_modules):
        """
        For calculating Jacobians in PyTorch
        """
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                train_modules.append(module)
                module.register_forward_hook(update_input)
        else:
            for child in list(module.children()):
                cls.set_train_modules(child, train_modules)

    @classmethod
    def calculate_grad(
        cls, loss, intermediate_outputs, train_modules, retain_graph=None
    ):
        """
        Calculate the gradient (part of calculating Jacobian) of the parameters lc wrt loss
        """
        linear_grad = torch.autograd.grad(
            loss, intermediate_outputs, retain_graph=retain_graph
        )
        grad = []
        for i, module in enumerate(train_modules):
            g = linear_grad[i]
            a = module.input.clone().detach()
            m = a.shape[0]

            if isinstance(module, nn.Linear):
                grad.append(torch.einsum("ij,ik->ijk", g, a))
                if module.bias is not None:
                    grad.append(g)

            if isinstance(module, nn.Conv2d):
                a = F.unfold(
                    a,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                _, k, hw = a.shape
                _, c, _, _ = g.shape
                g = g.view(m, c, -1)
                grad.append(torch.einsum("ijl,ikl->ijk", g, a))
                if module.bias is not None:
                    a = torch.ones((m, 1, hw), device=a.device)
                    grad.append(torch.einsum("ijl,ikl->ijk", g, a))

            if isinstance(module, nn.BatchNorm1d):
                grad.append(torch.mul(g, a))
                if module.bias is not None:
                    grad.append(g)

            if isinstance(module, nn.BatchNorm2d):
                grad.append(torch.einsum("ijkl->ij", torch.mul(g, a)))
                if module.bias is not None:
                    grad.append(torch.einsum("ijkl->ij", g))

        grad_m = parameters_to_matrix(grad)
        return grad_m.detach()

    @classmethod
    def calculate_jacobian(cls, output, intermediate_outputs, train_modules):
        """
        Calculate the Jacobian matrix
        """
        if output.dim() > 2:
            raise ValueError("the dimension of output must be smaller than 3.")
        else:  # output.dim() == 2:
            num_classes = output.shape[1]
        grad = []
        for i in range(num_classes):
            retain_graph = None if i == num_classes - 1 else True
            loss = output[:, i].sum()
            g = cls.calculate_grad(
                loss,
                intermediate_outputs,
                train_modules=train_modules,
                retain_graph=retain_graph,
            )
            grad.append(g)
        result = torch.zeros(
            (grad[0].shape[0], grad[0].shape[1], num_classes),
            dtype=grad[0].dtype,
            device=grad[0].device,
        )
        for i in range(num_classes):
            result[:, :, i] = grad[i]
        return result

    @classmethod
    def compute_covariance(cls, data, model):
        """
        After training on a new task, update the coviarance matrix estimate
        """
        train_modules = []
        cls.set_train_modules(model, train_modules)

        logits = model.forward(data)

        intermediate_outputs = []
        for module in train_modules:
            intermediate_outputs.append(module.output)

        jacobian = cls.calculate_jacobian(logits, intermediate_outputs, train_modules)
        if logits.shape[-1] == 1:
            hessian = logistic_hessian(logits).detach()
            hessian = hessian[:, :, None]
        else:
            hessian = full_softmax_hessian(logits).detach()
        return torch.einsum("ijd,idp,ijp->j", jacobian, hessian, jacobian)

    def post_backward(self, model):
        parameters = self.model.parameters()
        grad = parameter_grads_to_vector(parameters).detach()
        grad *= 1 / self.alpha

        grad_func_reg = torch.zeros_like(
            grad
        )  # The gradient corresponding to memorable points
        # compute predictions of memorable points (from previous task)
        with eval_state(self.model):
            memorable_data_prev = self.memorable_points_prev.to(self.device)
            self.optimizer.zero_grad()
            logits_prev_mem = self.model.forward(memorable_data_prev)

        num_classes = logits_prev_mem.shape[-1]
        if num_classes == 1:
            preds_prev_mem = torch.sigmoid(logits_prev_mem)
        else:
            preds_prev_mem = torch.softmax(logits_prev_mem, dim=-1)

        # collect all intermediate outputs:
        intermediate_outputs = []
        for module in self.train_modules:
            intermediate_outputs.append(module.output)

        # compute function loss for each output class:
        for class_id in range(num_classes):
            # \Lambda * Jacobian
            loss_for_class = preds_prev_mem[:, class_id].sum()
            retain_graph = (
                True if class_id < num_classes - 1 else None
            )  # only clean up the graph after the last class
            jacobian_t = self.calculate_grad(
                loss_for_class,
                intermediate_outputs,
                self.train_modules,
                retain_graph=retain_graph,
            )

            # m_t - m_{t-1}
            delta_preds = (
                preds_prev_mem[:, class_id].detach()
                - self.preds_prev_mem_prev_model[:, class_id]
            )

            # K_{t-1}^{-1}
            kernel_inv_prev = self.kernel_inv_prev_mem_prev_model[class_id]

            # Uncomment the following line for L2 variants of algorithms
            # kernel_inv_t = torch.eye(kernel_inv_t.shape[0], device=kernel_inv_t.device)

            # Calculate K_{t-1}^{-1} (m_t - m_{t-1})
            kinvf_t = torch.squeeze(
                torch.matmul(kernel_inv_prev, delta_preds[:, None]), dim=-1
            )

            grad_func_reg += torch.einsum("ij,i->j", jacobian_t, kinvf_t)

        grad += grad_func_reg

        # Do gradient norm clipping
        if self.grad_clip_norm is not None:
            grad_norm = torch.norm(grad)
            grad_norm = (
                1.0
                if grad_norm < self.grad_clip_norm
                else grad_norm / self.grad_clip_norm
            )
            grad /= grad_norm

        vector_to_parameter_grads(grad, parameters)
