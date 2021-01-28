import torch
from torch.nn import functional as F


def update_input(self, input, output):
    """
    Used to register forward hook
    Args:
        self:
        input:
        output:

    Returns:

    """
    self.input = input[0].data
    self.output = output


def logistic_hessian(f):
    """
    We only calculate the diagonal elements of the hessian
    """
    f = f[:, :]
    pi = torch.sigmoid(f)
    return pi * (1 - pi)


def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s * s


def full_softmax_hessian(f):
    """
    Calculate the full softmax hessian
    """
    s = F.softmax(f, dim=-1)
    e = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    return s[:, :, None] * e[None, :, :] - s[:, :, None] * s[:, None, :]


def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # check if in same gpu
            warn = param.get_device() != old_param_device
        else:  # check if in cpu
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device


def parameters_to_matrix(parameters):
    param_device = None
    mat = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        m = param.shape[0]
        mat.append(param.view(m, -1))
    return torch.cat(mat, dim=-1)


def parameter_grads_to_vector(parameters):
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        if param.grad is None:
            raise ValueError("gradient not available")
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec, dim=-1)


def vector_to_parameter_grads(vec, parameters):
    r"""Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer : pointer + num_param].view_as(param).grad

        # Increment the pointer
        pointer += num_param
