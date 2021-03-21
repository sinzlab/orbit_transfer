from copy import copy

import torch
import numpy as np
from tqdm import tqdm

from nntransfer.dataset.dataset_classes.npy_dataset import NpyDataset
from bias_transfer.trainer.main_loop_modules.function_regularization.fromp_utils import (
    logistic_hessian,
    softmax_hessian,
)


def extract_coreset(
    data_loader,
    method,
    size,
    model,
    seed,
    device,
    initial_method="",
    remove_from_data=True,
    save_trainset=False,
    **kwargs
):
    print(f"Extracting Coreset using {method}")
    collected_inputs = []
    collected_labels = []
    for src, trg in data_loader:
        collected_inputs.append(src)
        collected_labels.append(trg)
    inputs = torch.cat(collected_inputs).numpy()
    labels = torch.cat(collected_labels).numpy()
    indices = list(range(len(inputs)))
    if "k-center" in (method, initial_method):
        coreset_idx, remain_idx = k_center(inputs, indices, size)
    elif "fromp" in (method, initial_method):
        coreset_idx, remain_idx = select_memorable_points(
            inputs, labels, model, size, device, **kwargs
        )
    elif "random_class_balanced" in (method, initial_method):
        coreset_idx, remain_idx = random_class_balanced(labels, indices, seed, size)
    else:  # "random":
        coreset_idx, remain_idx = random(indices, seed, size)
    if method == "frcl":  # needs an initial extraction run
        coreset_idx, remain_idx = find_best_inducing_points(
            inputs, model, size, coreset_idx, remain_idx, device, **kwargs
        )
    if save_trainset:
        if not remove_from_data:
            remain_idx = list(range(len(inputs)))
        return {
            "source": inputs[remain_idx],
            "source_cs": inputs[coreset_idx],
            "target": labels[remain_idx],
            "target_cs": labels[coreset_idx],
        }
    else:
        return {
            "source_cs": inputs[coreset_idx],
            "target_cs": labels[coreset_idx],
        }


def random(indices, seed, size):
    np.random.seed(seed)
    np.random.shuffle(indices)
    coreset_idx, remain_idx = indices[:size], indices[size:]
    return coreset_idx, remain_idx


def random_class_balanced(labels, indices, seed, size):
    np.random.seed(seed)
    np.random.shuffle(indices)
    num_classes = max(labels) + 1
    size_per_class = size / num_classes
    labels_selected = {l: 0 for l in range(num_classes)}
    coreset_idx = []
    remain_idx = []
    for i, idx in enumerate(indices):
        if len(coreset_idx) >= size:
            remain_idx += indices[i:]
            break
        if labels_selected[labels[idx]] >= size_per_class:
            remain_idx.append(idx)
            continue
        labels_selected[labels[idx]] += 1
        coreset_idx.append(labels[idx])
    return coreset_idx, remain_idx


def k_center(dataset, indices, size):
    def update_distance(dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i, :] - x_train[current_id, :])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists

    dists = np.full(dataset.shape[0], np.inf)
    current_id = 0
    coreset_idx = []
    remain_idx = indices
    for _ in range(size):
        dists = update_distance(dists, dataset, current_id)
        coreset_idx.append(current_id)
        remain_idx.remove(current_id)
        current_id = np.argmax(dists)
    return coreset_idx, remain_idx


def calculate_induce_quality_statistic(idx, dataset, model, device):
    """
    Calculates trace statistic of inducing quality
    (up to multiplication by prior variance)
    """
    statistic = 0

    full_dataset_loader = torch.utils.data.DataLoader(
        NpyDataset(samples=dataset, targets=dataset),
        batch_size=500,
        shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        phi_z = model.core_forward(torch.tensor(dataset[idx]).to(device))
        k_zz = phi_z @ phi_z.T
        inv_k_zz = torch.inverse(k_zz + torch.eye(k_zz.shape[0]).to(device) * 1e-3)
        for x_batch, _ in full_dataset_loader:
            phi_x = model.core_forward(x_batch.to(device))
            k_xz = phi_x @ phi_z.T
            k_xx = phi_x @ phi_x.T
            statistic += torch.trace(k_xx - k_xz @ inv_k_zz @ k_xz.T).cpu()
    return statistic


def find_best_inducing_points(
    dataset,
    model,
    size,
    coreset_idx,
    remain_idx,
    device,
    max_iter=300,
    early_stop_num_iter=80,
    verbose=True,
):

    """Sequentially adds a new point instead of a random one in
    the initial set of inducing points, if the value of the statstic
    above lessens, and does not do anything otherwise.
    - start_inducing_set: list of points to start from
    - max_iter: maximum number of tries to add a point
    """
    score = calculate_induce_quality_statistic(coreset_idx, dataset, model, device)
    new_point_counter = 0
    early_stop_counter = 0
    for i in range(max_iter):
        add_point = np.random.randint(0, len(remain_idx))
        remove_point = np.random.randint(0, size)
        coreset_idx_new = copy(coreset_idx)
        coreset_idx_new[remove_point] = remain_idx[add_point]
        score_new = calculate_induce_quality_statistic(
            coreset_idx_new, dataset, model, device
        )
        if score_new < score:
            remain_idx[add_point] = coreset_idx[remove_point]
            score, coreset_idx = score_new, coreset_idx_new
            new_point_counter += 1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if verbose and i % 10 == 0:
            print("Iteration {} out of {} is in progress".format(i, max_iter))
            print("Current best statistic is ", round(score.item(), 3))
            print("New points added ", new_point_counter, "\n")
        if early_stop_counter == early_stop_num_iter:
            print("Early stop activated!")
            break
    return coreset_idx, remain_idx


def select_memorable_points(
    inputs,
    labels,
    model,
    size,
    device,
    descending=True,
):
    """
    Select memorable points ordered by their lambda values (descending=True picks most important points)
    Adapted from
    """
    batch_size = 500
    dataloader = torch.utils.data.DataLoader(
        NpyDataset(samples=inputs, targets=labels),
        batch_size=batch_size,
        shuffle=False,
    )
    num_classes = max(labels) + 1
    num_points_per_class = int(size / num_classes)
    scores = {class_id: [] for class_id in range(num_classes)}
    idx = {class_id: [] for class_id in range(num_classes)}
    indices = torch.tensor(list(range(inputs.shape[0])))
    # collect scores
    for i, (data, target) in tqdm(enumerate(dataloader)):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, inputs.shape[0])
        data = data.to(device)
        f = model.forward(data)
        if f.shape[-1] > 1:
            lamb = softmax_hessian(f)
            lamb = torch.sum(lamb, dim=-1)
        else:
            lamb = logistic_hessian(f)
            lamb = torch.squeeze(lamb, dim=-1)
        lamb = lamb.detach().cpu()  # hessian serves as a proxy for noise precision
        for class_id in range(num_classes):
            idx[class_id].append(indices[batch_start:batch_end][target == class_id])
            scores[class_id].append(lamb[target == class_id])

    # sort by scores
    coreset_idx = []
    remain_idx = []
    for class_id in range(num_classes):
        idx[class_id] = torch.cat(idx[class_id], dim=0)
        scores[class_id] = torch.cat(scores[class_id], dim=0)
        _, indices = scores[class_id].sort(descending=descending)

        coreset_idx.append(idx[class_id][indices[:num_points_per_class]])
        remain_idx.append(idx[class_id][indices[num_points_per_class:]])

    return torch.cat(coreset_idx), torch.cat(remain_idx)
