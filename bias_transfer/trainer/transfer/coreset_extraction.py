from copy import copy

import torch
import numpy as np

from bias_transfer.dataset.dataset_classes.npy_dataset import NpyDataset


def extract_coreset(
    data_loader, method, size, model, seed, device, initial_method="", remove_from_data=True, **kwargs
):
    print(method)
    collected_inputs = []
    collected_labels = []
    for src, trg in data_loader:
        collected_inputs.append(src)
        collected_labels.append(trg)
    inputs = torch.cat(collected_inputs).numpy()
    labels = torch.cat(collected_labels).numpy()
    indices = list(range(len(inputs)))
    if method == "k-center" or initial_method == "k-center":
        coreset_idx, remain_idx = k_center(inputs, indices, size)
    else:  # method == "random" or initial_method == "random":
        coreset_idx, remain_idx = random(indices, seed, size)
    if method == "frcl":
        coreset_idx, remain_idx = find_best_inducing_points(
            inputs, model, size, coreset_idx, remain_idx, device, **kwargs
        )
    if not remove_from_data:
        remain_idx = list(range(len(inputs)))
    outputs = {
        "source": inputs[remain_idx],
        "source_cs": inputs[coreset_idx],
        "target": labels[remain_idx],
        "target_cs": labels[coreset_idx],
    }
    return outputs


def random(indices, seed, size):
    np.random.seed(seed)
    np.random.shuffle(indices)
    coreset_idx, remain_idx = indices[:size], indices[size:]
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
        NpyDataset(samples=dataset, targets=dataset), batch_size=500, shuffle=False,
    )

    phi_z = model.core_forward(torch.tensor(dataset[idx]).to(device))
    k_zz = phi_z @ phi_z.T
    inv_k_zz = torch.inverse(k_zz + torch.eye(k_zz.shape[0]).to(device) * 1e-3)
    for x_batch, _ in full_dataset_loader:
        phi_x = model.core_forward(x_batch.to(device))
        k_xz = phi_x @ phi_z.T
        k_xx = phi_x @ phi_x.T
        statistic += torch.trace(k_xx - k_xz @ inv_k_zz @ k_xz.T)
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
