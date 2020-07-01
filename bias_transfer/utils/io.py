import os
import torch
from nnfabrik.utility.nn_helpers import load_state_dict


def restore_saved_state(
    model,
    saved_state,
    ignore_missing=False,
    ignore_unused=False,
    ignore_dim_mismatch=False,
    match_names=False,
    restriction=None,
):
    print("==> Loading model..", flush=True)
    if isinstance(saved_state, (str, os.path)):
        assert os.path.isfile(saved_state), "Error: no model file found!"
        state_dict = torch.load(saved_state)
    else:
        state_dict = saved_state
    if restriction:
        state_dict = {k: state_dict[k] for k in restriction if k in state_dict}
    load_state_dict(
        model=model,
        state_dict=state_dict,
        ignore_missing=ignore_missing,
        match_names=match_names,
        ignore_dim_mismatch=ignore_dim_mismatch,
        ignore_unused=ignore_unused,
    )
    return model


def load_checkpoint(path, model, optimizer=None, ignore_missing=False):
    assert os.path.isfile(path), "Error: no checkpoint file found!"
    checkpoint = torch.load(path)
    load_state_dict(
        model=model,
        state_dict=checkpoint["net"],
        ignore_missing=ignore_missing,
        match_names=True,
        ignore_dim_mismatch=True,
    )
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("==> Loading checkpoint from epoch {}".format(start_epoch), flush=True)
    return model, best_acc, start_epoch


def save_checkpoint(model, optimizer, acc, epoch, path, name):
    print("==> Saving..", flush=True)
    state = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "epoch": epoch,
    }
    if not os.path.isdir(path):
        os.mkdir(path)
    torch.save(state, os.path.join(path, name))
