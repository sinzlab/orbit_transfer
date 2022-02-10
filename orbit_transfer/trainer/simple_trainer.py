import time, copy
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn

from ..configs.trainer.simple_trainer import SimpleTrainerConfig
from ..models.mnist_1d import (
    StaticLearnedEquivariance,
    model_fn,
)
from .forward_methods import (
    ce_forward,
    equiv_transfer_forward,
    equiv_learn_forward,
    kd_forward,
    kd_match_forward,
    rdl_forward,
    attention_forward,
    cka_forward,
)
from nntransfer.models.utils import freeze_params
import random


def accuracy(model, inputs, targets):
    if isinstance(model, StaticLearnedEquivariance):
        return 0.0
    targets = targets.cpu().numpy().astype(np.float32)
    true_pred = 0
    if inputs.shape[0] > 1000:
        for b in range(inputs.shape[0] // 1000):
            upper = min(1000 * (b + 1), inputs.shape[0])
            preds = (
                model(inputs[1000 * b : upper].to("cuda"))[1]
                .argmax(-1)
                .detach()
                .cpu()
                .numpy()
            )
            true_pred += sum(preds == targets[1000 * b : upper])
    else:
        preds = model(inputs.to("cuda"))[1].argmax(-1).detach().cpu().numpy()
        true_pred = sum(preds == targets)
    return 100 * true_pred / len(targets)


def test_model(dataset, model, device="cuda"):
    x_test = torch.Tensor(dataset["x_test"]).to(device)
    y_test = torch.LongTensor(dataset["y_test"]).to(device)
    result = {}
    result["acc"] = accuracy(model.to(device), x_test, y_test)
    result["loss"] = F.cross_entropy(model(x_test)[1], y_test, reduction="mean").item()
    return result


def compute_measures(x, y, forward, model, teacher_model, gamma):
    result = {}
    result["acc"] = accuracy(model, x, y)

    total_loss = 0
    total_reg = 0
    if x.shape[0] > 500:
        for b in range(x.shape[0] // 500):
            upper = min(500 * (b + 1), x.shape[0])
            loss, reg = forward(
                model,
                x[500 * b : upper].to("cuda"),
                y[500 * b : upper].to("cuda"),
                teacher_model,
            )
            total_loss += loss.detach().item()
            total_reg += reg.detach().item()
    else:
        loss, reg = forward(model, x.to("cuda"), y.to("cuda"), teacher_model)
        total_loss += loss.detach().item()
        total_reg += reg.detach().item()
    result["ce_loss"] = total_loss
    result["reg"] = total_reg
    result["loss"] = (1.0 - gamma) * total_loss + gamma * total_reg
    return result


def full_eval(
    train_data,
    test_data,
    test_all_data,
    model,
    teacher,
    rep_layer,
    teacher_rep_layer,
    gamma,
    forward,
    device="cuda",
):
    result = {}

    def helper(data, postfix=""):
        x_train = torch.Tensor(data["x"])
        y_train = torch.LongTensor(data["y"])
        result["train" + postfix] = compute_measures(
            x_train,
            y_train,
            forward,
            model,
            teacher,
            gamma,
        )
        x_val = torch.Tensor(data["x_validation"])
        y_val = torch.LongTensor(data["y_validation"])
        result["validation" + postfix] = compute_measures(
            x_val, y_val, forward, model, teacher, gamma
        )
        x_test = torch.Tensor(data["x_test"])
        y_test = torch.LongTensor(data["y_test"])
        result["test" + postfix] = compute_measures(
            x_test, y_test, forward, model, teacher, gamma
        )

    helper(train_data, postfix="")
    helper(test_data, postfix="_shift")
    helper(test_all_data, postfix="_all")
    return result


def train(
    dataloaders,
    model,
    forward_fct,
    config,
    teacher_model=None,
):
    model = model.to(config.device)
    rep_dist = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=config.lr_decay, patience=config.patience
    )
    dataset = dataloaders["train"]
    x_train, x_val = torch.Tensor(dataset["x"]), torch.Tensor(dataset["x_validation"])
    y_train, y_val = torch.LongTensor(dataset["y"]), torch.LongTensor(
        dataset["y_validation"]
    )

    if teacher_model is not None:
        teacher_model = teacher_model.to(config.device)
        freeze_params(teacher_model)
        teacher_model.eval()
    x_train, x_val, y_train, y_val = [
        v.to(config.device) for v in [x_train, x_val, y_train, y_val]
    ]

    results = []
    t0 = time.time()
    best_obj = 1e10 if config.select_on_loss else 0
    best_model = None
    patience_counter = 0
    decay_counter = 0
    for step in range(config.total_steps + 1):
        bix = (step * config.batch_size) % len(x_train)  # batch index
        x, y = (
            x_train[bix : bix + config.batch_size],
            y_train[bix : bix + config.batch_size],
        )
        if config.augment:
            shift = random.randint(0, config.augment)
            x = torch.roll(x, shift, dims=-1)
        loss, regularizer = forward_fct(model, x, y, teacher_model=teacher_model)

        loss = (1.0 - config.gamma) * loss + config.gamma * regularizer

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (
            config.eval_every > 0 and step % config.eval_every == 0
        ):  # evaluate the model

            results.append(
                full_eval(
                    dataloaders["train"],
                    dataloaders["test"],
                    dataloaders["test_all"],
                    model,
                    teacher_model,
                    config.rep_layer,
                    config.teacher_rep_layer,
                    config.gamma,
                    forward_fct,
                    device="cuda",
                )
            )
            if config.select_on_loss:
                val_obj = results[-1]["validation"]["loss"]
                maximize = -1
            else:
                val_obj = results[-1]["validation"]["acc"]
                maximize = 1

            if val_obj * maximize > best_obj * maximize and step > 0:
                best_obj = val_obj
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.patience:
                print("Reduce lr!")
                if decay_counter >= config.lr_decay_steps:
                    print("Early stopping!")
                    break
                decay_counter += 1
                patience_counter = 0
            scheduler.step(val_obj)

        if step % config.print_every == 0:  # print out training progress
            t1 = time.time()
            print(
                f"step {step}, dt {t1 - t0}s, train_loss {results[-1]['train']['loss']}, "
                f"val_loss {results[-1]['validation']['loss']}, val_ce {results[-1]['validation']['ce_loss']}, "
                f"train_reg {results[-1]['train']['reg']}, val_reg {results[-1]['validation']['reg']}, "
                f"train_acc {results[-1]['train']['acc']}, val_acc {results[-1]['validation']['acc']}"
            )
            t0 = t1

        # if (
        #     config.checkpoint_every > 0 and step % config.checkpoint_every == 0
        # ):  # save model checkpoints
        #     model.step = step
        #     results["checkpoints"].append(copy.deepcopy(model.state_dict()))
    # results["best_model"] = best_model
    if config.restore_best:
        print("restoring best model")
        model.load_state_dict(best_model)
    return results


def trainer_fn(
    model, dataloaders, seed, uid, cb, eval_only=False, student_model=None, **kwargs
):
    if student_model:
        teacher_model = model
        model = model_fn(dataloaders, seed, **student_model)
    else:
        teacher_model = None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    config = SimpleTrainerConfig.from_dict(kwargs)
    print("Running: ", config.forward)
    if config.forward == "kd":
        forward = kd_forward
    elif config.forward == "kd_match":
        forward = kd_match_forward
    elif config.forward == "rdl":
        forward = rdl_forward
    elif config.forward == "cka":
        forward = cka_forward
    elif config.forward == "attention":
        forward = attention_forward
    elif config.forward == "equiv_transfer":
        forward = equiv_transfer_forward
    elif config.forward == "equiv_learn":
        forward = equiv_learn_forward
    else:
        forward = ce_forward
    forward = partial(forward, config=config)
    results = train(
        dataloaders,
        model,
        forward,
        teacher_model=teacher_model,
        config=config,
    )
    final_results = full_eval(
        dataloaders["train"],
        dataloaders["test"],
        dataloaders["test_all"],
        model,
        teacher_model,
        config.rep_layer,
        config.teacher_rep_layer,
        config.gamma,
        forward,
        device="cuda",
    )
    results = {"final": final_results, "progress": results}
    print(results["final"])
    if not isinstance(model, StaticLearnedEquivariance):
        return results["final"]["test_all"]["acc"], results, model.state_dict()
    else:
        return results["final"]["test_all"]["loss"], results, model.state_dict()
