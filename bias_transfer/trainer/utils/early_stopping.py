import numpy as np

from bias_transfer.trainer.utils import StopClosureWrapper


def early_stopping(
    model,
    objective_closure,
    config,
    optimizer,
    interval=5,
    patience=20,
    # start=0,
    max_iter=1000,
    maximize=True,
    tolerance=1e-5,
    switch_mode=True,
    restore_best=True,
    tracker=None,
    scheduler=None,
    lr_decay_steps=1,
    checkpointing=None,
):
    def _objective():
        if switch_mode:
            model.eval()
        ret = objective_closure()
        if switch_mode:
            model.train(training_status)
        return ret

    def decay_lr():
        if restore_best:
            restored_epoch, _ = checkpointing.restore(restore_only_state=True, action="best")
            print(
                "Restoring best model from epoch {} after lr-decay!".format(
                    restored_epoch
                )
            )

    def finalize():
        if restore_best:
            restored_epoch, _ = checkpointing.restore(restore_only_state=True, action="best")
            print("Restoring best model from epoch! {}".format(restored_epoch))
        else:
            print("Final best model! objective {}".format(best_objective))

    training_status = model.training
    objective_closure = StopClosureWrapper(objective_closure)

    # Try loading saved checkpoint:
    epoch, patience_counter = checkpointing.restore(action="last")
    # turn into a sign
    maximize = -1 if maximize else 1
    best_objective = current_objective = _objective()

    if scheduler is not None:
        if (
            config.scheduler == "adaptive"
        ):  # only works sofar with one task but not with MTL
            scheduler.step(
                list(current_objective.values())[0][
                    "eval" if config.maximize else "loss"
                ]
            )

    for repeat in range(lr_decay_steps):
        while patience_counter < patience and epoch < max_iter:
            for _ in range(interval):
                epoch += 1
                if tracker is not None:
                    tracker.log_objective(current_objective)

                # if not tracker.check_isfinite():
                #     print("Objective is not Finite. Stopping training")
                #     finalize(model, best_state_saved, current_objective, best_objective)
                #     return
                yield epoch, current_objective

            current_objective = _objective()

            # if a scheduler is defined, a .step with the current objective is all that is needed to reduce the LR
            if scheduler is not None:
                if (config.scheduler == "adaptive") and (
                    not config.scheduler_options["mtl"]
                ):  # only works sofar with one task but not with MTL
                    # scheduler.step(list(current_objective.values())[0]['eval' if config.maximize else 'loss'])
                    scheduler.step(current_objective)
                elif config.scheduler == "manual":
                    scheduler.step()

            def test_current_obj(obj, best_obj):
                obj_key = "eval" if config.maximize else "loss"
                result = [
                    obj[task][obj_key] * maximize
                    < best_obj[task][obj_key] * maximize - tolerance
                    for task in obj.keys()
                ]
                return np.array(result)

            # if (test_current_obj(current_objective, best_objective)).all():
            if maximize * current_objective < maximize * best_objective:
                tracker.log_objective(
                    patience_counter, keys=("Validation", "patience",)
                )
                # print(
                #     "Validation [{:03d}|{:02d}/{:02d}] ---> {}".format(
                #         epoch, patience_counter, patience, current_objective
                #     ),
                #     flush=True,
                # )
                patience_counter = -1
                best_objective = current_objective
            else:
                patience_counter += 1
                tracker.log_objective(
                    patience_counter, keys=("Validation", "patience",)
                )
                # print(
                #     "Validation [{:03d}|{:02d}/{:02d}] -/-> {}".format(
                #         epoch, patience_counter, patience, current_objective
                #     ),
                #     flush=True,
                # )
            checkpointing.save(
                epoch=epoch, score=current_objective, patience_counter=patience_counter,
            )  # save model

        if (epoch < max_iter) & (lr_decay_steps > 1) & (repeat < lr_decay_steps):
            if (config.scheduler == "adaptive") and (
                config.scheduler_options["mtl"]
            ):  # adaptive lr scheduling for mtl alongside early_stopping
                scheduler.step()
            decay_lr()

        patience_counter = -1

    finalize()