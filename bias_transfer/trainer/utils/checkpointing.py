import numpy as np

from mlutils.training import copy_state


class Checkpointing:
    def __init__(
        self, model, scheduler, tracker, chkpt_options, maximize_score, call_back=None
    ):
        self.call_back = call_back
        self.model = model
        self.scheduler = scheduler
        self.tracker = tracker
        self.chkpt_options = chkpt_options
        self.maximize_score = maximize_score

    def save(self, epoch, score, patience_counter):
        raise NotImplementedError

    def restore(self, restore_only_state=False):
        raise NotImplementedError


class RemoteCheckpointing(Checkpointing):
    def save(self, epoch, score, patience_counter):
        state = {
            "score": score,
            "maximize_score": self.maximize_score,
            "tracker": self.tracker,
            "patience_counter": patience_counter,
            **self.chkpt_options,
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler
        self.call_back(
            epoch=epoch, model=self.model, state=state,
        )  # save model

    def restore(self, restore_only_state=False):
        loaded_state = {
            "state": { "maximize_score": self.maximize_score,}
        }
        if not restore_only_state:
            loaded_state["state"]["tracker"] = self.tracker
            if self.scheduler is not None:
                loaded_state["scheduler"] = self.scheduler
        self.call_back(
            epoch=-1, model=self.model, state=loaded_state
        )  # load the last epoch if existing
        epoch = loaded_state.get("epoch", 0)
        patience_counter = loaded_state.get("patience_counter", -1)
        return epoch, patience_counter


class LocalCheckpointing(Checkpointing):
    def __init__(
        self, model, scheduler, tracker, chkpt_options, maximize_score, call_back=None
    ):
        super().__init__(
            model, scheduler, tracker, chkpt_options, maximize_score, call_back
        )
        # prepare state save
        self.score_save = np.infty * -1 if maximize_score else np.infty
        self.epoch_save = -1
        self.patience_counter_save = -1
        self.model_save = copy_state(model)
        self.scheduler_save = copy_state(scheduler) if scheduler is not None else {}
        self.tracker_save = copy_state(tracker)

    def save(self, epoch, score, patience_counter):
        self.score_save = score
        self.epoch_save = epoch
        self.patience_counter_save = patience_counter
        self.model_save = copy_state(self.model)
        self.scheduler_save = copy_state(self.scheduler) if self.scheduler else {}
        self.tracker_save = copy_state(self.tracker)

    def restore(self, restore_only_state=False):
        if self.epoch_save > -1:
            self.model.load_state_dict(self.model_save)
            if not restore_only_state:
                if self.scheduler:
                    self.scheduler.load_state_dict(self.scheduler_save)
                self.tracker.load_state_dict(self.tracker_save)
        return self.epoch_save, self.patience_counter_save
