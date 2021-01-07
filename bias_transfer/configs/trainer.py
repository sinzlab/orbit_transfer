from .base import BaseConfig, baseline
from bias_transfer.tables.nnfabrik import *


class TrainerConfig(BaseConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.img_classification"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.force_cpu = kwargs.pop("force_cpu", False)
        self.optimizer = kwargs.pop("optimizer", "Adam")
        self.optimizer_options = kwargs.pop(
            "optimizer_options", {"amsgrad": False, "lr": 0.0003, "weight_decay": 5e-4}
        )
        self.lr_decay = kwargs.pop("lr_decay", 0.8)
        self.lr_warmup = kwargs.pop("lr_warmup", 0)
        self.epoch = kwargs.pop("epoch", 0)
        self.scheduler = kwargs.pop("scheduler", None)
        self.scheduler_options = kwargs.pop("scheduler_options", {})
        self.chkpt_options = kwargs.pop("chkpt_options", {})
        self.patience = kwargs.pop("patience", 10)
        self.threshold = kwargs.pop("threshold", 0.0001)
        self.verbose = kwargs.pop("verbose", False)
        self.min_lr = kwargs.pop("min_lr", 0.00001)  # lr scheduler min learning rate
        self.threshold_mode = kwargs.pop("threshold_mode", "rel")
        self.train_cycler = kwargs.pop("train_cycler", "LongCycler")
        self.train_cycler_args = kwargs.pop("train_cycler_args", {})
        self.loss_functions = kwargs.pop(
            "loss_functions", {"img_classification": "CrossEntropyLoss"}
        )
        self.loss_weighing = kwargs.pop("loss_weighing", False)
        if (
            len(self.loss_functions) > 1
            or "img_classification" not in self.loss_functions.keys()
        ):
            self.threshold_mode = "abs"
            self.scale_loss = kwargs.pop("scale_loss", True)
            self.avg_loss = kwargs.pop("avg_loss", False)

        self.maximize = kwargs.pop(
            "maximize", True
        )  # if stop_function maximized or minimized
        self.loss_accum_batch_n = kwargs.pop(
            "loss_accum_batch_n", None
        )  # for gradient accumulation how often to call opt.step

        self.mtl = kwargs.pop("mtl", False)

        self.interval = kwargs.pop(
            "interval", 1
        )  # interval at which objective evaluated for early stopping
        self.max_iter = kwargs.pop(
            "max_iter", 100
        )  # maximum number of iterations (epochs)

        self.restore_best = kwargs.pop(
            "restore_best", True
        )  # in case of loading best model at the end of training
        self.lr_decay_steps = kwargs.pop(
            "lr_decay_steps", 3
        )  # Number of times the learning rate should be reduced before stopping the training.

        self.eval_with_bn_train = kwargs.pop("eval_with_bn_train", False)
        # noise
        self.add_noise = kwargs.pop("add_noise", False)
        self.noise_std = kwargs.pop("noise_std", None)
        self.noise_snr = kwargs.pop("noise_snr", None)
        self.noise_test = kwargs.pop(
            "noise_test",
            {
                "noise_snr": [
                    {5.0: 1.0},
                    {4.0: 1.0},
                    {3.0: 1.0},
                    {2.0: 1.0},
                    {1.0: 1.0},
                    {0.5: 1.0},
                    {0.0: 1.0},
                ],
                "noise_std": [
                    {0.0: 1.0},
                    {0.05: 1.0},
                    {0.1: 1.0},
                    {0.2: 1.0},
                    {0.3: 1.0},
                    {0.5: 1.0},
                    {1.0: 1.0},
                ],
            },
        )
        self.apply_noise_to_validation = kwargs.pop("apply_noise_to_validation", True)
        self.noise_adv_classification = kwargs.pop("noise_adv_classification", False)
        self.noise_adv_regression = kwargs.pop("noise_adv_regression", False)
        self.noise_adv_loss_factor = kwargs.pop("noise_adv_loss_factor", 1.0)
        self.noise_adv_gamma = kwargs.pop("noise_adv_gamma", 10.0)
        self.representation_matching = kwargs.pop("representation_matching", None)
        # transfer
        self.synaptic_intelligence_computation = kwargs.pop(
            "synaptic_intelligence_computation", False
        )
        self.freeze = kwargs.pop("freeze", None)
        self.freeze_bn = kwargs.pop("freeze_bn", False)
        self.transfer_restriction = kwargs.pop("transfer_restriction", [])
        self.transfer_after_train = kwargs.pop("transfer_after_train", False)
        self.single_input_stream = kwargs.pop("single_input_stream", True)
        self.readout_name = kwargs.pop("readout_name", "fc")
        self.reset = kwargs.pop("reset", ())
        self.reset_linear_frequency = kwargs.pop("reset_linear_frequency", None)
        self.transfer_from_path = kwargs.pop("transfer_from_path", None)
        self.regularization = kwargs.pop(
            "regularization",
            {},  # {"regularizer": "L2SP/Mixup/RDL/KnowledgeDistillation", "alpha": 1.0, "decay_alpha": True, }
        )
        self.lottery_ticket = kwargs.pop("lottery_ticket", {})
        if self.lottery_ticket:
            self.max_iter = self.lottery_ticket.get(
                "rounds", 1
            ) * self.lottery_ticket.get("round_length", 100)
        self.show_epoch_progress = kwargs.pop("show_epoch_progress", False)
        self.data_transfer = kwargs.pop("data_transfer", False)

    @property
    def main_loop_modules(self):
        modules = []
        if self.representation_matching:
            modules.append("RepresentationMatching")
        elif (
            self.noise_snr or self.noise_std or self.noise_test
        ):  # Logit matching includes noise augmentation
            modules.append("NoiseAugmentation")
        if self.noise_adv_classification or self.noise_adv_regression:
            modules.append("NoiseAdvTraining")
        if self.reset_linear_frequency:
            modules.append("RandomReadoutReset")
        if self.lottery_ticket:
            modules.append("LotteryTicketPruning")
        if self.synaptic_intelligence_computation:
            modules.append("SynapticIntelligence")
        if self.regularization:
            modules.append(self.regularization["regularizer"])
        modules.append("ModelWrapper")
        return modules


class RegressionTrainerConfig(TrainerConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.regression"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.threshold_mode = kwargs.pop("threshold_mode", "rel")
        self.loss_functions = kwargs.pop("loss_functions", {"regression": "MSELoss"})
        self.maximize = False
        self.noise_test = {}
        self.apply_noise_to_validation = False
        # self.show_epoch_progress = kwargs.pop("show_epoch_progress", True)


class TransferTrainerConfig(TrainerConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.transfer"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_transfer = True
        self.save_input = kwargs.pop("save_input", False)
        self.save_representation = kwargs.pop("save_representation", False)
        self.compute_fisher = kwargs.pop("compute_fisher", {})
        self.compute_si_omega = kwargs.pop("compute_si_omega", {})
        self.extract_coreset = kwargs.pop("extract_coreset", {})
        self.reset_for_new_task = kwargs.pop("reset_for_new_task", False)


class TransferTrainerRegressionConfig(RegressionTrainerConfig):
    config_name = "trainer"
    table = Trainer()
    fn = "bias_transfer.trainer.regression_transfer"

    @baseline
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_transfer = True
        self.save_input = kwargs.pop("save_input", False)
        self.save_representation = kwargs.pop("save_representation", False)
        self.compute_fisher = kwargs.pop("compute_fisher", {})
        self.compute_si_omega = kwargs.pop("compute_si_omega", {})
        self.extract_coreset = kwargs.pop("extract_coreset", {})
        self.reset_for_new_task = kwargs.pop("reset_for_new_task", False)
