from typing import Dict

from bias_transfer.configs.base import BaseConfig


class LotteryTicketMixin(BaseConfig):
    def __init__(self, **kwargs):
        self.load_kwargs(**kwargs)

        self.lottery_ticket: Dict = {}
        if self.lottery_ticket:
            self.max_iter = self.lottery_ticket.get(
                "rounds", 1
            ) * self.lottery_ticket.get("round_length", 100)
            self.main_loop_modules.append("LotteryTicketPruning")

        super().__init__(**kwargs)
