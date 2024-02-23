from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ConstantLRwithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_step: int,
        last_epoch: int = -1,
        eta_min: float = 0,
    ) -> None:
        self.warmup_step = warmup_step
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch <= self.warmup_step:
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_step)
                for base_lr in self.base_lrs
            ]
        return self.base_lrs
