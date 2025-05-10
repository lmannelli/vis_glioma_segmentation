from torch.optim.lr_scheduler import _LRScheduler
import math
class SegResNetScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, alpha, last_epoch=-1):
        self.total_epochs = total_epochs
        self.alpha = alpha
        super(SegResNetScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        factor = (1 - current_epoch / self.total_epochs) ** 0.9
        return [self.alpha * factor for _ in self.optimizer.param_groups]

class WarmupCosineScheduler(_LRScheduler):
    """
    Scheduler con:
      - Warm-up lineal de 0 → initial_lr en las primeras warmup_epochs.
      - Luego cosine annealing de initial_lr → 0 en el resto de total_epochs.
    """
    def __init__(
        self,
        optimizer,
        total_epochs: int,
        initial_lr: float,
        warmup_epochs: int = 10,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            # Warm-up lineal: escala de 0 → initial_lr
            scale = epoch / float(max(1, self.warmup_epochs))
        else:
            # Cosine annealing: initial_lr → 0
            progress = (epoch - self.warmup_epochs) / float(
                max(1, self.total_epochs - self.warmup_epochs)
            )
            # 0 ≤ progress ≤ 1
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.initial_lr * scale for _ in self.optimizer.param_groups]
class PolyDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, initial_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        super(PolyDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        factor = (1 - current_epoch / self.total_epochs) ** 0.9
        return [self.initial_lr + (self.initial_lr * factor) for _ in self.optimizer.param_groups]