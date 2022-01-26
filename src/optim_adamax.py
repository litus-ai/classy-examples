import torch
import transformers
from classy.optim.factories import Factory
from torch.optim import Adamax


class AdamaxFactory(Factory):
    """
    Factory for AdamW optimizer with warmup learning rate scheduler
    reference paper for Adamax: https://arxiv.org/abs/1412.6980

    In order to define your own optimizer, you just have to implement a Factory that when called (using the __call__
    method) returns either only the optimizer or a dict containing the optimizer and the scheduler with the interval
    with which it is updated.

    We suggest to check the PyTorch Lightning documentation for more information. Please note that this call method must
    return the same objects that a LightningModule returns in its "configure_optimizers" method:
    https://pytorch-lightning.readthedocs.io/en/latest/starter/converting.html#move-the-optimizer-s-and-schedulers
    """

    def __init__(
        self, lr: float, warmup_steps: int, total_steps: int, weight_decay: float
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, module: torch.nn.Module):
        optimizer = Adamax(
            module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, self.warmup_steps, self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
