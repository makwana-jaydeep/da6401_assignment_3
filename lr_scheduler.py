"""
Noam LR Scheduler

paper ref:
"Attention Is All You Need"
https://arxiv.org/abs/1706.03762

lr formula used:

lrate = d_model^(-0.5) *
        min(step^(-0.5),
            step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    implementation of noam scheduler from transformer paper.

    lr first increases during warmup and after that
    slowly starts decreasing.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:

        # storing params for lr computation later
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        # parent class init
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        """
        calculate scaling factor based on current step.
        """

        # step cant be zero otherwise pow issue happens
        step = max(1, self.last_epoch + 1)

        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,

            # warmup phase
            step * (self.warmup_steps ** -1.5),
        )

        return scale

    def get_lr(self) -> list:
        """
        returns lr for each optimizer param group.
        """

        scale = self._get_lr_scale()

        # multiplying original base lr with noam scale
        updated_lrs = [
            base_lr * scale
            for base_lr in self.base_lrs
        ]

        return updated_lrs


def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list:
    """
    helper fn to visualize how lr changes over time.
    """

    # dummy layer just for attaching optimizer
    dummy_model = torch.nn.Linear(1, 1)

    optimizer = optim.Adam(
        dummy_model.parameters(),
        lr=1.0,
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
    )

    lr_history = []

    # simulate training steps
    for _ in range(total_steps):

        curr_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(curr_lr)

        optimizer.step()
        scheduler.step()

    return lr_history


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20_000

    # generate lr values
    lrs = get_lr_history(
        D_MODEL,
        WARMUP_STEPS,
        TOTAL_STEPS,
    )

    plt.figure(figsize=(9, 4))

    # plotting lr curve
    plt.plot(lrs)

    # warmup boundary
    plt.axvline(
        WARMUP_STEPS,
        color="red",
        linestyle="--",
        label=f"warmup={WARMUP_STEPS}"
    )

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")

    plt.title(
        f"Noam Schedule plot (d_model={D_MODEL})"
    )

    plt.legend()

    # avoids label cutoff sometimes
    plt.tight_layout()

    plt.show()