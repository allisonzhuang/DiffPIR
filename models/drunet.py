from typing import Dict

from torch import Tensor
from deepinv import models

from interfaces import DenoiserPrior


class DRUNet(DenoiserPrior):
    def __init__(self) -> None:
        super().__init__()
        self.net = models.DRUNet()

    def denoise(
        self, x_t: Tensor, t: int, noise_schedule: Dict[str, Tensor]
    ) -> Tensor:
        sigma = noise_schedule["sigma_bar"][t]
        return self.net(x_t, sigma)
