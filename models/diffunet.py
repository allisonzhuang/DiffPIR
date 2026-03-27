from typing import Dict
from math import sqrt

import torch
from torch import Tensor
from deepinv import models

from interfaces import DenoiserPrior


class DiffUNet(DenoiserPrior):
    def __init__(self) -> None:
        super().__init__()
        self.net = models.DiffUNet()

    def denoise(
        self, x_t: Tensor, t: int, noise_schedule: Dict[str, Tensor]
    ) -> Tensor:
        alpha_bar = noise_schedule["alpha_bar"]

        # deepinv DiffUNet uses 0-indexed timesteps [0, 999]; our schedule is [1, 1000]
        eps = self.net.forward_diffusion(x_t, torch.tensor([t - 1]))[:, :3]
        x0 = (x_t - sqrt(1 - alpha_bar[t]) * eps) / sqrt(alpha_bar[t])

        return x0
