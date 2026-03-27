from typing import Any, Callable, Dict
from math import sqrt, floor

import torch
from torch import Tensor

from configs import SolverConfig, DiffusionConfig
from interfaces import DenoiserPrior, PnPSolver


def build_noise_scheduler(cfg: DiffusionConfig):
    T = cfg.T

    beta_min = cfg.beta_start
    beta_max = cfg.beta_end

    beta = [0.0] * (T + 1)
    alpha = [0.0] * (T + 1)
    alpha_bar = [1.0] * (T + 1)
    sigma_bar = [0.0] * (T + 1)

    for t in range(T + 1):
        beta[t] = beta_min + t * (beta_max - beta_min) / T
        alpha[t] = 1 - beta[t]
        alpha_bar[t] = alpha_bar[t - 1] * alpha[t]
        sigma_bar[t] = sqrt((1 - alpha_bar[t]) / alpha_bar[t])

    return {
        "T": T,
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "sigma_bar": sigma_bar,
    }


def diffpir_restore(
    cfg: SolverConfig,
    y: Tensor,
    prior: DenoiserPrior,
    pnp_solver: PnPSolver,
    pnp_step: Callable,
    noise_scheduler: Dict[str, Any],
) -> Tensor:
    lambda_ = cfg.lambda_
    sigma = cfg.sigma_n
    zeta = cfg.zeta
    n_steps = cfg.n_steps

    T = noise_scheduler["T"]
    timesteps = [
        max(1, floor((sqrt(T) * i / n_steps) ** 2))
        for i in range(1, n_steps + 1)
    ]

    x = torch.randn_like(y)

    alpha_bar = noise_scheduler["alpha_bar"]
    sigma_bar = noise_scheduler["sigma_bar"]

    rev_ts = list(reversed(timesteps))
    for i, t in enumerate(rev_ts):

        t_prev = rev_ts[i + 1] if i + 1 < n_steps else 0
        rho = lambda_ * sigma**2 / (sigma_bar[t] ** 2)

        prox_f = lambda x, _rho=rho: pnp_solver.data_step(x, y, _rho)
        prox_g = lambda z, _t=t: prior.denoise(z, _t, noise_scheduler)

        x_hat = pnp_step(prox_f, prox_g, x)

        eps_hat = (x - sqrt(alpha_bar[t]) * x_hat) / sqrt(1 - alpha_bar[t])
        eps = torch.randn_like(y)

        x = sqrt(alpha_bar[t_prev]) * x_hat + sqrt(1 - alpha_bar[t_prev]) * (
            sqrt(1 - zeta) * eps_hat + sqrt(zeta) * eps
        )

    return x
