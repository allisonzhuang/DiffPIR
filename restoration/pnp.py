from typing import Callable
from torch import Tensor


def hqs_step(prox_f: Callable, prox_g: Callable, x: Tensor) -> Tensor:
    z = prox_g(x)
    return prox_f(z)


def drs_step(prox_f: Callable, prox_g: Callable, x: Tensor) -> Tensor:
    z = 2 * prox_f(x) - x
    x_hat = 2 * prox_g(z) - z

    return (x + x_hat) / 2
