import torch
import torch.nn as nn
from torch import Tensor
import math
import typing as ty
from dataclasses import dataclass


class PointConsistencyModel(nn.Module):
    def __init__(self, model: nn.Module, sigma_data=0.5, sigma_min=0.0002):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.epsilon = sigma_min

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        f = self.model(x, t)
        c_out = self.c_out(t)
        c_skip = self.c_skip(t)
        out = c_out * f 
        out += c_skip * x
        return out

    def c_skip(self, t: Tensor) -> Tensor:
        return torch.div(self.sigma_data**2, (t - self.epsilon)**2 + self.sigma_data**2)
    
    def c_out(self, t: Tensor) -> Tensor:
        return torch.div(self.sigma_data * (t - self.epsilon), torch.sqrt(self.sigma_data**2 + t**2))
    


@dataclass
class PointConsistencySettings:
    training_iterations: int
    target_disc_steps: tuple[int, int] = (1, 150)
    initial_ema_decay_rate: float = 0.95
    initial_timesteps: int = 2
    sigma_range: tuple[float, float] = (0.002, 8.0)
    rho: float = 7.0


class PointConsistencyTraining(nn.Module):
    def __init__(self, settings: PointConsistencySettings):
        super().__init__()
        self.conf = settings
        
    def step_schedule_n(self, k: float) -> int:
        s, K = self.conf.target_disc_steps, self.conf.training_iterations

        num_timesteps = (s[1] + 1)**2 - s[0]**2
        num_timesteps = k * num_timesteps / K
        num_timesteps = num_timesteps + s[0]**2
        num_timesteps = math.sqrt(num_timesteps)
        num_timesteps = math.ceil(-1. + num_timesteps)
        return 1 + num_timesteps

    def ema_decay_rate_schedule_mu(self, n_k: int) -> float:
        s, mu_0 = self.conf.target_disc_steps, self.conf.initial_ema_decay_rate
        
        return math.exp(s[0] * math.log(mu_0) / float(n_k))

    def karras_schedule_t(self, n_k: int, device: torch.device | None = None) -> Tensor:
        (eps, T), rho = self.conf.sigma_range, self.conf.rho

        rho_inv = 1.0 / rho
        steps = torch.arange(n_k, device=device) / max(n_k - 1, 1)
        sigmas = eps**rho_inv + steps * (T**rho_inv - eps**rho_inv)
        sigmas = sigmas**rho
        return sigmas

    def ema_decay_rate_schedule(self, num_timesteps: int) -> float:
        return math.exp(
            (self.conf.initial_timesteps * math.log(self.conf.initial_ema_decay_rate)) / num_timesteps
        )

    def train_step(
        self, 
        iteration: int, 
        x: Tensor, 
        model: PointConsistencyModel, 
        ema_model: PointConsistencyModel
    ):
        num_timesteps = self.step_schedule_n(iteration)
        sigmas = self.karras_schedule_t(num_timesteps, device=x.device)
        noise = torch.randn_like(x)

        timesteps = torch.randint(0, num_timesteps - 1, (x.shape[0], ), device=x.device)
        current_sigmas = sigmas[timesteps]
        next_sigmas = sigmas[timesteps + 1]

        next_x = x + (noise * next_sigmas)
        next_x = model(next_x, next_sigmas)

        with torch.no_grad():
            current_x = x + (noise * current_sigmas)
            current_x = ema_model(current_x, current_sigmas)

        return next_x, current_x

    def after_train_step(self, iteration: int, model: PointConsistencyModel, ema_model: PointConsistencyModel):
        num_timesteps = self.step_schedule_n(iteration)
        ema_decay_rate = self.ema_decay_rate_schedule(num_timesteps)
        self._update_ema_weights(
            ema_model.parameters(), model.parameters(), ema_decay_rate
        )
        return ema_model

    def _update_ema_weights(
        self,
        ema_weight_iter: ty.Iterator[Tensor],
        online_weight_iter: ty.Iterator[Tensor],
        ema_decay_rate: float,
    ) -> None:
        for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
            if ema_weight.data is None:
                ema_weight.data.copy_(online_weight.data)
            else:
                ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)

