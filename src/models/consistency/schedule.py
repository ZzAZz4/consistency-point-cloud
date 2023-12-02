import math
from abc import ABC, abstractmethod
import torch


class BaseStepSchedule(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, iteration: int) -> int:
        ...


class TimestepSchedule(BaseStepSchedule):
    def __init__(
        self,
        training_iterations: int, 
        initial_timesteps: int, 
        final_timesteps: int
    ) -> None:
        super().__init__()
        self.training_iterations = training_iterations
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

    def __call__(self, iteration: int) -> int:
        num_timesteps = (self.final_timesteps + 1)**2 - self.initial_timesteps**2
        num_timesteps = iteration * num_timesteps / self.training_iterations
        num_timesteps = num_timesteps + self.initial_timesteps**2
        num_timesteps = math.sqrt(num_timesteps)
        num_timesteps = math.ceil(-1. + num_timesteps)
        return 1 + num_timesteps


class BaseTimeSchedule(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, up_to: int, device: torch.device | None = None) -> torch.Tensor:
        ...


class KarrasTimeSchedule(BaseTimeSchedule):
    def __init__(self, min_time: float, max_time: float, rho: float) -> None:
        super().__init__()
        self.sigma_range = (min_time, max_time)
        self.rho = rho

    def __call__(self, up_to: int, device: torch.device | None = None) -> torch.Tensor:
        (min_time, max_time), rho = self.sigma_range, self.rho
        rho_inv = 1.0 / rho
        steps = torch.arange(up_to, device=device) / max(up_to - 1, 1)
        sigmas = min_time**rho_inv + steps * (max_time**rho_inv - min_time**rho_inv)
        sigmas = sigmas**rho
        return sigmas


class BaseIndexSampler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, num_timesteps: int, sigmas: torch.Tensor, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        ...


class UniformSampler(BaseIndexSampler):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, num_timesteps: int, sigmas: torch.Tensor, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randint(0, num_timesteps - 1, (batch_size,), device=device)



class ImprovedTimestepSchedule(BaseStepSchedule):
    def __init__(self, 
                 training_iterations: int, 
                 initial_timesteps: int, 
                 final_timesteps: int):
        self.training_iterations = training_iterations
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps

    def __call__(self, iteration: int) -> int:
        """Computes: N(k)"""
        total_training_steps_prime = math.floor(
            self.training_iterations
            / (math.log2(math.floor(self.final_timesteps / self.initial_timesteps)) + 1)
        )
        num_timesteps = self.initial_timesteps * math.pow(
            2, math.floor(iteration / total_training_steps_prime)
        )
        num_timesteps = min(num_timesteps, self.final_timesteps) + 1
        return int(num_timesteps)


class LognormalSampler(BaseIndexSampler):
    def __init__(self, 
                 mean: float,
                 std: float):
        self.mean = mean
        self.std = std

    def __call__(self, num_timesteps: int, sigmas: torch.Tensor, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        pdf = torch.erf((torch.log(sigmas[1:]) - self.mean) / (self.std * math.sqrt(2))) - \
            torch.erf((torch.log(sigmas[:-1]) - self.mean) / (self.std * math.sqrt(2)))
        pdf = pdf / pdf.sum()

        timesteps = torch.multinomial(pdf, batch_size, replacement=True)
        return timesteps.to(device)


class TimeScheduler:
    def __init__(self, step_schedule: BaseStepSchedule, time_schedule: BaseTimeSchedule, index_sampler: BaseIndexSampler):
        self.step_schedule = step_schedule
        self.time_schedule = time_schedule
        self.index_sampler = index_sampler

    def get_times(self, iteration: int, batch_size: int, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        num_timesteps = self.step_schedule(iteration)
        sigmas = self.time_schedule(num_timesteps, device=device)
        time_indices = self.index_sampler(num_timesteps, sigmas, batch_size, device=device)

        current_times = sigmas[time_indices]
        next_times = sigmas[time_indices + 1]

        return current_times, next_times
