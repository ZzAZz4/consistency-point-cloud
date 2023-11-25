from dataclasses import dataclass
import torch


@dataclass
class ConsistencySettings:
    training_iterations: int = 100_000
    min_time_partitions: int = 1
    max_time_partitions: int = 150
    initial_ema_decay: float = 0.99
    min_time: float = 1e-4
    data_time: torch.Tensor = torch.tensor(1.0)
    max_time: float = 1.0
    rho: float = 7.0
    w_min: float = 2.0
    w_max: float = 14.0