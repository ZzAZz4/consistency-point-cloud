from abc import ABC, abstractmethod
import copy
import math
from torch import nn, Tensor, no_grad


class BaseEMADecay(ABC):
    @abstractmethod
    def __call__(self, iteration: int) -> float:
        ...

    def __init__(self) -> None:
        super().__init__()


class ExponentialDecay(BaseEMADecay):
    def __init__(self, initial_decay: float, training_iterations: int) -> None:
        super().__init__()
        self.initial_decay = initial_decay
        self.training_iterations = training_iterations

    def __call__(self, iteration: int) -> float:
        return math.exp(
            iteration * math.log(self.initial_decay) / self.training_iterations
        )
    


class EMAModel:
    def __init__(self, model: nn.Module, decay: BaseEMADecay) -> None:
        super().__init__()
        self.model = copy.deepcopy(model)
        self.ema_decay = decay
        self.model.eval()

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    @no_grad()
    def update(self, model: nn.Module, iteration: int):
        alpha = self.ema_decay(iteration)
        for p, ema_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data = alpha * ema_p.data + (1 - alpha) * p.data
        