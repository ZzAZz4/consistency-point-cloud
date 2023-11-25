from typing import Sequence
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor


class BaseParametrization(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor, t: Tensor, batch: Tensor) -> Tensor:
        ...
        

class BaseResampler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: Tensor, t: Tensor, batch: Tensor) -> Tensor:
        ...


class EpsilonParametrization(BaseParametrization):
    def __init__(self, min_time: torch.Tensor, data_time: torch.Tensor) -> None:
        super().__init__()
        self.data_time = data_time
        self.min_time = min_time

    def __call__(self, x: Tensor, y: Tensor, t: Tensor, batch: Tensor) -> Tensor:
        factor = (torch.max(t**2 - self.min_time**2, torch.zeros_like(t))**0.5)
        factor = factor[batch, None]
        return x + factor * y 



class SkipParametrization(BaseParametrization):
    def __init__(self, min_time: float, data_time: Tensor) -> None:
        super().__init__()
        self.min_time = min_time
        self.data_time = data_time

    def __call__(self, x: Tensor, y: Tensor, t: Tensor, batch: Tensor) -> Tensor:
        c_skip = self.skip(t)[batch]
        c_out = self.out(t)[batch]
        
        return c_skip * x + c_out * y
    
    def skip(self, t: Tensor) -> Tensor:
        return (self.data_time ** 2) / ((t - self.min_time) ** 2 + (self.data_time ** 2))

    def out(self, t: Tensor) -> Tensor:
        return (t - self.min_time) * self.data_time / (self.data_time**2 + t**2) ** 0.5    

    

class EpsilonResampler(BaseResampler):
    def __init__(self, min_time: float) -> None:
        super().__init__()
        self.min_time = min_time

    def __call__(self, x: Tensor, t: Tensor, batch: Tensor) -> Tensor:
        mul = (t**2 - self.min_time**2)**0.5
        return x + mul[batch, None] * torch.randn_like(x)

from typing import Protocol

class BaseConditionedModel(Protocol):
    def forward(self, x: Tensor, t: Tensor, par: Tensor, batch: Tensor, par_batch: Tensor) -> Tensor:
        ...

    def __call__(self, x: Tensor, t: Tensor, par: Tensor, batch: Tensor, par_batch: Tensor) -> Tensor:
        ...
    


class Consistency(torch.nn.Module):
    def __init__(self, model: BaseConditionedModel, reparametrization: SkipParametrization) -> None:
        super().__init__()
        self.model = model
        self.parametrization = reparametrization

    def forward(self, x: Tensor, t: Tensor, par: Tensor, x_batch: Tensor, par_batch: Tensor) -> Tensor:
        x_div = x / torch.sqrt(t[:, None]**2 + self.parametrization.data_time**2)[x_batch]
        par_div = par / self.parametrization.data_time

        # print(x_sub.shape, par_sub.shape)

        y = self.model(x=x_div, t=t, par=par_div, batch=x_batch, par_batch=par_batch)
        d = self.parametrization(x=x, y=y, t=t[:, None], batch=x_batch)
        return d





