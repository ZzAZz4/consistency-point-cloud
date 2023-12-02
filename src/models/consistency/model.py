from typing import Sequence
from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
import logging

class BaseParametrization(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def skip(self, t: Tensor) -> Tensor:
        ...
        
    @abstractmethod
    def out(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def in_(self, t: Tensor) -> Tensor:
        ...

class BaseResampler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: Tensor, t: Tensor, batch: Tensor) -> Tensor:
        ...


class KarrasParametrization(BaseParametrization):
    def __init__(self, min_time: float, data_time: torch.Tensor) -> None:
        super().__init__()
        self.data_time = data_time
        self.min_time = min_time

    def skip(self, t: Tensor) -> Tensor:
        return (self.data_time ** 2) / ((t - self.min_time) ** 2 + (self.data_time ** 2))

    def out(self, t: Tensor) -> Tensor:
        return (t - self.min_time) * self.data_time / (self.data_time**2 + t**2) ** 0.5   
    
    def in_(self, t: Tensor) -> Tensor:
        return 1. / (t**2 + self.data_time**2)**0.5


class EpsilonParametrization(BaseParametrization):
    def __init__(self, min_time: float, data_time: torch.Tensor) -> None:
        super().__init__()
        self.data_time = data_time
        self.min_time = min_time

    def skip(self, t: Tensor) -> Tensor:
        return (self.data_time ** 2) / ((t - self.min_time) ** 2 + (self.data_time ** 2))

    def out(self, t: Tensor) -> Tensor:
        return (t - self.min_time) * self.data_time / (self.data_time**2 + t**2) ** 0.5   
    
    def in_(self, t: Tensor) -> Tensor:
        return 1. / (t**2 + self.data_time**2)**0.5

    

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
    

from dataclasses import dataclass

@dataclass
class ConsistencyOutput:
    model: Tensor
    output: Tensor



class Consistency(torch.nn.Module):
    def __init__(self, model: BaseConditionedModel, reparametrization: BaseParametrization) -> None:
        super().__init__()
        self.model = model
        self.parametrization = reparametrization

    def forward(self, x: Tensor, t: Tensor, par: Tensor, x_batch: Tensor, par_batch: Tensor) -> ConsistencyOutput:
        # c_in = self.parametrization.in_(t[:, None])
        c_out = self.parametrization.out(t[:, None])
        c_skip = self.parametrization.skip(t[:, None])

        y = self.model(x=x, t=t, par=par, batch=x_batch, par_batch=par_batch)
        d = c_skip[x_batch] * x + c_out[x_batch] * y

        d = d - d.mean(dim=0)
        
        return ConsistencyOutput(model=y, output=d)
    
    def __call__(self, x: Tensor, t: Tensor, par: Tensor, x_batch: Tensor, par_batch: Tensor) -> ConsistencyOutput:
        return super().__call__(x=x, t=t, par=par, x_batch=x_batch, par_batch=par_batch)





