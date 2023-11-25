from abc import ABC, abstractmethod
from torch import Tensor


class BaseDistanceFunc(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def __call__(self, lhs: Tensor, rhs: Tensor, batch: Tensor | None) -> Tensor:
        ...
    