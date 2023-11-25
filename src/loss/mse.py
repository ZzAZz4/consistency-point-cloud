from .base import BaseDistanceFunc
from torch import nn, Tensor

class MSEDistance(BaseDistanceFunc):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, lhs: Tensor, rhs: Tensor, batch: Tensor | None) -> Tensor:
        return self.loss(lhs, rhs)