from .base import BaseDistanceFunc
from torch import nn, Tensor
from torch_geometric.utils import to_dense_batch

from geomloss import SamplesLoss


class SinkhornEMDistance(BaseDistanceFunc):
    def __init__(self, reduce: bool = True):
        self.loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.01)
        self.reduce = reduce
        
    def __call__(self, lhs: Tensor, rhs: Tensor, batch: Tensor | None) -> Tensor:
        lhs, rhs = to_dense_batch(lhs, batch)[0], to_dense_batch(rhs, batch)[0]
        if self.reduce:
            return self.loss(lhs, rhs).mean()
        else:
            return self.loss(lhs, rhs)