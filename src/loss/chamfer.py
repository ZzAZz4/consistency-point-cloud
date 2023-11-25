from .base import BaseDistanceFunc
from pytorch3d.loss import chamfer_distance
from torch_geometric.utils import to_dense_batch
from torch import nn, Tensor


class ChamferDistance(BaseDistanceFunc):
    def __init__(self, reduce: bool=True) -> None:
        super().__init__()
        self.reduce = reduce

    def __call__(self, lhs: Tensor, rhs: Tensor, batch: Tensor | None) -> Tensor:
        pred = to_dense_batch(lhs, batch)[0]
        target = to_dense_batch(rhs, batch)[0]
        if self.reduce:
            return chamfer_distance(pred, target, batch_reduction='mean')[0]
        else:
            return chamfer_distance(pred, target, batch_reduction=None)[0]
        
