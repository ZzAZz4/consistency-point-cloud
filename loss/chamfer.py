import torch
from pytorch3d.loss import chamfer_distance
from torch_geometric.utils import to_dense_batch


class CDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, batch, reduction='mean'):
        pred, target = to_dense_batch(pred, batch)[0], to_dense_batch(target, batch)[0]
        return chamfer_distance(pred, target, batch_reduction=reduction)[0]
    
