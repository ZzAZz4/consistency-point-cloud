import torch
from pytorch3d.loss import chamfer_distance
from torch_geometric.utils import to_dense_batch

class CDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor):
        pred, target = to_dense_batch(pred, batch)[0], to_dense_batch(target, batch)[0]
        return chamfer_distance(pred, target, batch_reduction='mean')[0]
    
if __name__ == '__main__':
    sample = torch.randn(10240, 3)
    target = torch.randn(10240, 3)
    batch = torch.repeat_interleave(torch.arange(10), 1024)

    loss = CDLoss()
    print(loss(sample, target, batch))