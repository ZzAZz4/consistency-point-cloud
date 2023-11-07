from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn
import torch



class KNNSplit(BaseTransform):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def __call__(self, data: Data):
        center = data.pos.mean(dim=0)
        scale = (data.pos - center).norm(dim=1).max()

        xyz = torch.randn((3,), dtype=torch.float32)
        sqrt2 = 1.4142135623730951
        xyz = xyz / xyz.norm() * scale * sqrt2 + center 

        closest = knn(data.pos, xyz[None], self.n)[1]
        data = data.clone()
        data.incomplete = data.pos[closest]
        return data