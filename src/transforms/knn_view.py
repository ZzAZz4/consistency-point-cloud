from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn, fps
import torch
import math


class FPS(BaseTransform):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, data: Data):
        num_nodes = data.num_nodes
        choice = fps(data.pos, ratio=self.n / num_nodes)

        for key, item in data: # type: ignore
            if key == 'num_nodes':
                data.num_nodes = choice.size(0) # type: ignore
            elif (torch.is_tensor(item) and item.size(0) == num_nodes and item.size(0) != 1):
                data[key] = item[choice]
        return data



class KNNSplit(BaseTransform):
    def __init__(self, ratio: float, attr: str = 'par'):
        super().__init__()
        self.ratio = ratio
        self.attr = attr
        assert 0 < ratio < 1

    def __call__(self, data: Data):
        assert data.num_nodes is not None
        n = int(round(data.num_nodes * self.ratio, 0))
        center = data.pos.mean(dim=0)
        scale = (data.pos - center).norm(dim=1).max()

        xyz = torch.randn((3,), dtype=torch.float32)
        sqrt2 = 1.4142135623730951
        xyz = xyz / xyz.norm() * scale * sqrt2 + center 

        closest = knn(data.pos, xyz[None], n)[1]
        data = data.clone()
        data.__setattr__(self.attr, data.pos[closest])
        return data
        

from dataclasses import dataclass

@dataclass
class Partition:
    par: torch.Tensor
    rest: torch.Tensor


@dataclass
class PartitionedData(Data):
    par: Partition


class KNNPartition(BaseTransform):
    def __init__(self, n: int, attr: str = 'par'):
        super().__init__()
        self.n = n
        self.attr = attr

    def __call__(self, data: Data):
        center = data.pos.mean(dim=0)
        scale = (data.pos - center).norm(dim=1).max() * 1.4142135623730951

        viewpoint = torch.randn((3,), dtype=torch.float32)
        viewpoint = viewpoint / viewpoint.norm() * scale + center 

        closest = knn(data.pos, viewpoint[None], self.n)[1] # index vector
        nonclosest = torch.ones(data.pos.size(0), dtype=torch.bool)
        nonclosest[closest] = False # mask vector
        
        closest_points = data.pos[closest]
        nonclosest_points = data.pos[nonclosest]

        data = data.clone()
        data.__setattr__(self.attr, Partition(closest_points, nonclosest_points))
        return data
        