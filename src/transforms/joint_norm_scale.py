from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeScale
import torch


class JointNormalizeScale(BaseTransform):
    def __init__(self) -> None:
        self.normalize = NormalizeScale()

    def __call__(self, data: Data):
        joint = Data(
            pos=torch.cat([data.pos, data.y], dim=0)
        )
        joint = self.normalize(joint)
        return Data(
            pos=joint.pos[:data.pos.shape[0]], 
            y=joint.pos[data.pos.shape[0]:]
        )
