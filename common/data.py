import torch
from torch_geometric.data import Data, Batch


class MyData(Data):
    pos: torch.Tensor
    incomplete: torch.Tensor

class MyDataBatched(Data):
    pos: torch.Tensor
    incomplete: torch.Tensor
    pos_batch: torch.Tensor
    incomplete_batch: torch.Tensor