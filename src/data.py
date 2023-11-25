import torch
from torch_geometric.data import Data, Batch
from dataclasses import dataclass


@dataclass
class BatchedData(Data):
    par: torch.Tensor
    pos_batch: torch.Tensor
    par_batch: torch.Tensor

