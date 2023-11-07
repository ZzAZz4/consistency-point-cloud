from torch.nn import Linear, SiLU
from torch_geometric.nn import MLP, MaxAggregation
from torch_geometric.nn import PointNetConv
import torch

import torch
from torch_cluster import knn_graph, fps



class PointNetEncoder(torch.nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.conv1 = PointNetConv(MLP([3 + 3, 64], act=SiLU(), plain_last=False),)
        self.conv2 = PointNetConv(MLP([64 + 3, 128, zdim], act=SiLU(), plain_last=False),)
        self.aggr = MaxAggregation()

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        h: torch.Tensor
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv1(x=pos, pos=pos, edge_index=edge_index)

        index = fps(pos, batch, ratio=0.5)
        h, pos, batch = h[index], pos[index], batch[index]
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv2(x=(h, h), pos=pos, edge_index=edge_index)

        h = self.aggr(h, batch) 
        return h
    
    def __call__(self, pos: torch.Tensor, batch: torch.Tensor):
        return super().__call__(pos, batch)

