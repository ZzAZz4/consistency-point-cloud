from torch.nn import Linear, SiLU
from torch_geometric.nn import MLP, MaxAggregation
from torch_geometric.nn import PointNetConv
import torch

import torch
from torch_cluster import knn_graph, fps



class PointNetEncoder(torch.nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.conv1 = PointNetConv(
            local_nn=MLP([3 + 3, 32], act=SiLU(), plain_last=True), 
            global_nn=SiLU(), 
            aggr=MaxAggregation()
        )
        self.conv2 = PointNetConv(
            local_nn=MLP([32 + 3, 32], act=SiLU(), plain_last=True), 
            global_nn=SiLU(), 
            aggr=MaxAggregation()
        )
        self.aggr = MaxAggregation()
        self.net = Linear(32, zdim)

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        h: torch.Tensor
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv1(x=pos, pos=pos, edge_index=edge_index)

        index = fps(pos, batch, ratio=0.5)
        h, pos, batch = h[index], pos[index], batch[index]
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        h = self.conv2(x=h, pos=pos, edge_index=edge_index)

        h = self.aggr(h, batch) 
        return self.net(h)