from torch.nn import Linear, SiLU
from torch_geometric.nn import MLP, MaxAggregation, AttentionalAggregation
from torch_geometric.nn import PointNetConv
import torch

import torch
from torch_cluster import knn_graph, fps


class AttnPointNetConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=8):
        super().__init__()
        aggr = AttentionalAggregation(MLP([out_channels, 1], act=SiLU(), plain_last=False))
        self.conv = PointNetConv(MLP([3 + in_channels, out_channels], act=SiLU(), plain_last=False), k=k, aggr=aggr) # type: ignore
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index):
        return self.conv(x=x, pos=pos, edge_index=edge_index)


class AttnPointNetEncoder(torch.nn.Module):
    def __init__(self, zdim, k=16):
        super().__init__()
        self.k = k
        self.conv1 = AttnPointNetConv(3, 64)
        self.conv2 = AttnPointNetConv(64, 128)
        self.conv3 = AttnPointNetConv(128, zdim)
        self.aggr = AttentionalAggregation(MLP([zdim, zdim], act=SiLU(), plain_last=False))

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        h: torch.Tensor
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        h = self.conv1(x=pos, pos=pos, edge_index=edge_index)

        index = fps(pos, batch, ratio=0.5)
        h, pos, batch = h[index], pos[index], batch[index]
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        h = self.conv2(x=(h, h), pos=pos, edge_index=edge_index)

        index = fps(pos, batch, ratio=0.5)
        h, pos, batch = h[index], pos[index], batch[index]
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        h = self.conv3(x=(h, h), pos=pos, edge_index=edge_index)

        h = self.aggr(h, batch) 
        return h
    
    def __call__(self, pos: torch.Tensor, batch: torch.Tensor):
        return super().__call__(pos, batch)

