from torch.nn.functional import silu
from torch_geometric.nn import MLP, AttentionalAggregation
from torch_geometric.nn import DynamicEdgeConv
import torch


class AttnEdgeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=8):
        super().__init__()
        aggr = AttentionalAggregation(MLP([out_channels, 1], act=silu, plain_last=False))
        self.conv = DynamicEdgeConv(MLP([2 * in_channels, out_channels], act=silu, plain_last=False), k=k, aggr=aggr) # type: ignore
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        return self.conv(x, batch)

class AttDGCNNEncoder(torch.nn.Module):
    def __init__(self, zdim, k=8):
        super().__init__()
        self.conv1 = AttnEdgeConv(3, 32, k=k)
        self.conv2 = AttnEdgeConv(32, 32, k=k)
        self.conv3 = AttnEdgeConv(32, 64, k=k)
        self.conv4 = AttnEdgeConv(64, 128, k=k)
        self.shared = MLP([256, 512], act=silu, plain_last=False)
        self.aggr = AttentionalAggregation(MLP([512, 512], act=silu, plain_last=False))
        self.out = MLP([512, zdim], act=silu, plain_last=False)
        
    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        x = pos
        x_1 = self.conv1(x, batch)
        x_2 = self.conv2(x_1, batch)
        x_3 = self.conv3(x_2, batch)
        x_4 = self.conv4(x_3, batch)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = self.aggr(self.shared(x), batch)
        print(x.shape)
        x = self.out(x)
        return x
    
    def __call__(self, pos: torch.Tensor, batch: torch.Tensor):
        return super().__call__(pos, batch)

