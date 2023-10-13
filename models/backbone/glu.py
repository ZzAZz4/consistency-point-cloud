import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import PositionalEncoding



class ConcatSquashLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, batch: torch.Tensor):
        gate: torch.Tensor = torch.sigmoid(self._hyper_gate(ctx))
        bias: torch.Tensor = self._hyper_bias(ctx)
        ret: torch.Tensor = self._layer(x) * gate[batch] + bias[batch]
        return ret
    

class PointwiseNet(torch.nn.Module):
    def __init__(self, dim_ctx):
        super().__init__()
        self.embedding = PositionalEncoding(dim_ctx)
        self.net = ModuleList([
            ConcatSquashLinear(3, 256, dim_ctx + dim_ctx),
            ConcatSquashLinear(256, 512, dim_ctx + dim_ctx),
            ConcatSquashLinear(512, 512, dim_ctx + dim_ctx),
            ConcatSquashLinear(512, 256, dim_ctx + dim_ctx),
            ConcatSquashLinear(256, 128, dim_ctx + dim_ctx),
        ])
        self.out = ConcatSquashLinear(128, 3, dim_ctx + dim_ctx)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t: torch.Tensor, batch: torch.Tensor):
        ctx2 = self.embedding(t) # this probably shouldn't be here, but it works
        ctx = torch.cat([ctx, ctx2], dim=-1) 
        
        out: torch.Tensor = x
        for layer in self.net:
            out = layer(out, ctx, batch)
            out = torch.nn.functional.silu(out)

        out = self.out(out, ctx, batch)
        return x + out
