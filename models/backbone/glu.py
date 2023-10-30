import torch
from torch.nn import Linear, ModuleList
from torch_geometric.nn import PositionalEncoding



class GLU(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(GLU, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, batch: torch.Tensor):
        gate: torch.Tensor = torch.sigmoid(self._hyper_gate(ctx))
        bias: torch.Tensor = self._hyper_bias(ctx)
        ret: torch.Tensor = self._layer(x) * gate[batch] + bias[batch]
        return ret
    

class GLUDecoder(torch.nn.Module):
    def __init__(self, dim_ctx, residual=True):
        super().__init__()
        self.residual = residual
        self.embedding = PositionalEncoding(dim_ctx)
        self.net = ModuleList([
            GLU(3 + dim_ctx, 512, dim_ctx + dim_ctx),
            GLU(512, 256, dim_ctx + dim_ctx),
            GLU(256, 128, dim_ctx + dim_ctx),
        ])
        self.out = GLU(128, 3, dim_ctx + dim_ctx)

    def forward(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, batch: torch.Tensor):
        out: torch.Tensor = torch.cat([x, ctx[batch]], dim=-1)
        ctx2 = self.embedding(t) # this probably shouldn't be here, but it works
        ctx = torch.cat([ctx, ctx2], dim=-1) 
        
        for layer in self.net:
            out = layer(out, ctx, batch)
            out = torch.nn.functional.silu(out)

        out = self.out(out, ctx, batch)
        if self.residual:
            return x + out
        else:
            return out
        

    def __call__(self, x: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, batch: torch.Tensor):
        return super().__call__(x, t, ctx, batch)