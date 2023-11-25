
import torch
from torch import nn, Tensor
from torch.distributions import Normal



class LinearResBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
            nn.SiLU(),
        )
        self.residual = nn.Linear(dim_in, dim_out)


    def forward(self, x: Tensor) -> Tensor:
        return self.model(x) + self.residual(x)
    

class SharedEncoder(nn.Module):
    def __init__(self, encoder, global_feat_size=512, shared_feat_size=128):
        super().__init__()
        self.encoder = encoder
        self.complete_linear = LinearResBlock(global_feat_size, shared_feat_size)
        self.incomplete_linear = LinearResBlock(global_feat_size, shared_feat_size)
        self.code_linear = LinearResBlock(shared_feat_size, global_feat_size)

    def forward(
        self, 
        ctx_pos: Tensor, 
        ctx_batch: Tensor,  
        is_complete: bool = False
    ) -> Tensor:
        
        encoding = self.encoder(pos=ctx_pos, batch=ctx_batch)
        encoding_2: Tensor
        if is_complete:
            encoding_2 = self.complete_linear(encoding)
        else:
            encoding_2 = self.incomplete_linear(encoding)
        
        return encoding + self.code_linear(encoding_2)

    def __call__(
        self,
        ctx_pos: Tensor, 
        ctx_batch: Tensor,  
        is_complete: bool = False
    ) -> Tensor:
        return self.forward(ctx_pos, ctx_batch, is_complete)

