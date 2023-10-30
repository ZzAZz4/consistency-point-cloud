import typing

if typing.TYPE_CHECKING:
    import torch


def forward(
    xyz1: torch.Tensor, 
    xyz2: torch.Tensor, 
    dist: torch.Tensor, 
    assignment: torch.Tensor, 
    price: torch.Tensor, 
    assignment_inv: torch.Tensor,
    bid: torch.Tensor, 
    bid_increments: torch.Tensor, 
    max_increments: torch.Tensor, 
    unass_idx: torch.Tensor, 
    unass_cnt: torch.Tensor, 
    unass_cnt_sum: torch.Tensor, 
    cnt_tmp: torch.Tensor, 
    max_idx: torch.Tensor, 
    eps: float, 
    iters: int
): ...



def backward(
    xyz1: torch.Tensor, 
    xyz2: torch.Tensor, 
    gradxyz1: torch.Tensor, 
    graddist: torch.Tensor, 
    assignment: torch.Tensor
): ...