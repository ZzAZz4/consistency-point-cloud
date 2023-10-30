
from emd.emd_module import emdModule as EMDModule

import torch
from torch import Tensor
from torch_geometric.utils import to_dense_batch
import warnings
from typing import Literal, overload


def from_dense_batch(dense_bath, mask):
    # dense batch, B, N, F
    # mask, B, N
    B, N, F = dense_bath.size()
    flatten_dense_batch = dense_bath.view(-1, F)
    flatten_mask = mask.view(-1)
    data_x = flatten_dense_batch[flatten_mask, :]
    num_nodes = torch.sum(mask, dim=1)  # B, like 3,4,3
    pr_value = torch.cumsum(num_nodes, dim=0)  # B, like 3,7,10
    indicator_vector = torch.zeros(torch.sum(num_nodes, dim=0)) # type: ignore
    indicator_vector[pr_value[:-1]] = 1  # num_of_nodes, 0,0,0,1,0,0,0,1,0,0,1
    data_batch = torch.cumsum(indicator_vector, dim=0)  # num_of_nodes, 0,0,0,1,1,1,1,1,2,2,2
    return data_x, data_batch


class EMDLoss(torch.nn.Module):
    def __init__(self, eps=0.005, iters=50):
        super().__init__()
        self._eps = eps
        self._iters = iters
        self._emd = EMDModule()

    def forward(self, pred: Tensor, target: Tensor, batch: Tensor, with_assignment=False):
        pred, target = to_dense_batch(pred, batch)[0], to_dense_batch(target, batch)[0]

        if not pred.is_cuda:
            warnings.warn('Input is not in GPU. Results will be moved to GPU temporarily.')
        elif not target.is_cuda:
            warnings.warn('Input is not in GPU. Results will be moved to GPU temporarily.')

        # EMDModule expects the PC's to be between 0 and 1
        shared = torch.cat([pred, target], dim=1)
        offset = torch.min(shared, dim=1)[0][:, None, :]
        scale = 0.99999 / torch.max(shared - offset)
        
        pred, target = (pred - offset) * scale, (target - offset) * scale
        dis, assignment = self._emd(pred, target, self._eps, self._iters)

        reduced = (torch.sqrt(dis).mean() / scale).to(pred.device)
        if with_assignment:
            assignment: Tensor
            assignment = assignment.flatten() + batch * assignment.size(1)
            return reduced, assignment.to(pred.device)
        return reduced
    
    

if __name__ == '__main__':
    def main():
        sample = torch.randn(4 * 1024, 3).cuda()
        target = torch.randn(4 * 1024, 3).cuda()
        batch = torch.repeat_interleave(torch.arange(4), 1024).cuda()

        loss = EMDLoss(eps=0.002, iters=10000)
        dis, assignment = loss(sample, target, batch, with_assignment=True) 
        
        print(dis)
        reordered_target = target[assignment]
        print((sample - reordered_target).pow(2).sum(dim=1).sqrt().mean())


    main()