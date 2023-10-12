from torch.nn import Linear, SiLU, ModuleList
from torch_geometric.nn import MLP, MaxAggregation
from torch_geometric.nn import PointNetConv, PositionalEncoding
import torch

import torch
from torch_cluster import knn_graph, fps

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

from pytorch3d.loss import chamfer_distance
from torch_geometric.utils import to_dense_batch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import math

import logging
import os
import time

from models.backbone.pointnet import PointNetEncoder
from models.backbone.glu import PointwiseNet
from common.visualization import visualize_batch_results
from loss.chamfer import CDLoss



class VarianceSchedule(torch.nn.Module):
    def __init__(self, num_steps, beta_1, beta_T):
        super().__init__()
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T

        betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alpha_bars: torch.Tensor
        self.sigmas_flex: torch.Tensor
        self.sigmas_inflex: torch.Tensor
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    


class Model(torch.nn.Module):
    def __init__(self, zdim, num_steps, beta_1, beta_T):
        super().__init__()
        self.encoder = PointNetEncoder(zdim)
        self.decoder = PointwiseNet(zdim)
        self.schedule = VarianceSchedule(num_steps, beta_1, beta_T)

    def encode(self, pos: torch.Tensor, batch: torch.Tensor):
        return self.encoder(pos, batch)
    
    def decode(self, shape: tuple, ctx: torch.Tensor, batch: torch.Tensor, flex: float=0.0):
        x_t = torch.randn(shape).to(ctx.device)
        batch_size = int(batch.max() + 1)

        for t in range(self.schedule.num_steps, 0, -1):
            alpha = self.schedule.alphas[t]
            alpha_bar = self.schedule.alpha_bars[t]
            sigma = self.schedule.get_sigmas(t, flex)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            beta = self.schedule.betas[t].repeat(batch_size).view(-1, 1)
            e_theta = self.decoder(x_t, t=beta, ctx=ctx, batch=batch)
            
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            x_t = c0 * (x_t - c1 * e_theta) + sigma * z

        return x_t
    

    def forward(self, pos: torch.Tensor, batch: torch.Tensor):
        z: torch.Tensor = self.encoder(pos, batch)
        batch_size = z.size(0)
 
        t = self.schedule.uniform_sample_t(batch_size)
        alpha_bar = self.schedule.alpha_bars[t]
        beta = self.schedule.betas[t]

        c0 = torch.sqrt(alpha_bar)       
        c1 = torch.sqrt(1 - alpha_bar)   
        c0, c1 = c0[batch].view(-1, 1), c1[batch].view(-1, 1)

        e_rand = torch.randn_like(pos)
        e_theta = self.decoder(c0 * pos + c1 * e_rand, t=beta, ctx=z, batch=batch)

        return e_theta, e_rand
    


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'samples'), exist_ok=True)
    return log_dir



log_dir = get_new_log_dir(prefix='diffusion_')
train_logger = get_logger('train', log_dir=log_dir)
val_logger = get_logger('val', log_dir=log_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('logs/diffusion_2023_10_12__03_36_44/checkpoints/checkpoint_6000.pt')
path = "data/ShapeNet"
category = 'Airplane' 

transform = T.Compose([
    T.FixedPoints(1024),
    # T.RandomRotate(15, axis=0),
    # T.RandomRotate(15, axis=1),
    # T.RandomRotate(15, axis=2),
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval', transform=transform, pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test', transform=transform, pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = Model(zdim=train_dataset.num_classes, num_steps=1000, beta_1=1e-4, beta_T=0.02).to('cuda')
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

criterion = torch.nn.MSELoss()  # Define loss criterion.
cd = CDLoss()

@torch.no_grad()
def validate_loss(it):
    all_loss = 0
    for i, data in enumerate(test_loader):
        model.eval()
        data = data.to(device)
        
        z = model.encode(data.pos, data.batch)
        recons = model.decode(data.pos.size(), z, data.batch)
        loss = cd(recons, data.pos, data.batch, reduction='sum')

        if i == 0:
            fig = visualize_batch_results(data.pos, recons, data.batch)
            path = os.path.join(log_dir, 'samples', 'sample_{}.png'.format(it))
            fig.savefig(path)
            val_logger.info("Validation:: Saved sample image to {}".format(path))
            
        val_logger.info("Validation:: Iteration: {}, Loss: {}".format(i, loss.item() / test_loader.batch_size))
        all_loss += loss.item()

    
    error = all_loss / len(test_loader.dataset) # type: ignore
    val_logger.info("Validation:: Average Loss: {}".format(error)) 
    return error 

    

def train():
    ma_loss = 0
    val_loss = validate_loss(0)
    for i, data in enumerate(get_data_iterator(train_loader), 1):
        optimizer.zero_grad()  # Clear gradients.
        model.train() 
        data = data.to(device)
        
        e_theta, e_rand = model(data.pos, data.batch)  # Forward pass.
        loss = criterion(e_theta, e_rand)  # Loss computation.
        
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        
        ma_loss = 2 / (i + 1) * (loss.item() - ma_loss) + ma_loss
        if i % 1000 == 0:
            val_loss = validate_loss(i)
            
            train_logger.info("Train:: Saving epoch {} with loss {}".format(i, val_loss))
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'model_buffers': model._buffers,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(log_dir, 'checkpoints', 'checkpoint_{}.pt'.format(i)))
        
        train_logger.info("Train:: Iteration: {}, Loss: {}, MA Loss: {}".format(i, loss.item(), ma_loss))


train()