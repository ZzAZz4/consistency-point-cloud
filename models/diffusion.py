import torch

import numpy as np

from models.backbone.pointnet import PointNetEncoder
from models.backbone.glu import GLUDecoder


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
        self.decoder = GLUDecoder(zdim)
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
    

