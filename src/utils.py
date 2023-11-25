from datetime import datetime
import os
import math
import matplotlib.pyplot as plt
from src.models.consistency.model import Consistency

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib.animation import FuncAnimation, FFMpegWriter
from torch_geometric.utils import to_dense_batch
import torch

from src.data import BatchedData

from IPython.display import display
from IPython.core.display import HTML


class LogLoss:
    def __init__(self, writer: SummaryWriter):
        self.writer = writer

    def __call__(self, iteration, loss):
        if isinstance(loss, dict):
            for k, v in loss.items():
                self.writer.add_scalar(k, v, iteration)
        else:
            self.writer.add_scalar("Loss", loss, iteration)


class PrintAmortizedLoss:
    def __init__(self, n):
        self.cum = dict()
        self.n = n

    def __call__(self, iteration, loss):
        if isinstance(loss, dict):
            for k, v in loss.items():
                if k not in self.cum:
                    self.cum[k] = 0
                self.cum[k] += v
        else:
            if "loss" not in self.cum:
                self.cum["loss"] = 0
            self.cum["loss"] += loss

        if iteration % self.n == 0:
            print(f"Iteration {iteration}: ", end="")
            for k, v in self.cum.items():
                print(f"{k}: {v / self.n} ", end="")
            print()
            self.cum = dict()
        # self.cum += loss
        # if iteration % self.n == 0:
        #     print(f"Iteration {iteration}: Loss: {self.cum / 10}")
        #     self.cum = 0


class SaveCheckpoint:
    def __init__(self, model, optimizer, ckpt_dir, every=1):
        self.model = model
        self.optimizer = optimizer
        self.ckpt_dir = ckpt_dir
        self.every = every

    def __call__(self, epoch):
        if not epoch or epoch % self.every != 0:
            return
        
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt"))



def make_visualization_frame(n, max_in_row=5):
    num_rows = math.ceil(n / max_in_row)
    num_cols = min(n, max_in_row)
    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))
    axs, graphs = [], []
    
    for i in range(n):
        ax = fig.add_subplot(2 * num_rows, num_cols, i+1, projection='3d')
        graph, = ax.plot([-1, 1], [-1, 1], [-1, 1], marker='o', linestyle='None', c='red', markersize=1, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        graphs.append(graph); axs.append(ax)

    low_axs, low_graphs = [], []
    for i in range(n):
        ax = fig.add_subplot(2 * num_rows, num_cols, n + i + 1, projection='3d')
        graph, = ax.plot([-1, 1], [-1, 1], [-1, 1], marker='o', linestyle='None', c='blue', markersize=1, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')
        low_graphs.append(graph); low_axs.append(ax)

    return fig, axs, graphs, low_axs, low_graphs


class Visualize:
    def __init__(self, vid_dir, model: Consistency, dataloader, device, every=1):
        self.vid_dir = vid_dir
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.every = every

    def __call__(self, epoch):
        if epoch % self.every != 0:
            return
            
        self.model.eval()
        data = next(iter(self.dataloader))
        data: BatchedData = data.to(self.device) # type: ignore

        noise = torch.randn_like(data.pos, device=self.device)
        batch_size = int(data.pos_batch.max()) + 1

        frame_source, axs_source, graphs_source, axs_result, graphs_result = make_visualization_frame(batch_size, max_in_row=batch_size)

        def animation(t):
            t = t + 1e-3
            dirty = data.pos + t * noise

            t = torch.ones_like(data.pos_batch) * t
            out = self.model(x=dirty, t=t, x_batch=data.pos_batch, par=data.par, par_batch=data.par_batch)

            dirty = dirty / dirty.std(dim=0, keepdim=True) * out.std(dim=0, keepdim=True)

            source_out = dirty.detach().cpu()
            result_out = out.detach().cpu()
            batch = data.batch.detach().cpu()
            source_pos, _ = to_dense_batch(source_out, batch)
            result_pos, _ = to_dense_batch(result_out, batch)

            for i in range(batch_size):
                x, y, z = source_pos[i][:, 0], source_pos[i][:, 1], source_pos[i][:, 2]
                graphs_source[i].set_data(x, y)
                graphs_source[i].set_3d_properties(z)
                # title
                axs_source[i].set_title(f"t = {t[i].item():.2f}")
                axs_source[i].set_aspect('equal', adjustable='datalim')

                x, y, z = result_pos[i][:, 0], result_pos[i][:, 1], result_pos[i][:, 2]
                graphs_result[i].set_data(x, y)
                graphs_result[i].set_3d_properties(z)
                axs_result[i].set_aspect('equal', adjustable='datalim')

            return graphs_source + graphs_result
            

        frames = range(0, 70)
        anim = FuncAnimation(frame_source, animation, frames=frames, interval=200, blit=True)
        # writer = FFMpegWriter(fps=10)

        # if not os.path.exists(self.vid_dir):
        #     os.makedirs(self.vid_dir)

        # anim.save(os.path.join(self.vid_dir, f"epoch_{epoch}.mp4"), writer=writer)

        display(HTML(anim.to_jshtml()))