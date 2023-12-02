import math
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_batch
from mpl_toolkits.mplot3d.axes3d import Axes3D


def visualize_points(pos, alpha=0.2):
    fig = plt.figure(figsize=(4, 4))
    ax: Axes3D = fig.add_subplot(projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=1, alpha=alpha)
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()


def visualize_batch_points(pos, batch, max_in_row=5, alpha=0.2):
    pos = pos.detach().cpu()
    batch = batch.detach().cpu()
    pos, _ = to_dense_batch(pos, batch)
    n = pos.size(0)

    num_rows = math.ceil(n / max_in_row)
    num_cols = min(n, max_in_row)

    fig = plt.figure(figsize=(num_cols * 4, num_rows * 4))
    for i in range(n):
        ax = fig.add_subplot(2 * num_rows, num_cols, i+1, projection='3d')
        ax.scatter(pos[i][:, 0], pos[i][:, 1], pos[i][:, 2], c='red', s=1, alpha=alpha) # type: ignore
        ax.set_aspect('equal', adjustable='datalim')

    return fig
