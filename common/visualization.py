import math
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_batch


def visualize_points(pos, c=None):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c='blue' if c is None else c, s=3)
    plt.show()


def visualize_batched_points(pos, batch, index, c=None):
    tg, _ = to_dense_batch(pos, batch)
    tg = tg[index]
    visualize_points(tg.cpu())


def visualize_batch_results(pos, newpos, batch, max_in_row=5):
    pos, _ = to_dense_batch(pos, batch)
    newpos, _ = to_dense_batch(newpos, batch)
    n = pos.size(0)

    num_rows = math.ceil(n / max_in_row)
    num_cols = min(n, max_in_row)

    fig = plt.figure(figsize=(num_cols * 4, num_rows * 4))
    for i in range(n):
        ax = fig.add_subplot(2 * num_rows, num_cols, i+1, projection='3d')
        pc = pos[i].cpu()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='red', s=3)
        ax.set_aspect('equal', adjustable='datalim')

    for i in range(n):
        ax = fig.add_subplot(2 * num_rows, num_cols, i +
                             num_cols * num_rows + 1, projection='3d')
        pc = newpos[i].cpu()
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c='blue', s=3)
        ax.set_aspect('equal', adjustable='datalim')

    return fig
