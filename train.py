
import torch
from torch_geometric.transforms import Compose, FixedPoints, NormalizeScale, RandomRotate
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models.diffusion import Model
from common.visualization import visualize_batch_results
from common.logs import get_logger, get_new_log_dir
from loss.chamfer_loss import CDLoss
from torch.nn import MSELoss
from argparse import ArgumentParser

import os
import time


def get_data_iterator(iterable):
    while True:
        for element in iterable:
            yield element

parser = ArgumentParser()
parser.add_argument('--logdir', type=str, default='logs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--zdim', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--validate_every', type=int, default=1000)

args = parser.parse_args()


writer = SummaryWriter()
path = "data/ShapeNet"
category = 'Airplane'

transform = Compose([
    FixedPoints(1024),
    # RandomRotate(15, axis=0),
    # RandomRotate(15, axis=1),
    # RandomRotate(15, axis=2),
])
pre_transform = NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval',
                         transform=transform, pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test',
                        transform=transform, pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(zdim=args.zdim, num_steps=args.num_steps, beta_1=args.beta_1, beta_T=args.beta_T).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# checkpoint = torch.load('checkpoints/2023_10_13__03_44_10/ckpt_66000.pt')
checkpoint = args.checkpoint
if checkpoint is not None:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

criterion = MSELoss()  # Define loss criterion.
cd = CDLoss()

time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
os.makedirs('checkpoints', exist_ok=True)
checkpoint_dir = 'checkpoints/{}'.format(time_str)
os.makedirs(checkpoint_dir, exist_ok=True)

logdir = get_new_log_dir(root=args.logdir, prefix='diffusion_')
logger = get_logger('train', log_dir=logdir)


@torch.no_grad()
def validate_loss():
    model.eval()
    all_loss = 0
    for i, data in enumerate(test_loader):
        data = data.to(device)

        z = model.encode(data.pos, data.batch)
        recons = model.decode(data.pos.size(), z, data.batch)
        loss = cd(recons, data.pos, data.batch)
        logger.info(f'Validation Iteration: {i}, Loss: {loss.item()}')

        all_loss += loss.item()

    error = all_loss / len(test_loader) # type: ignore
    return error


@torch.no_grad()
def create_sample_figure(dataset, num_samples):
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data = next(iter(loader))
    data = data.to(device)

    z = model.encode(data.pos, data.batch)
    recons = model.decode(data.pos.size(), z, data.batch)
    fig = visualize_batch_results(recons, data.batch, max_in_row=num_samples)

    return fig

def train():
    validation_interval = args.validate_every
    for i, data in enumerate(get_data_iterator(train_loader), 1):
        optimizer.zero_grad()  # Clear gradients.
        model.train()
        data = data.to(device)

        e_theta, e_rand = model(data.pos, data.batch)  # Forward pass.
        loss = criterion(e_theta, e_rand)  # Loss computation.

        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.

        logger.info(f'Iteration: {i}, Loss: {loss.item()}')
        writer.add_scalar('Train Loss', loss.item(), i)

        if i % validation_interval == 0:
            logger.info("Creating sample image and validation loss")

            train_fig = create_sample_figure(train_dataset, 10)
            test_fig = create_sample_figure(test_dataset, 10)

            writer.add_figure('Train Results', train_fig, i)
            writer.add_figure('Validation Results', test_fig, i)

            test_cd = validate_loss()

            logger.info(f'Iteration: {i}, Validation Loss: {test_cd}')
            writer.add_scalar('Validation Loss', test_cd, i)

            checkpoint_name = os.path.join(checkpoint_dir, f'ckpt_{i}.pt')
            torch.save(
                {
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'test_cd': test_cd
                },
                checkpoint_name
            )
        

train()
