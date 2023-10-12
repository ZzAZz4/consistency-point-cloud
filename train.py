
import torch
from torch_geometric.transforms import Compose, FixedPoints, NormalizeScale
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

from models.diffusion import Model
from common.visualization import visualize_batch_results
from common.logs import get_logger, get_new_log_dir
from loss.chamfer import CDLoss

import os


def get_data_iterator(iterable):
    while True:
        for element in iterable:
            yield element
            


log_dir = get_new_log_dir(prefix='diffusion_')
train_logger = get_logger('train', log_dir=log_dir)
val_logger = get_logger('val', log_dir=log_dir)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('logs/diffusion_2023_10_12__03_36_44/checkpoints/checkpoint_6000.pt')
path = "data/ShapeNet"
category = 'Airplane' 

transform = Compose([
    FixedPoints(1024),
    # RandomRotate(15, axis=0),
    # RandomRotate(15, axis=1),
    # RandomRotate(15, axis=2),
])
pre_transform = NormalizeScale()
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