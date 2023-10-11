import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn

from models.consistency import PointConsistencyModel, PointConsistencySettings, PointConsistencyTraining
from models.model import Model

from pytorch3d.loss import chamfer_distance
from torch_geometric.utils import to_dense_batch
from torch import nn

class CDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, batch):
        pred, target = to_dense_batch(pred, batch)[0], to_dense_batch(target, batch)[0]
        return chamfer_distance(pred, target)[0]


path = "data/ShapeNet"
category = 'Airplane' 
transform = T.Compose([
    T.NormalizeRotation(),
    T.FixedPoints(1024),
])
test_transform = T.Compose([
    T.NormalizeRotation(),
    T.FixedPoints(1024),
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='trainval', transform=transform, pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test', transform=test_transform, pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)


settings = PointConsistencySettings(training_iterations=1_000_000, sigma_range=(0.02, 8))
ct = PointConsistencyTraining(settings)

model = PointConsistencyModel(Model(), sigma_data=0.5, sigma_min=settings.sigma_range[0])
ema_model = PointConsistencyModel(Model(), sigma_data=0.5, sigma_min=settings.sigma_range[0])
ema_model.load_state_dict(model.state_dict())
loss = CDLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, ema_model = model.to(device), ema_model.to(device)

k = 1
for epoch in range(2, settings.training_iterations // len(train_loader) + 1):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        next_x, cur_x = ct.train_step(k, data.pos, data.batch, model, ema_model)

        loss_val = loss(next_x, cur_x, data.batch)
        loss_val.backward()

        optimizer.step()
        ema_model = ct.after_train_step(k, model, ema_model)

        print(f"Epoch: {epoch}, Iteration: {k}, Loss: {loss_val.item()}")        
        k += 1
