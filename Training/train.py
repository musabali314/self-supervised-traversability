import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import WayfastDataset
from resnet_depth_unet import ResnetDepthUnet
from params import Params
from tqdm import tqdm

params = Params()

dataset = WayfastDataset(params.train_csv)
loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)

model = ResnetDepthUnet(params).to(params.device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=params.lr)

for epoch in range(params.epochs):
    running_loss = 0.0
    loop = tqdm(loader)
    for batch in loop:
        rgb = batch['rgb'].to(params.device)
        depth = batch['depth'].to(params.device)
        mu_gt = batch['mu'].to(params.device)

        mu_pred = model(rgb, depth)

        loss = criterion(mu_pred, mu_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{params.epochs}]")
        loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
