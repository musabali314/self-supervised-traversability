import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class WayfastDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_depth = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_mu = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path = '../' + self.data.iloc[idx, 0]
        depth_path = '../' + self.data.iloc[idx, 1]
        mu_path = '../' + self.data.iloc[idx, 3]

        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        mu = Image.open(mu_path).convert('L')

        return {
            'rgb': self.transform_rgb(rgb),
            'depth': self.transform_depth(depth),
            'mu': self.transform_mu(mu)
        }
