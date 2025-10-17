import torch
from resnet_depth_unet import ResnetDepthUnet
from params import Params
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

params = Params()

model = ResnetDepthUnet(params).to(params.device)
model.load_state_dict(torch.load('model_epoch_50.pth'))
model.eval()

rgb_path = '../color/some_rgb_image.png'
depth_path = '../depth/some_depth_image.png'

rgb = Image.open(rgb_path).convert('RGB')
depth = Image.open(depth_path).convert('L')

transform = transforms.ToTensor()

rgb_tensor = transform(rgb).unsqueeze(0).to(params.device)
depth_tensor = transform(depth).unsqueeze(0).to(params.device)

with torch.no_grad():
    mu_pred = model(rgb_tensor, depth_tensor)

mu_pred = mu_pred.squeeze().cpu().numpy()

plt.imshow(mu_pred, cmap='gray')
plt.show()
