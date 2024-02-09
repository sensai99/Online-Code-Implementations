import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from scheduler import Scheduler

'''
    Visualise the forward diffusion process for a batch of images - Plots a 5x5 grid of images for each time step
'''

scheduler = Scheduler()
Ts = torch.Tensor([49, 99, 199, 499, 999]).to(torch.int32)

transform = Compose([Resize(size = (32, 32)), ToTensor()])
train_dataset = MNIST(root = '../data', train = True, download = True, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 5, shuffle = True)
train_first_batch = next(iter(train_loader))
data, target = train_first_batch

batch_size = data.shape[0]
fig, axes = plt.subplots(batch_size, len(Ts), figsize=(len(Ts) * 2, batch_size * 1))
fig.suptitle('Forward Diffusion Process for Batch')

for image_idx in range(batch_size):
    current_image = data[image_idx]
    for t_idx, t in enumerate(Ts):
        x_t, _ = scheduler.apply_forward_diffusion(current_image.unsqueeze(0), torch.tensor([t]))
        ax = axes[image_idx, t_idx]
        ax.imshow(x_t.squeeze())
        ax.set_title(f'Time Step {t.item() + 1}')
        ax.axis('off')

plt.tight_layout()
plt.show()
