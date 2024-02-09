import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from scheduler import Scheduler
from trainer import Trainer
from model import Model

# Data
transform = Compose([Resize(size = (32, 32)), ToTensor()])
train_dataset = MNIST(root = '../data', train = True, download = True, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)


scheduler = Scheduler()
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(train_loader, scheduler, model, optimizer, criterion, epochs, device)
trainer.train()



# Algorithm
    
    # Training
        #   1. Pick a timestep uniformly at random
        #   2. Apply the forward diffusion process to the data
        #   3. Send the data through the model

    # Inference
        #   1. Given gaussian noise, apply the reverse diffusion process to the noise
        #   2. Send the data through the model

    






