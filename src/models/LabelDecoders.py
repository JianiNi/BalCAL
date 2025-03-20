import torch.nn as nn
import torch
import torch.nn.functional as F

class CIFAR10SimpelLabelDecoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.fc1_y = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        y = self.fc1_y(z)
        return y