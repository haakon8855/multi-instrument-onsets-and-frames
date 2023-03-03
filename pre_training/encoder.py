"""haakoas, matsjno"""

import torch
from torch import nn
import torchvision


class Encoder(nn.Module):
    """
    Class contaning an encoder for the Simple Siamese network.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.resnet = torchvision.models.resnet34()
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 3), stride=2, padding=(0, 1)),
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 3), stride=2, padding=(0, 1)),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=2, padding=(0, 0)),
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 3), stride=2, padding=(0, 1)),
            nn.ConvTranspose2d(32, 1, kernel_size=(2, 3), stride=2, padding=(0, 1)),
        )

    def forward(self, x):
        x = x[:, None, :, :]  # Add empty dimension to act as number of channels
        x = self.network(x)  # Expand number of channels to 3
        x = self.resnet(x)
        x = x.reshape((-1, 512, 20, 8))
        x = self.upscaler(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)
