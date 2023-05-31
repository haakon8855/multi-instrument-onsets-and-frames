"""haakoas, matsjno"""

import torch
from torch import nn

from encoder import Encoder
from encoder import summary


class SimSiam(nn.Module):
    """
    Class contaning the Simple Siamese network made up of an encoder
    and a predictor.
    """

    def __init__(self, device):
        super().__init__()
        self.output_size = 640 * 229
        self.predictor_hidden_size = 512
        self.encoder = Encoder().to(device)
        self.predictor = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 768)).to(
            device
        )

    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

    def save(self, path: str):
        self.encoder.save(path)


def main():
    """
    Main function for running this python script.
    """
    device = torch.device("cuda")
    simsiam = SimSiam(device)
    summary(simsiam)
    output = simsiam(torch.rand(8, 640, 229).to(device), torch.rand(8, 640, 229).to(device))
    print(output[0].shape, output[2].shape)


if __name__ == "__main__":
    main()
