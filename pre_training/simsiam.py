"""haakoas, matsjno"""

from torch import nn

from encoder import Encoder


class SimSiam(nn.Module):
    """
    Class contaning the Simple Siamese network made up of an encoder
    and a predictor.
    """

    def __init__(self, device):
        super().__init__()
        self.output_size = 28 * 28
        self.predictor_hidden_size = 512
        self.encoder = Encoder().to(device)
        self.predictor = nn.Sequential(
            # Two fully connected layers creating a bottleneck
            nn.Linear(self.output_size, self.predictor_hidden_size),
            nn.BatchNorm1d(self.predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(self.predictor_hidden_size, self.output_size),
        ).to(device)

    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detatch(), z2.detatch()

    def save(self, path: str):
        self.encoder.save(path)
