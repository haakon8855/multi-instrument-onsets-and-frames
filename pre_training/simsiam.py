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
        self.output_size = 640 * 229
        self.predictor_hidden_size = 512
        self.encoder = Encoder().to(device)
        self.predictor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=(5, 6), stride=2, padding=(0, 0)),
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=2, padding=(0, 0)),
            nn.ConvTranspose2d(64, 32, kernel_size=(6, 5), stride=2, padding=(0, 0)),
            nn.ConvTranspose2d(32, 1, kernel_size=(6, 5), stride=2, padding=(0, 0)),
        ).to(device)

    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

    def save(self, path: str):
        self.encoder.save(path)
