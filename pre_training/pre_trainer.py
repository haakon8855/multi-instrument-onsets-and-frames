"""haakoas, matsjno"""

import torch
import numpy as np
from augment import Augment

# from siamese_network import SiameseNetwork
from simsiam import SimSiam


class PreTrainer:
    """
    Pre-Training class for running a SimSiam pre-training procedure.
    """

    def __init__(self, device):
        self.simsiam = SimSiam(device)

    def train(self, training_loader, epochs=10):
        optimizer = torch.optim.SGD(
            self.simsiam.parameters(),
            lr=0.025,
            weight_decay=0.0001,
            momentum=0.9,
        )
        report_interval = 10
        for i in range(epochs):
            self.simsiam.train()
            running_loss = 0
            last_loss = 0
            for j, (data, _) in enumerate(training_loader):
                optimizer.zero_grad()
                x1, augmentation = PreTrainer.rand_augment(data)
                x2, _ = PreTrainer.rand_augment(data, avoid_augmentation=augmentation)

                p1, p2, z1, z2 = self.simsiam(x1, x2)

                loss = -(PreTrainer.cos_sim(p1, z2).mean() + PreTrainer.cos_sim(p2, z1).mean()) * 0.5

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (j + 1) % report_interval == 0:
                    last_loss = running_loss / report_interval
                    print(f"Batch {j+1} Loss: {last_loss}")
                    running_loss = 0

    def save(self, path: str):
        self.simsiam.save(path)

    @staticmethod
    def cos_sim(p, z):
        return torch.nn.CosineSimilarity(dim=1)(p, z)

    @staticmethod
    def rand_augment(data, avoid_augmentation=None):
        augmentations = [
            Augment.random_erase,
            Augment.noise_injection,
            Augment.gaussian_blur,
            Augment.random_pitch_shift,
            Augment.random_crop_and_stretch,
        ]
        if avoid_augmentation is not None:
            augmentations.remove(avoid_augmentation)
        aug_idx = np.random.randint(0, len(augmentations))
        augmentation = augmentations[aug_idx]
        return augmentation(data), augmentation


def main():
    """
    Main function for running this python script.
    """
    device = torch.device("cuda:0")

    cuda_avail = torch.cuda.is_available()
    print(f"Cuda: {torch.cuda.is_available()}")
    if cuda_avail:
        print(f"Device: {torch.cuda.get_device_name(0)}")

    pre_trainer = PreTrainer(device)
    pre_trainer.train(None, epochs=1)
    pre_trainer.save("/pre_training/sim_siam_encoder.pt")


if __name__ == "__main__":
    main()
