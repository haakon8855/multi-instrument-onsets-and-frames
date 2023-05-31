"""haakoas, matsjno"""

import torch
import math
import numpy as np
from augment import Augment
from torch.utils.data import DataLoader
from time import time

from simsiam import SimSiam
from dataset import MTGJamendo
from preprocessor import Preprocessor


class PreTrainer:
    """
    Pre-Training class for running a SimSiam pre-training procedure.
    """
    def __init__(self, device):
        self.device = device
        self.simsiam = SimSiam(device)
        self.preprocessor = Preprocessor(device)
        self.augmenter = Augment(device)

    def train(self, training_loader, batch_size, epochs=10):
        init_lr = 0.05 * batch_size / 256
        optimizer = torch.optim.SGD(
            self.simsiam.parameters(),
            lr=init_lr,
            weight_decay=0.0001,
            momentum=0.9,
        )
        report_interval = 10
        for i in range(epochs):
            time1 = time()
            self.simsiam.train()
            running_loss = 0
            epoch_loss = 0
            counter = 0
            last_loss = 0
            adjust_learning_rate(optimizer, init_lr, i, epochs)
            for j, data in enumerate(training_loader):
                audio = data.audio.to(self.device)
                audio = self.preprocessor.mel(audio)
                optimizer.zero_grad()
                x1, augmentation = self.rand_augment_distribution(audio)
                x2, _ = self.rand_augment_distribution(
                    audio, avoid_augmentation=augmentation)

                p1, p2, z1, z2 = self.simsiam(x1, x2)

                loss = -(PreTrainer.cos_sim(p1, z2).mean() +
                         PreTrainer.cos_sim(p2, z1).mean()) * 0.5

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                counter += 1
                if (j + 1) % report_interval == 0:
                    last_loss = running_loss / report_interval
                    print(f"Batch {j+1} Loss: {last_loss}")
                    running_loss = 0
            epoch_loss = epoch_loss / counter
            time2 = time()
            print(f"Epoch {i+1} Loss: {epoch_loss}, Time: {time2-time1}")
            if (i + 1) % 5 == 0:
                self.save(
                    f"pre_training/checkpoints/sim_siam_encoder_{i+1}.pt")

    def save(self, path: str):
        self.simsiam.save(path)

    @staticmethod
    def cos_sim(p, z):
        return torch.nn.CosineSimilarity(dim=1)(p, z)

    def rand_augment(self, data, avoid_augmentation=None):
        augmentations = [
            self.augmenter.random_erase,
            self.augmenter.noise_injection,
            self.augmenter.gaussian_blur,
            # self.augmenter.random_pitch_shift,
            self.augmenter.random_crop_and_stretch,
        ]
        if avoid_augmentation is not None:
            augmentations.remove(avoid_augmentation)
        aug_idx = np.random.randint(0, len(augmentations))
        augmentation = augmentations[aug_idx]
        return augmentation(data), augmentation

    def rand_augment_distribution(self, data, avoid_augmentation=None):
        augmentations = [
            self.augmenter.random_crop_and_stretch,
            self.augmenter.random_erase,
            self.augmenter.gaussian_blur,
            self.augmenter.random_pitch_shift,
            self.augmenter.noise_injection,
        ]
        probabilities = np.array([
            [1.0, 1.0],
            [0.2, 0.2],
            [0.8, 0.1],
            [0.0, 0.3],
            [0.3, 0.3],
        ])
        aug_randoms = np.random.rand(1, len(augmentations))
        col = 0
        if avoid_augmentation is not None:
            col = 1
        for i, augmentation in enumerate(augmentations):
            if probabilities[i, col] > aug_randoms[0, i]:
                data = augmentation(data)
        return data, 1


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def main():
    """
    Main function for running this python script.
    """
    device = torch.device("cuda")

    cuda_avail = torch.cuda.is_available()
    print(f"Cuda: {torch.cuda.is_available()}")
    if cuda_avail:
        print(f"Device: {torch.cuda.get_device_name(0)}")

    batch_size = 40
    dataset = MTGJamendo(path="data/mtg-jamendo/", device=device)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=10)

    pre_trainer = PreTrainer(device)
    pre_trainer.train(loader, batch_size, epochs=100)


if __name__ == "__main__":
    main()
