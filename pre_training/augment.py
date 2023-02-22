import torch
import torchvision
import numpy as np


class Augment:
    @staticmethod
    def random_erase(spectogram):
        """
        Randomly selects a rectangular area of the given size range and fills
        that area with the tensor's overall minimum value.
        """
        eraser = torchvision.transforms.RandomErasing(p=1, scale=(0.1, 0.2), value=int(spectogram.min()))
        return eraser(spectogram)

    @staticmethod
    def noise_injection(spectogram, std=1.5, mean=0):
        """
        Adds Gaussian noise to the spectrogram with the given standard deviation
        and mean.
        """
        return spectogram + torch.randn(spectogram.size()) * std + mean

    @staticmethod
    def gaussian_blur(spectrogram, kernel=3, sigma=(1.0, 2.0)):
        """
        Applies a Gaussian blur transformation to the spectrogram using the given
        kernel size and range for a randomly selected standard deviation.
        """
        augmentation = torchvision.transforms.GaussianBlur(kernel, sigma)
        spectrogram = augmentation(spectrogram)
        return spectrogram

    @staticmethod
    def pitch_shift(spectrogram, shift_n_bins=12):
        """
        Applies a pitch shift to the given spectrogram, moving the notes up or down
        in pitch equal to the number of rows given in the shift_n_bins argument.
        """
        spectrogram = torchvision.transforms.functional.affine(
            img=spectrogram, angle=0, translate=(0, shift_n_bins), scale=1, shear=0, fill=spectrogram.min()
        )
        return spectrogram

    @staticmethod
    def random_pitch_shift(spectrogram, shift_n_bins_range=(-12, 12)):
        """
        Applies a pitch shift to the given spectrogram, moving the notes up or down
        in pitch a random value sampled from the given range.
        """
        possible_shifts = np.arange(shift_n_bins_range[0], shift_n_bins_range[1])
        possible_shifts = possible_shifts[possible_shifts != 0]
        shift = np.random.choice(possible_shifts, 1)
        return Augment.pitch_shift(spectrogram, shift_n_bins=shift)

    @staticmethod
    def random_crop_and_stretch(spectrogram, cols_to_crop_range=(20, 50)):
        """
        Crops the image in the horizontal axis (removes columns on each side) and
        stretches the image back to its original size.
        """
        crop_cols = np.random.randint(cols_to_crop_range[0], cols_to_crop_range[1])
        orig_shape = spectrogram.shape
        spectrogram = spectrogram[:, :, :, crop_cols:-crop_cols]
        resizer = torchvision.transforms.Resize(orig_shape[2:])
        return resizer(spectrogram)
