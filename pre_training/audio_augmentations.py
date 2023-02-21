import os
import glob
import torch
import torchvision
import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np


class Preprocessor:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.file_paths = os.listdir(data_path)
        self.file_paths = glob.glob(data_path + "*.flac")
        self.last_index = -1

    def get_next_n_spectrograms(self, num: int = 1):
        spectrograms = []
        first_index = self.last_index + 1
        files = self.file_paths[first_index : first_index + num]
        for filename in files:
            full_path = filename
            spectrograms.append(Preprocessor.get_mel_spectrogram(full_path))
        self.last_index += num
        return spectrograms

    @staticmethod
    def get_mel_spectrogram(
        audio_file_path: str,
        sample_rate: int = 16000,
        hop_length: int = 512,
        win_length: int = 2048,
        n_fft: int = 2048,
        fmin: int = 30,
        fmax: int = 8000,
        n_mels: int = 229,
    ) -> np.ndarray:
        x, sr = librosa.load(audio_file_path, sr=sample_rate)
        mel = librosa.feature.melspectrogram(
            y=x, sr=sr, hop_length=hop_length, win_length=win_length, n_fft=n_fft, fmin=fmin, fmax=fmax, n_mels=n_mels
        )
        return librosa.power_to_db(abs(mel))

    @staticmethod
    def display_spectrogram(spectrogram):
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(np.squeeze(spectrogram), sr=16000, fmax=8000, fmin=30, x_axis="time", y_axis="mel")
        plt.colorbar()
        plt.show()


def augmentation_gaussian_blur(spectrogram, kernel=3, sigma=(1.0, 2.0)):
    """
    Applies a Gaussian blur transformation to the spectrogram using the given
    kernel size and range for a randomly selected standard deviation.
    """
    augmentation = torchvision.transforms.GaussianBlur(kernel, sigma)
    spectrogram = augmentation(spectrogram)
    return spectrogram


def augmentation_pitch_shift(spectrogram, shift_n_bins=12):
    """
    Applies a pitch shift to the given spectrogram, moving the notes up or down
    in pitch equal to the number of rows given in the shift_n_bins argument.
    """
    spectrogram = torchvision.transforms.functional.affine(
        img=spectrogram, angle=0, translate=(0, shift_n_bins), scale=1, shear=0, fill=spectrogram.min()
    )
    return spectrogram


def augmentation_random_pitch_shift(spectrogram, shift_n_bins_range=(-12, 12)):
    """
    Applies a pitch shift to the given spectrogram, moving the notes up or down
    in pitch a random value sampled from the given range.
    """
    possible_shifts = np.arange(shift_n_bins_range[0], shift_n_bins_range[1])
    possible_shifts = possible_shifts[possible_shifts != 0]
    shift = np.random.choice(possible_shifts, 1)
    return augmentation_pitch_shift(spectrogram, shift_n_bins=shift)


def augmentation_random_crop_and_stretch(spectrogram, cols_to_crop_range=(20, 50)):
    """
    Crops the image in the horizontal axis (removes columns on each side) and
    stretches the image back to its original size.
    """
    crop_cols = np.random.randint(cols_to_crop_range[0], cols_to_crop_range[1])
    orig_shape = spectrogram.shape
    spectrogram = spectrogram[:, :, :, crop_cols:-crop_cols]
    resizer = torchvision.transforms.Resize(orig_shape[2:])
    return resizer(spectrogram)


def main():
    """
    Main function for running this python script.
    """
    prep = Preprocessor("./data/slakh2100_flac_16k/test/Track01877/")
    spects = prep.get_next_n_spectrograms(num=1)
    mel = spects[0]
    min_value = np.min(mel)
    input_length_msecs = 229
    mel = np.pad(mel, ((0, 0), (input_length_msecs // 2, 0)), "constant", constant_values=min_value)
    inputs = []
    for i in range(mel.shape[1] - input_length_msecs):
        inputs.append(mel[:, i : i + input_length_msecs])

    spec1 = torch.tensor(inputs[200]).reshape((1, 1, 229, 229))
    spec2 = torch.tensor(inputs[200]).reshape((1, 1, 229, 229))

    spec2 = augmentation_random_crop_and_stretch(spec2)

    spec1 = np.array(spec1).reshape((229, 229))
    spec2 = np.array(spec2).reshape((229, 229))

    fig, (axs1, axs2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    librosa.display.specshow(spec1, sr=16000, fmax=8000, fmin=30, x_axis="time", y_axis="mel", cmap="inferno", ax=axs1)
    librosa.display.specshow(spec2, sr=16000, fmax=8000, fmin=30, x_axis="time", y_axis="mel", cmap="inferno", ax=axs2)
    # plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
