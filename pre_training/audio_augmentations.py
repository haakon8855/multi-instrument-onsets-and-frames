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


def random_erase(spectogram):
    eraser = torchvision.transforms.RandomErasing(p=1, scale=(0.1, 0.2), value=int(spectogram.min()))
    return eraser(spectogram)


def noise_injection(spectogram, std=1.5, mean=0):
    return spectogram + torch.randn(spectogram.size()) * std + mean


def main():
    """
    Main function for running this python script.
    """
    prep = Preprocessor("./data/slakh2100_flac_16k/test/Track01877/")
    spects = prep.get_next_n_spectrograms(num=1)
    mel = spects[0]
    min_value = np.min(mel)
    input_length_msecs = 229
    # append input_length_msecs//2 cols with zeros to mel
    mel = np.pad(mel, ((0, 0), (input_length_msecs // 2, 0)), "constant", constant_values=min_value)
    inputs = []
    for i in range(mel.shape[1] - input_length_msecs):
        inputs.append(mel[:, i : i + input_length_msecs])

    spec1 = torch.tensor(inputs[200]).reshape((1, 1, 229, 229))
    spec2 = torch.tensor(inputs[200]).reshape((1, 1, 229, 229))

    min_value = np.array(spec2).min()

    # TODO: Apply augmentation
    spec2 = noise_injection(spec2)

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
