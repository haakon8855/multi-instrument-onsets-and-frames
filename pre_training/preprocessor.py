import os
import glob
import torch
import soundfile
from torchaudio.transforms import MelSpectrogram
from math import exp, log

import matplotlib.pyplot as plt
import numpy as np

from augment import Augment
import constants


class Preprocessor:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.file_paths = os.listdir(data_path)
        self.file_paths = glob.glob(data_path + "*.flac")
        self.last_index = -1
        self.window_length = constants.WINDOW_LENGTH
        self._mel_clamp_value = exp(-log(self.window_length))
        self.melspectrogram = MelSpectrogram(
            sample_rate=constants.SAMPLE_RATE,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=constants.HOP_LENGTH,
            power=1.0,
            f_min=constants.MEL_FMIN,
            f_max=constants.MEL_FMAX,
            n_mels=constants.N_MELS,
        )

    def mel(self, wav: torch.tensor) -> torch.tensor:
        mel_output = self.melspectrogram(wav.reshape(-1, wav.shape[-1])[:, :-1]).transpose(-1, -2)
        mel_output = torch.log(torch.clamp(mel_output, min=self._mel_clamp_value))
        return mel_output

    def get_next_n_spectrograms(self, num: int = 1):
        spectrograms = []
        first_index = self.last_index + 1
        files = self.file_paths[first_index : first_index + num]
        for filename in files:
            full_path = filename
            audio = torch.tensor(soundfile.read(full_path, start=0, dtype="float32", frames=-1)[0])  # .to(self.device)
            spectrograms.append(self.mel(audio))
        self.last_index += num
        return spectrograms


def main():
    """
    Main function for running this python script.
    """
    prep = Preprocessor("./data/slakh2100_flac_16k/test/Track01877/")
    spects = prep.get_next_n_spectrograms(num=1)
    input_length_msecs = 229
    start = 400
    spec1 = spects[0][:, start : start + input_length_msecs]
    spec2 = spec1

    spec1 = torch.tensor(spec1).reshape((1, 1, 229, input_length_msecs))
    spec2 = torch.tensor(spec2).reshape((1, 1, 229, input_length_msecs))

    spec2 = Augment.gaussian_blur(spec2)

    spec1 = np.array(spec1).reshape((229, input_length_msecs))
    spec2 = np.array(spec2).reshape((229, input_length_msecs))

    fig, (axs1, axs2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    axs1.imshow(spec1, cmap="inferno")
    axs2.imshow(spec2, cmap="inferno")
    plt.show()


if __name__ == "__main__":
    main()
