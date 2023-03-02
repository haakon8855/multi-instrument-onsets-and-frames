import torch
from torchaudio.transforms import MelSpectrogram
from math import exp, log

import constants


class Preprocessor:
    def __init__(self, device) -> None:
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
        ).to(device)

    def mel(self, wav: torch.tensor) -> torch.tensor:
        mel_output = self.melspectrogram(wav.reshape(-1, wav.shape[-1])[:, :-1]).transpose(-1, -2)
        mel_output = torch.log(torch.clamp(mel_output, min=self._mel_clamp_value))
        return mel_output
