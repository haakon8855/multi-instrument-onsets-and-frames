import hashlib
import os
from abc import abstractmethod
from glob import glob
from typing import List, NamedTuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from constants import (
    DEFAULT_DEVICE,
    HOP_LENGTH,
    SAMPLE_RATE,
)

# torchaudio.set_audio_backend("soundfile")
torchaudio.set_audio_backend("sox_io")


class Audio(NamedTuple):
    """Audio and label class with data that will be on GPU"""

    audio_path: str
    audio: torch.FloatTensor  # [num_steps, n_mels]
    start_time: float
    end_time: float


def load_audio(paths: List[str], frame_offset: int = 0, num_frames: int = -1, normalize: bool = False) -> torch.Tensor:
    audio = torchaudio.load(paths, frame_offset=frame_offset, num_frames=num_frames, normalize=normalize, format="mp3")[
        0
    ]
    if audio.dtype == torch.float32 and len(audio.shape) == 2 and audio.shape[0] == 1:
        audio.squeeze_()
    else:
        raise RuntimeError(f"Unsupported tensor shape f{audio.shape} of type f{audio.dtype}")
    # TODO: Check that all files in MTG-Jamendo are at 44100 kHz sample rate
    resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
    audio = resampler(audio)
    return audio


class UnlabbeledAudioDataset(Dataset):
    def __init__(
        self,
        path,
        groups=None,
        sequence_length=None,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
        max_harmony=None,
    ):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.sequence_length = sequence_length
        self.device = device
        self.random = np.random.RandomState(seed)
        self.num_files = num_files
        self.max_harmony = max_harmony
        self.reproducable_load_sequences = reproducable_load_sequences

        self.file_list = []
        for group in groups:
            for file in self.files(group):
                if num_files is not None and len(self.file_list) >= num_files:
                    break
                self.file_list.append(file)
        self.file_list.sort(key=lambda x: len(x[1]), reverse=True)
        self.labels = [None] * len(self.file_list)

        self.max_files_in_memory = len(self.file_list) if max_files_in_memory < 0 else max_files_in_memory
        if self.max_files_in_memory > 0:
            self.audios = [None] * min(len(self.file_list), self.max_files_in_memory)

    def __getitem__(self, index) -> torch.Tensor:
        audio_path = self.file_list[index]
        audio = None
        if index < self.max_files_in_memory:
            audio = self.audios[index]

        if audio is not None:
            audio_length = audio.shape[0]
        else:
            audio_length = (torchaudio.info(audio_path).num_frames // 44100) * 16000
        start_frame = None
        end_frame = None
        if self.sequence_length is not None:
            possible_start_interval = audio_length - self.sequence_length
            if self.reproducable_load_sequences:
                step_begin = (
                    int(hashlib.sha256("".join(audio_path).encode("utf-8")).hexdigest(), 16) % possible_start_interval
                )
            else:
                step_begin = self.random.randint(possible_start_interval)
            step_begin //= HOP_LENGTH

            begin = step_begin * HOP_LENGTH
            end = begin + self.sequence_length
            num_frames = end - begin

            if audio is None:
                audio = load_audio(audio_path, frame_offset=begin, num_frames=num_frames, normalize=False).to(
                    self.device
                )
            else:
                audio = audio[begin:end].to(self.device)
            start_frame = begin
            end_frame = end

        else:
            if audio is None:
                audio = load_audio(audio_path, normalize=False).to(self.device)
            else:
                audio = audio.to(self.device)

            start_frame = 0
            end_frame = audio_length

        return Audio(
            audio_path=audio_path,
            audio=audio,
            start_time=start_frame / SAMPLE_RATE,
            end_time=end_frame / SAMPLE_RATE,
        )

    def __len__(self):
        return len(self.file_list)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError


class MTGJamendo(UnlabbeledAudioDataset):
    def __init__(
        self,
        path: str,
        groups=None,
        sequence_length=327680,
        seed=42,
        device=DEFAULT_DEVICE,
        num_files=None,
        max_files_in_memory=-1,
        reproducable_load_sequences=False,
        skip_missing_tracks=False,
    ):
        self.skip_missing_tracks = skip_missing_tracks
        super().__init__(
            path,
            groups=groups if groups is not None else ["train"],
            sequence_length=sequence_length,
            seed=seed,
            device=device,
            num_files=num_files,
            max_files_in_memory=max_files_in_memory,
            reproducable_load_sequences=reproducable_load_sequences,
        )

    @classmethod
    def available_groups(cls):
        return ["train", "validation", "test"]

    def files(self, group):
        audio_files = sorted(glob(os.path.join(self.path, group, "*", "*.low.mp3")))
        if len(audio_files) == 0:
            raise RuntimeError(f"Group {group} is empty")
        return audio_files
