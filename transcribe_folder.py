import argparse
import os

import numpy as np
import torch
import torchaudio
from mir_eval.util import midi_to_hz
from slakh_dataset.data_classes import MusicAnnotation

from onsets_and_frames.constants import HOP_LENGTH, MIN_MIDI, SAMPLE_RATE
from onsets_and_frames.decoding import extract_notes
from onsets_and_frames.midi import save_midi
from onsets_and_frames.utils import summary


def load_and_process_audio(audio_path, sequence_length, device):

    random = np.random.RandomState(seed=42)

    audio_info = torchaudio.info(audio_path)
    sr = audio_info.sample_rate

    if sequence_length is not None:
        audio_length = audio_info.num_frames
        step_begin = random.randint(audio_length -
                                    sequence_length) // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length
        num_frames = end - begin

        audio = torchaudio.load(audio_path,
                                frame_offset=begin,
                                num_frames=num_frames)[0].to(device)
    else:
        audio = torchaudio.load(audio_path)[0].to(device)

    if audio_info.num_channels == 2:
        audio = torch.mean(audio, 0)
        audio = torch.unsqueeze(audio, 0)

    if sr != SAMPLE_RATE:
        audio = audio.to("cpu")
        audio = torchaudio.transforms.Resample(orig_freq=sr,
                                               new_freq=SAMPLE_RATE)(audio)
        audio = audio.to(device)

    assert len(audio.shape) == 2 and audio.shape[0] == 1
    return audio


def transcribe_folder(
    model_file,
    audio_paths,
    save_folder,
    is_drum,
    sequence_length,
    onset_threshold,
    frame_threshold,
    device,
):

    model = torch.load(model_file, map_location=device).to(device).eval()
    summary(model)
    print(audio_paths)

    for audio_path in audio_paths:
        print(f"Processing {audio_path}")

        try:
            audio = load_and_process_audio(audio_path, sequence_length, device)
            mel = model.mel(audio)
            onset_pred, offset_pred, _, frame_pred, velocity_pred = model(mel)

            pred = MusicAnnotation(
                onset=onset_pred.reshape(
                    (onset_pred.shape[1], onset_pred.shape[2])),
                offset=offset_pred.reshape(
                    (offset_pred.shape[1], offset_pred.shape[2])),
                frame=frame_pred.reshape(
                    (frame_pred.shape[1], frame_pred.shape[2])),
                velocity=velocity_pred.reshape(
                    (velocity_pred.shape[1], velocity_pred.shape[2]))
                if velocity_pred is not None else None,
            )

            p_est, i_est, v_est = extract_notes(pred.onset, pred.frame,
                                                pred.velocity, onset_threshold,
                                                frame_threshold)

            # pred_notes = notes_music_annotation(p_est, i_est, pred.frame.shape)
            scaling = HOP_LENGTH / SAMPLE_RATE

            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            transcription_folder = f"{audio_path}_o{str(onset_threshold)[-1]}f{str(frame_threshold)[-1]}"
            transcription_folder = os.path.join(save_folder,
                                                transcription_folder)
            os.makedirs(transcription_folder, exist_ok=True)
            programs = {
                0: "piano",
                1: "brite",
                3: "honkeytonk",
                4: "elpiano",
                6: "harpsi",
                16: "organ",
                17: "hammond",
                24: "guitar-nylon",
                25: "guitar-steel",
                73: "flute",
                74: "recorder",
                56: "trumpet",
                42: "viola",
                35: "el-bass"
            }
            for program in programs:
                midi_path = os.path.join(
                    transcription_folder,
                    f"{program}_{programs[program]}" + ".pred.mid")
                print(f"midipath: {midi_path}")
                save_midi(
                    midi_path,
                    p_est,
                    i_est,
                    v_est,
                    midi_program=program,
                    is_drum=is_drum,
                )
        except:
            print(f"An error occured when transcribing file {audio_path}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("folder_path", type=str)
    parser.add_argument("--save-folder", type=str, default="output")
    parser.add_argument("--midi-program", default=0, type=int)
    parser.add_argument("--is-drum", action="store_true")
    parser.add_argument("--sequence-length", default=None, type=int)
    parser.add_argument("--onset-threshold", default=0.5, type=float)
    parser.add_argument("--frame-threshold", default=0.5, type=float)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # save_folder = os.path.join(args.save_folder,
    #                           args.model_file.replace(os.sep, "-"))

    save_folder = os.path.join(args.save_folder)
    folder_path = args.folder_path
    filenames = os.listdir(folder_path)
    filepaths = []
    for filename in filenames:
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            filepaths.append(filepath)

    with torch.no_grad():
        transcribe_folder(
            model_file=args.model_file,
            audio_paths=filepaths,
            save_folder=save_folder,
            is_drum=args.is_drum,
            sequence_length=args.sequence_length,
            onset_threshold=args.onset_threshold,
            frame_threshold=args.frame_threshold,
            device=args.device,
        )