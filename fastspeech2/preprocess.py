import json

import numpy as np
import os
import tgt
from datasets import Audio
from scipy.io.wavfile import read
import pyworld as pw
import torch
from . import hparams as hp


def build_from_path(in_dir, out_dir):
    index = 1

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        i = 0
        for line in f:
            parts = line.strip().split('|')
            basename = parts[0]

            pitch, energy = process_utterance(in_dir, out_dir, basename, i)

            if index % 100 == 0:
                print("Done %d" % index)
            index = index + 1
            i += 1

    pitch_min, pitch_max = max_finder(
        os.path.join(out_dir, "f0")
    )
    energy_min, energy_max = max_finder(
        os.path.join(out_dir, "energy")
    )
    with open(os.path.join(out_dir, "min_max_data.json"), "w") as f:
        stats = {
            "pitch": [
                float(pitch_min),
                float(pitch_max),
            ],
            "energy": [
                float(energy_min),
                float(energy_max),
            ],
        }
        f.write(json.dumps(stats))

    return pitch_min, pitch_max, energy_min, energy_max


def process_utterance(in_dir, out_dir, basename, iteration):
    wav_path = os.path.join(in_dir, 'wavs', '{}.wav'.format(basename))

    duration = np.load(os.path.join(
        "/content/alignments", str(iteration) + ".npy"))

    _, wav = read(wav_path)
    wav = wav.astype(np.float32)

    # Compute fundamental frequency
    f0, _ = pw.dio(wav.astype(np.float64), 22050, 256 / 22050 * 1000)
    f0 = f0[:sum(duration)]

    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(
        torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)[
                      :, :sum(duration)]
    # print(energy)
    energy = energy.astype(np.float32)[:sum(duration)]
    # print(energy)

    f0_filename = '{}-f0.npy'.format(iteration)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    energy_filename = '{}-energy.npy'.format(iteration)
    np.save(os.path.join(out_dir, 'energy', energy_filename),
            energy, allow_pickle=False)
    return f0, energy


def max_finder(in_dir):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        values = np.load(filename)
        max_value = max(max_value, max(values))
        min_value = min(min_value, min(values))
    return min_value, max_value
