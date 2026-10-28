"""Training data: MUSDB18 7-second clips, cut into random chunks."""

import random
from pathlib import Path

import soundfile as sf
import torch

from src.spectrograms import N_SAMPLES_IN_CHUNK

# Downloaded and converted by 0_download_data.py
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_DIR = DATA_DIR / "musdb7" / "train"
TEST_DIR = DATA_DIR / "musdb7" / "test"
# "Phone recording" copy of the same clips, made by 5_degrade_data.py
PHONE_TRAIN_DIR = DATA_DIR / "musdb7_phone" / "train"
PHONE_TEST_DIR = DATA_DIR / "musdb7_phone" / "test"


def load(path: Path):
    return torch.tensor(sf.read(file=str(path), dtype="float32")[0].T)  # [2, samples]


def load_clips(clips_dir: Path):
    """Load every clip into memory: a list of (vocals, accompaniment) waveforms."""
    return [
        (load(clip / "vocals.wav"), load(clip / "accompaniment.wav"))
        for clip in sorted(clips_dir.iterdir())
    ]


def random_chunk(wave: torch.Tensor):
    """
    wave: stereo waveform
    Extract a subpart of the waveform.
    """
    start = random.randint(a=0, b=wave.shape[1] - N_SAMPLES_IN_CHUNK)
    return wave[:, start : start + N_SAMPLES_IN_CHUNK]


def make_batch(
    clips: list[tuple[torch.Tensor, torch.Tensor]], batch_size: int, remix: bool
):
    """
    Returns (mixture, vocals) waveforms,
    in a batch of dimension [batch_size, 2, N_SAMPLES_IN_CHUNK]
    The 2 stands for the Left and Right channels.

    The model processes the whole batch in one training step, and the loss is averaged over it.

    remix=True: vocals of one clip + accompaniment of another, which makes
    many new 'songs' from few (data augmentation).
    """
    vocals, mixes = [], []
    for _ in range(batch_size):
        v, a = random.choice(clips)
        if remix:
            a = random.choice(clips)[1]
        v, a = random_chunk(v), random_chunk(a)
        vocals.append(v)
        mixes.append(v + a)
    return torch.stack(mixes), torch.stack(vocals)
