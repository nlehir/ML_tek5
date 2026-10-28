"""Download the free MUSDB18 7-second sample and convert it to WAV files.

Usage (from the repo root):  uv run my_models/0_download_data.py
Needs ffmpeg (sudo apt install ffmpeg).

MUSDB18-7 is a free preview of MUSDB18: one 7 s excerpt per song, as
.stem.mp4 files (AAC) with 5 streams: mixture, drums, bass, other, vocals.
As we focus on vocal removal only, we only keep the mixture and the vocals, and build
accompaniment = mixture - vocals.

Result (one folder per clip, 44.1 kHz stereo):
    my_models/data/musdb7/train/<song>/{mixture,vocals,accompaniment}.wav  (94 clips)
    my_models/data/musdb7/test/<song>/{mixture,vocals,accompaniment}.wav   (50 clips)
"""

import subprocess
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

from src.data_preparation import DATA_DIR

URL = "https://zenodo.org/record/3270814/files/MUSDB18-7-STEMS.zip"
# Streams inside a .stem.mp4: 0 = mixture, 1 = drums, 2 = bass, 3 = other, 4 = vocals
STREAMS = {"mixture": 0, "vocals": 4}


def ffmpeg_extract(src: Path, stream: int, dst: Path):
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    command += ["-map", f"0:{stream}", "-ar", "44100", "-ac", "2", str(dst)]
    subprocess.run(args=command, check=True)


# ---- 1. Download and unzip ----
zip_path = DATA_DIR / "MUSDB18-7-STEMS.zip"
if not zip_path.exists():
    print("Downloading", URL, "(147 MB)...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    part = zip_path.with_suffix(".part")
    urllib.request.urlretrieve(url=URL, filename=part)
    part.rename(zip_path)  # only keep complete downloads
raw = DATA_DIR / "musdb7_raw"
if not raw.exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)

# ---- 2. Convert each clip to WAV ----
for stem_file in sorted(raw.rglob("*.stem.mp4")):
    split = "test" if "test" in stem_file.parts else "train"
    clip_dir = DATA_DIR / "musdb7" / split / stem_file.name.replace(".stem.mp4", "")
    if (clip_dir / "accompaniment.wav").exists():
        continue
    clip_dir.mkdir(parents=True, exist_ok=True)
    for name, stream in STREAMS.items():
        ffmpeg_extract(src=stem_file, stream=stream, dst=clip_dir / f"{name}.wav")
    # accompaniment = everything except vocals
    mix, sr = sf.read(file=str(clip_dir / "mixture.wav"), dtype="float32")
    voc, _ = sf.read(file=str(clip_dir / "vocals.wav"), dtype="float32")
    n = min(len(mix), len(voc))
    sf.write(
        file=str(clip_dir / "accompaniment.wav"), data=mix[:n] - voc[:n], samplerate=sr
    )
    print(split, clip_dir.name)

for split in ("train", "test"):
    print(split, len(list((DATA_DIR / "musdb7" / split).iterdir())), "clips")
