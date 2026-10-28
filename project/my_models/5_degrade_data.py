"""Make a "phone recording" copy of the MUSDB18 7-second clips.

Usage (from the repo root):  uv run my_models/5_degrade_data.py
Needs the clean clips (downloaded by 0_download_data.py).

For each clip, one random room and phone filter (src/degradation.py) is
applied to the vocals and to the accompaniment; the degraded mixture is their
sum. The degraded stems are the new labels: the task is "separate a phone
recording", not "separate and restore the original sound".

Result, same layout as the clean data, plus the settings used:
    my_models/data/musdb7_phone/{train,test}/<clip>/
        mixture.wav, vocals.wav, accompaniment.wav, degradation.json
"""

import json
import zlib

import numpy as np
import soundfile as sf

from src.data_preparation import PHONE_TEST_DIR, PHONE_TRAIN_DIR, TEST_DIR, TRAIN_DIR
from src.degradation import SR, degrade, draw_params, make_impulse_response

SEED = 0

for clean_dir, phone_dir in [(TRAIN_DIR, PHONE_TRAIN_DIR), (TEST_DIR, PHONE_TEST_DIR)]:
    for clip in sorted(clean_dir.iterdir()):
        out = phone_dir / clip.name
        if (out / "degradation.json").exists():
            continue  # already done
        """
        Create a random generator per clip.
        On a new run of the same file, the settings will be identical
        (Useful for recall)
        """
        rng = np.random.default_rng([SEED, zlib.crc32(clip.name.encode())])
        params = draw_params(rng)
        """
        Compute the audio transformation operator
        """
        ir = make_impulse_response(rt60=params["rt60"], rng=rng)
        stems = {}
        for stem in ["vocals", "accompaniment"]:
            audio = sf.read(file=str(clip / f"{stem}.wav"), dtype="float32")[0]
            """
            Apply the transformation
            """
            stems[stem] = degrade(audio=audio, params=params, ir=ir)  # [samples, 2]
        n = min(len(a) for a in stems.values())
        stems = {stem: a[:n] for stem, a in stems.items()}
        stems["mixture"] = stems["vocals"] + stems["accompaniment"]
        # one common gain for all files: keeps mixture = vocals + accompaniment
        gain = min(1.0, 0.9 / max(np.abs(a).max() for a in stems.values()))
        """
        Finally save the new audio files
        """
        out.mkdir(parents=True, exist_ok=True)
        for stem, audio in stems.items():
            sf.write(file=str(out / f"{stem}.wav"), data=audio * gain, samplerate=SR)
        (out / "degradation.json").write_text(json.dumps(obj=params, indent=2))
        print(phone_dir.name, clip.name)

for phone_dir in [PHONE_TRAIN_DIR, PHONE_TEST_DIR]:
    print(phone_dir.name, len(list(phone_dir.iterdir())), "clips")
