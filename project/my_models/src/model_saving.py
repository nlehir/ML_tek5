"""Save a model where UVR's MDX-Net engine can load it: ONNX export, hash, settings.

How UVR's MDX-Net engine calls an ONNX model (separate.py, SeperateMDX):
  - cuts the song into chunks of hop * (dim_t - 1) samples,
  - computes a spectrogram of shape [batch, 4, dim_f, dim_t]
    (the 4 channels are Left-real, Left-imag, Right-real, Right-imag),
  - runs the model: output = spectrogram of the *primary stem*, same shape,
  - turns it back into audio; the other stem is mixture - primary stem.
Any network with that input/output contract can be loaded by UVR.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from src.spectrograms import DIM_F, DIM_T, N_SAMPLES_IN_WINDOW

UVR_DIR = Path(__file__).resolve().parents[2]
MDX_DIR = UVR_DIR / "models" / "MDX_Net_Models"


def export_onnx(model: nn.Module, path: Path, dim_f: int = DIM_F):
    model = model.cpu().eval()
    dummy = torch.zeros(1, 4, dim_f, DIM_T)
    torch.onnx.export(
        model=model,
        args=(dummy,),
        f=str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print("exported", path)


def uvr_hash(path: Path):
    """UVR identifies a model by the MD5 of its last 10 MB (UVR.py, get_model_hash)."""
    with open(file=path, mode="rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()


def load_settings(onnx_path: Path):
    """Find the model's settings the way UVR does: first model_data/<hash>.json
    (our models), then the shared model_data/model_data.json (UVR's models)."""
    model_hash = uvr_hash(onnx_path)
    own_json = MDX_DIR / "model_data" / f"{model_hash}.json"
    if own_json.exists():
        return json.loads(own_json.read_text())
    shared = json.loads(
        (MDX_DIR / "model_data" / "model_data.json").read_text(encoding="utf-8")
    )
    return shared[model_hash]


def install_in_uvr(onnx_path: Path, settings: dict | None = None):
    """Write the settings of a model already in UVR's MDX-Net folder, so the
    model appears under MDX-Net with no "unrecognised model" pop-up.
    UVR reads these settings (not the ONNX file) to build the model's input
    spectrogram, name its output stem and scale its output.

    settings=None: the model takes and returns spectrograms in this repo's
    default format (stft() defaults: N_SAMPLES_IN_WINDOW, DIM_F; DIM_T
    frames) and outputs the vocals.

    Otherwise, pass the settings the model was trained with, e.g. the base
    model's settings for a fine-tuned UVR model
    """
    if settings is None:
        settings = {
            # UVR multiplies the primary stem by this factor. Ideally, should be the factor
            # that maximises SDR, measured on the validation set; 1.0 = no
            # rescaling (we don't measure it)
            "compensate": 1.0,
            "mdx_dim_f_set": DIM_F,
            "mdx_dim_t_set": DIM_T.bit_length() - 1,  # power of 2: 2**8 = 256
            "mdx_n_fft_scale_set": N_SAMPLES_IN_WINDOW,
            "primary_stem": "Vocals",
        }
    json_path = MDX_DIR / "model_data" / f"{uvr_hash(onnx_path)}.json"
    json_path.write_text(json.dumps(obj=settings, indent=4))
    print("settings", json_path)
