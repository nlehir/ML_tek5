"""Step 1: build the benchmark model (no training) and add it to UVR.

Usage (from the repo root):  uv run my_models/1_create_benchmark_model.py
Then in UVR: Process Method = MDX-Net, model = Benchmark_CentreChannel.
Keep Segment Size = 256, otherwise UVR doesn't use the ONNX file directly.
"""

import torch
from torch import nn

from src.model_saving import MDX_DIR, export_onnx, install_in_uvr


class CentreChannel(nn.Module):
    """
    Super basic benchmark model (we do not expect good vocal removal with this one !)

    Vocals are usually mixed in the centre, i.e. identical in both channels.

    Estimate vocals = (Left + Right) / 2 in both channels (= Mono mix)

    UVR then outputs instrumental = mixture - vocals = the difference signal:
    (Left - Right) / 2 on the left, its opposite on the right.
    Zero parameters.
    """

    def forward(self, spectrogram: torch.Tensor):
        """
        spectrogram: shape [batch_size, 4, DIM_F, DIM_T]
        Each spectrogram has 4 channels:
            - Left real part
            - Left imaginary part
            - Right real part
            - Right imaginary part
        the slice [:, 0:2] yields the left channel: (real, imaginary),
            shape [batch_size, 2, DIM_F, DIM_T]
        the slice [:, 2:4] yields the right channel: (real, imaginary)
        Hence the following formula yields the sum of the two channels, divided by 2.
        """
        mid = (spectrogram[:, 0:2] + spectrogram[:, 2:4]) / 2
        """
        Send (L+R)/2 to the left and the right channel of the spectrogram
        """
        return torch.cat(tensors=[mid, mid], dim=1)


out = MDX_DIR / "Benchmark_CentreChannel.onnx"
export_onnx(model=CentreChannel(), path=out)
install_in_uvr(out)
