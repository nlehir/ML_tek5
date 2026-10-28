"""Compute spectrogram from audio.

Spectrograms are what the inputs expected by UVR's MDX-Net engine.

The audio constants defined below define the format of the spectrogram.

How UVR turns a song into model inputs (44.1 kHz stereo, hop = 1024 samples,
fixed in separate.py):

1. The song is cut into chunks of hop * (segment_size - 1) samples.
   With segment_size = 256: 1024 * 255 = 261120 samples, about 5.9 s.
2. Each chunk goes through an STFT (Short-Term-Fourier-Transform): a window of N_SAMPLES_IN_WINDOW samples
   slides along the audio in steps of hop samples, and each position gives one
   "frame" (one column of the spectrogram).
3. Each frame has N_SAMPLES_IN_WINDOW // 2 + 1 frequency bins; UVR keeps the
   first DIM_F.
4. The model receives spectrograms of shape [batch_size, 4, DIM_F, segment_size]
   (4 = Left-real, Left-imag, Right-real, Right-imag).
"""

import torch

# Step between two STFT windows, in samples. Fixed at 1024 by UVR for
# MDX-Net ONNX models (separate.py:218).
HOP = 1024

# Number of audio samples in each STFT window, i.e. the FFT size
# (called n_fft in torch.stft, and mdx_n_fft_scale_set in UVR's JSON files).
#   4096 samples = 93 ms at 44.1 kHz. The window moves by hop = 1024 samples
#   (23 ms), so consecutive windows overlap by 75%.
#   It gives N_SAMPLES_IN_WINDOW // 2 + 1 = 2049 frequency bins, from 0 Hz to
#   22050 Hz (half the sample rate), spaced 44100 / 4096 = 10.8 Hz apart.
#   Bigger window: finer frequency detail, blurrier timing.
N_SAMPLES_IN_WINDOW = 4096

# Number of frequency bins given to the model (the rows of the spectrogram).
#   UVR keeps bins 0..2047 of the 2049, i.e. 0 Hz to ~22 kHz: only the very
#   top bin is dropped, and UVR puts it back as zeros when converting back to
#   audio. 2048 is a power of 2, which suits networks that halve the
#   frequency axis several times.
DIM_F = 2048

# Number of frames per chunk (the columns of the spectrogram).
#   This corresponds to the GUI's "Segment Size" (default 256): 256 frames = one chunk of
#   about 5.9 s. A bigger segment gives the model more context per chunk but
#   uses more memory.
#   UVR runs the ONNX file directly only if the GUI's Segment Size equals
#   DIM_T (separate.py:489). Otherwise it converts the file with onnx2pytorch,
#   which may not support our layers. So keep Segment Size = 256 in the GUI.
#   In the hash JSON it is stored as a power of 2: mdx_dim_t_set = 8 (2**8 = 256).
DIM_T = 256

# Length of one chunk in samples: 1024 * 255 = 261120, about 5.9 s.
N_SAMPLES_IN_CHUNK = HOP * (DIM_T - 1)


def stft(
    wave: torch.Tensor,
    n_samples_in_window: int = N_SAMPLES_IN_WINDOW,
    dim_f: int = DIM_F,
):
    """
        stft stands for "Short Term Fourier Transform"

    [batch_size, 2, n_samples] -> [batch_size, 4, dim_f, n_frames], same as UVR's STFT class (lib_v5/tfc_tdf_v3.py).

    dim_f (int): number of frequency bins given to the model: typically 2048

    n_samples_in_window (int): number of audio samples used in an FFT (Fast Fourier Transform) window

    wave: contains a batch of chunks
        torch.Tensor of dimension [batch_size, 2, n_samples]
        batch_size: number of chunks passed (e.g. 4, 8)
        2 stands for stereo: left + right
        n_samples: number of samples in an audio chunk.
        A chunk is N_SAMPLES_IN_CHUNK = 255 × 1024 samples.
        The STFT turns it into 256 frames, one every 1024 samples (23 ms),
        each computed from a 4096-sample window.

    returns spectrogram
        torch.Tensor of dimension [batch_size, 4, dim_f, n_frames]
        dim_f: number of frequency bins: might depend on the model

    A wave of N_SAMPLES_IN_CHUNK samples gives DIM_T frames.

    The 2 input channels are L and R; the 4 output channels are
    L-real, L-imag, R-real, R-imag.

    n_samples_in_window and dim_f default to our models' settings; UVR's own
    models use others (e.g. 6144 and 3072 for UVR-MDX-NET-Inst_HQ_3)."""
    batch_size, n_channels, n_samples = wave.shape  # n_channels = 2 (L, R)
    window = torch.hann_window(
        window_length=n_samples_in_window, periodic=True, device=wave.device
    )
    """
    Perform the Short-Term-Fourier-Transform (STFT)

    F (number of frequency bins) = n_samples_in_window // 2 + 1, from 0 Hz to 22050 Hz
    (half the sample rate)
    F=2049 with n_samples_in_window=4096
    F=3073 for HQ 3, as wider windows are used by this model.

    The return line keeps only the first dim_f of them (2048, or 3072 for HQ 3).

    torch.stft() returns complex values
    """
    spectrogram = torch.stft(
        # torch.stft takes a list of 1-D signals: each channel of each chunk
        # becomes one signal, [batch_size * n_channels, n_samples]
        input=wave.reshape(batch_size * n_channels, n_samples),
        n_fft=n_samples_in_window,
        hop_length=HOP,
        window=window,
        center=True,
        return_complex=True,
    )  # [batch_size*2, F, DIM_T]
    """
    Convert the complex tensor to a real tensor with an extra dimension,
    for real and imaginary part.
    """
    spectrogram = torch.view_as_real(spectrogram).permute(
        0, 3, 1, 2
    )  # [batch_size*2, 2, F, DIM_T]
    """
    Final reshape and removal of the highest frequency bin
    to pass the spectrogram to the model.

    Dimension 2 has 4 elements:
        - Left channel, real part
        - Left channel, imaginary part
        - Right channel, real part
        - Right channel, imaginary part
    """
    n_bins = spectrogram.shape[2]  # F
    n_frames = spectrogram.shape[3]  # DIM_T
    return spectrogram.reshape(batch_size, 4, n_bins, n_frames)[:, :, :dim_f]
