"""Simulate a song recorded with a phone in a room: room reverb, then a band-pass.

Both effects are linear: degrade(a + b) = degrade(a) + degrade(b). So applying
the same chain (same random settings) to the vocals and the accompaniment of a
clip, then summing them, gives exactly the degraded mixture.
"""

import numpy as np
from scipy import signal

SR = 44100

# Ranges for the random degradation settings (one draw per clip)
PARAM_RANGES = {
    "rt60": (0.3, 1.0),  # reverb decay time in seconds (small room -> large room)
    "wet": (0.2, 0.5),  # share of reverberated sound
    "low_cut": (100.0, 300.0),  # Hz, phone microphones lose the low bass
    "high_cut": (3000.0, 5000.0),  # Hz, and the high frequencies
}


def draw_params(rng: np.random.Generator):
    return {
        name: float(rng.uniform(low=lo, high=hi))
        for name, (lo, hi) in PARAM_RANGES.items()
    }


def make_impulse_response(rt60: float, rng: np.random.Generator, channels: int = 2):
    """Synthetic room: exponentially decaying noise, independent per channel."""
    length = int(rt60 * 1.5 * SR)
    t = np.arange(length) / SR
    decay = np.exp(-6.91 * t / rt60)  # -60 dB after rt60 seconds
    ir = rng.standard_normal((length, channels)) * decay[:, None]
    ir /= np.sqrt(np.sum(a=ir**2, axis=0, keepdims=True))  # unit energy per channel
    return ir.astype(np.float32)


def degrade(audio: np.ndarray, params: dict[str, float], ir: np.ndarray):
    """Reverb then band-pass. audio: [samples, channels]."""
    n_samples = audio.shape[0]
    n_channels = audio.shape[1]
    """
    reverberated: the "wet" signal
    reverberated has the same shape as audio (n_samples, channels)
    """
    reverberated = np.stack(
        arrays=[
            signal.fftconvolve(in1=audio[:, c], in2=ir[:, c])[:n_samples]
            for c in range(n_channels)
        ],
        axis=1,
    )
    """
    Blend the raw audio (dry signal) with the reverberated signal (wet)
    """
    out = (1 - params["wet"]) * audio + params["wet"] * reverberated
    """
    Define the Butterworth filter
    sos stands for "second order sections" (scipy)
    """
    sos = signal.butter(
        N=4,  # filter order
        Wn=[params["low_cut"], params["high_cut"]],  # critical frequencies
        btype="band",  # type of filter
        fs=SR,  # sampling rate
        output="sos",
    )
    """
    Return the filtered signal
    """
    return np.asarray(a=signal.sosfilt(sos=sos, x=out, axis=0), dtype=np.float32)
