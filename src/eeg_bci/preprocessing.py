from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


def notch_filter(X: np.ndarray, sfreq: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq, quality, sfreq)
    return _apply_iir(X, b, a)


def bandpass_filter(X: np.ndarray, sfreq: float, low: float = 0.1, high: float = 30.0, order: int = 4) -> np.ndarray:
    nyquist = 0.5 * sfreq
    low_norm = max(low / nyquist, 1e-5)
    high_norm = min(high / nyquist, 0.999)
    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    return _apply_sos(X, sos)


def preprocess_eeg(
    X: np.ndarray,
    sfreq: float,
    apply_notch: bool = True,
    notch_freq: float = 50.0,
    band_low: float = 0.1,
    band_high: float = 30.0,
) -> np.ndarray:
    signal = np.asarray(X, dtype=float)
    if apply_notch:
        signal = notch_filter(signal, sfreq=sfreq, freq=notch_freq)
    signal = bandpass_filter(signal, sfreq=sfreq, low=band_low, high=band_high)
    signal = signal - np.mean(signal, axis=-1, keepdims=True)
    std = np.std(signal, axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return signal / std


def extract_epochs(
    X: np.ndarray,
    event_samples: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(event_samples, dtype=int).ravel()
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    n_times = stop_offset - start_offset

    epochs = []
    valid_indices = []
    for idx, sample in enumerate(samples):
        start = sample + start_offset
        stop = sample + stop_offset
        if start < 0 or stop > X.shape[1]:
            continue
        epochs.append(X[:, start:stop])
        valid_indices.append(idx)

    if not epochs:
        return np.empty((0, X.shape[0], n_times)), np.array([], dtype=int)

    return np.stack(epochs), np.asarray(valid_indices, dtype=int)


def _apply_iir(X: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return np.asarray([filtfilt(b, a, channel) for channel in X])
    if X.ndim == 3:
        return np.asarray([_apply_iir(epoch, b, a) for epoch in X])
    raise ValueError(f"Expected 2D or 3D EEG array, got shape {X.shape}")


def _apply_sos(X: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return np.asarray([sosfiltfilt(sos, channel) for channel in X])
    if X.ndim == 3:
        return np.asarray([_apply_sos(epoch, sos) for epoch in X])
    raise ValueError(f"Expected 2D or 3D EEG array, got shape {X.shape}")
