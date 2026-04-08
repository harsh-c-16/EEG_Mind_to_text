from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

SpellerMatrix = np.ndarray


@dataclass(slots=True)
class BCIConfig:
    sfreq: float = 250.0
    notch_freq: float = 50.0
    bandpass_low: float = 0.1
    bandpass_high: float = 30.0
    epoch_tmin: float = 0.0
    epoch_tmax: float = 0.8
    training_trials: int = 15
    random_state: int = 42
    test_size: float = 0.25
    window_samples: Tuple[int, int] | None = None
    letters_per_word: int = 5
    rows: int = 6
    cols: int = 6
    speller_matrix: np.ndarray = field(default_factory=lambda: default_speller_matrix())


def default_speller_matrix() -> np.ndarray:
    return np.array(
        [
            ["A", "B", "C", "D", "E", "F"],
            ["G", "H", "I", "J", "K", "L"],
            ["M", "N", "O", "P", "Q", "R"],
            ["S", "T", "U", "V", "W", "X"],
            ["Y", "Z", "1", "2", "3", "4"],
            ["5", "6", "_", "<", "?", "!"],
        ],
        dtype=object,
    )


@dataclass(slots=True)
class FeatureConfig:
    use_psd: bool = True
    use_csp: bool = True
    csp_components: int = 4
    bands: Tuple[Tuple[float, float], ...] = (
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 13.0),
        (13.0, 30.0),
    )


@dataclass(slots=True)
class ModelConfig:
    classifier: str = "svm"
    probability: bool = True
