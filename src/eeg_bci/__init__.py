"""EEG BCI package for P300 mind-to-text decoding."""

from .config import BCIConfig, SpellerMatrix
from .dataset import EEGDataset, FlashEvent, load_dataset
from .training import train_and_save

__all__ = [
    "BCIConfig",
    "SpellerMatrix",
    "EEGDataset",
    "FlashEvent",
    "load_dataset",
    "train_and_save",
]
