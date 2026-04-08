from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BCIConfig
from .dataset import assign_trials_to_events, extract_flash_events, load_dataset, split_trials
from .decoding import decode_trial_letters, decoded_trials_to_words
from .modeling import load_artifacts
from .preprocessing import extract_epochs, preprocess_eeg


def predict_words(data_path: str | Path, model_dir: str | Path, config: BCIConfig | None = None) -> list[str]:
    config = config or BCIConfig()
    dataset = load_dataset(data_path, sfreq=config.sfreq)
    artifacts = load_artifacts(model_dir)

    preprocessed = preprocess_eeg(
        dataset.X,
        sfreq=config.sfreq,
        notch_freq=config.notch_freq,
        band_low=config.bandpass_low,
        band_high=config.bandpass_high,
    )

    flash_events = assign_trials_to_events(extract_flash_events(dataset), dataset.trial)
    valid_events = [event for event in flash_events if event.label is not None]
    event_samples = np.asarray([event.sample for event in valid_events], dtype=int)
    event_trials = np.asarray([event.trial_id for event in valid_events], dtype=int)
    event_stimulations = np.asarray([event.stimulation for event in valid_events], dtype=int)

    epochs, valid_indices = extract_epochs(
        preprocessed,
        event_samples,
        sfreq=config.sfreq,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
    )
    event_trials = event_trials[valid_indices]
    event_stimulations = event_stimulations[valid_indices]

    features = artifacts.feature_extractor.transform(epochs)
    probabilities = _positive_scores(artifacts.model, features)

    train_trials, test_trials = split_trials(len(np.unique(dataset.trial)), training_trials=config.training_trials)
    test_mask = np.isin(event_trials, test_trials)
    decoded_trials = decode_trial_letters(
        trial_ids=event_trials[test_mask],
        stimulations=event_stimulations[test_mask],
        probabilities=probabilities[test_mask],
        speller_matrix=config.speller_matrix,
    )
    return decoded_trials_to_words(decoded_trials, letters_per_word=config.letters_per_word)


def _positive_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(X), dtype=float)
