from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any

import json
import numpy as np

from .config import BCIConfig, FeatureConfig, ModelConfig
from .dataset import assign_trials_to_events, extract_flash_events, load_dataset, split_trials
from .decoding import decoded_trials_to_words, decode_trial_letters
from .features import FeatureExtractor
from .modeling import ModelArtifacts, build_classifier, evaluate_model, save_artifacts
from .preprocessing import extract_epochs, preprocess_eeg


@dataclass(slots=True)
class TrainingResult:
    metrics: dict[str, Any]
    decoded_trials: list[Any]
    decoded_words: list[str]
    artifacts_path: Path


def train_and_save(
    data_path: str | Path,
    model_dir: str | Path,
    config: BCIConfig | None = None,
    feature_config: FeatureConfig | None = None,
    model_config: ModelConfig | None = None,
) -> TrainingResult:
    config = config or BCIConfig()
    feature_config = feature_config or FeatureConfig()
    model_config = model_config or ModelConfig()

    dataset = load_dataset(data_path, sfreq=config.sfreq)
    preprocessed = preprocess_eeg(
        dataset.X,
        sfreq=config.sfreq,
        notch_freq=config.notch_freq,
        band_low=config.bandpass_low,
        band_high=config.bandpass_high,
    )

    flash_events = assign_trials_to_events(extract_flash_events(dataset), dataset.trial)
    valid_events = [event for event in flash_events if event.label is not None]
    if not valid_events:
        raise ValueError("No labeled flash events were found in the dataset")

    event_samples = np.asarray([event.sample for event in valid_events], dtype=int)
    event_labels = np.asarray([event.label for event in valid_events], dtype=int)
    event_trials = np.asarray([event.trial_id for event in valid_events], dtype=int)
    event_stimulations = np.asarray([event.stimulation for event in valid_events], dtype=int)

    epochs, valid_indices = extract_epochs(
        preprocessed,
        event_samples,
        sfreq=config.sfreq,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
    )
    if epochs.size == 0:
        raise ValueError("No valid epochs could be extracted from the EEG signal")

    event_labels = event_labels[valid_indices]
    event_trials = event_trials[valid_indices]
    event_stimulations = event_stimulations[valid_indices]

    train_trials, test_trials = split_trials(len(np.unique(dataset.trial)), training_trials=config.training_trials)
    train_mask = np.isin(event_trials, train_trials)
    test_mask = np.isin(event_trials, test_trials)

    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Training or test split is empty; check the trial structure in your dataset")

    feature_extractor = FeatureExtractor(config=feature_config)
    X_train = feature_extractor.fit(epochs[train_mask], event_labels[train_mask]).transform(epochs[train_mask])
    X_test = feature_extractor.transform(epochs[test_mask])

    classifier = build_classifier(model_config.classifier, random_state=config.random_state)
    classifier.fit(X_train, event_labels[train_mask])

    metrics = evaluate_model(classifier, X_test, event_labels[test_mask])
    decoded_trials = decode_trial_letters(
        trial_ids=event_trials[test_mask],
        stimulations=event_stimulations[test_mask],
        probabilities=_positive_scores(classifier, X_test),
        speller_matrix=config.speller_matrix,
    )
    decoded_words = decoded_trials_to_words(decoded_trials, letters_per_word=config.letters_per_word)

    artifacts = ModelArtifacts(
        model=classifier,
        feature_extractor=feature_extractor,
        metadata={
            "config": asdict(config),
            "feature_config": asdict(feature_config),
            "model_config": asdict(model_config),
            "trial_ids": dataset.trial.tolist(),
            "train_trials": train_trials.tolist(),
            "test_trials": test_trials.tolist(),
            "metrics": metrics,
        },
    )

    artifacts_path = save_artifacts(model_dir, artifacts)
    summary_path = Path(model_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"metrics": metrics, "decoded_words": decoded_words}, indent=2), encoding="utf-8")

    return TrainingResult(metrics=metrics, decoded_trials=decoded_trials, decoded_words=decoded_words, artifacts_path=artifacts_path)


def _positive_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(X), dtype=float)
