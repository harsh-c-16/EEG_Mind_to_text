from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import io as sio


@dataclass(slots=True)
class EEGDataset:
    X: np.ndarray
    y: np.ndarray
    y_stim: np.ndarray | None
    trial: np.ndarray
    flash: Any
    sfreq: float = 250.0
    source_path: str | None = None


@dataclass(slots=True)
class FlashEvent:
    sample: int
    duration: int
    stimulation: int
    label: int | None = None
    trial_id: int | None = None


def load_dataset(path: str | Path, sfreq: float = 250.0) -> EEGDataset:
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()

    if suffix == ".mat":
        raw = _load_mat(dataset_path)
    elif suffix == ".npz":
        raw = dict(np.load(dataset_path, allow_pickle=True))
    elif suffix in {".pkl", ".pickle"}:
        import pickle

        with dataset_path.open("rb") as handle:
            raw = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported dataset extension: {suffix}")

    data = raw.get("data", raw)
    if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
        data = data.item()

    X = _extract_field(data, "X")
    y = _extract_field(data, "y")
    y_stim = _extract_optional_field(data, "y_stim")
    trial = _extract_field(data, "trial")
    flash = _extract_field(data, "flash")

    X = ensure_channel_first(np.asarray(X, dtype=float))
    y = np.asarray(y).squeeze()
    trial = np.asarray(trial).squeeze().astype(int)
    y_stim = None if y_stim is None else np.asarray(y_stim).squeeze()

    return EEGDataset(X=X, y=y, y_stim=y_stim, trial=trial, flash=flash, sfreq=sfreq, source_path=str(dataset_path))


def ensure_channel_first(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"EEG matrix must be 2D, got shape {X.shape}")
    if X.shape[0] <= 16 and X.shape[1] > X.shape[0]:
        return X
    if X.shape[1] <= 16 and X.shape[0] > X.shape[1]:
        return X.T
    if X.shape[0] <= X.shape[1]:
        return X
    return X.T


def extract_flash_events(dataset: EEGDataset) -> list[FlashEvent]:
    flash = dataset.flash
    flash_array = np.asarray(flash, dtype=object if _looks_structured(flash) else None)
    events: list[FlashEvent] = []

    if isinstance(flash, np.ndarray) and flash.dtype.names:
        rows = flash.reshape(-1)
        for row in rows:
            events.append(
                FlashEvent(
                    sample=int(_get_struct_value(row, ["sample", "start", "onset", 0])),
                    duration=int(_get_struct_value(row, ["duration", 1], default=0)),
                    stimulation=int(_get_struct_value(row, ["stimulation", "stim", "stimulus", 2], default=0)),
                    label=_normalize_label(_get_struct_value(row, ["hit", "label", "y", 3], default=None)),
                )
            )
        return events

    flash_array = np.asarray(flash_array)
    if flash_array.ndim == 1 and flash_array.size and isinstance(flash_array[0], (list, tuple, np.ndarray)):
        flash_array = np.stack([np.asarray(row) for row in flash_array])

    flash_array = np.atleast_2d(flash_array)
    for row in flash_array:
        row_values = list(np.ravel(row))
        if len(row_values) < 3:
            continue
        sample = int(row_values[0])
        duration = int(row_values[1]) if len(row_values) > 1 else 0
        stimulation = int(row_values[2])
        label = _normalize_label(row_values[3]) if len(row_values) > 3 else None
        events.append(FlashEvent(sample=sample, duration=duration, stimulation=stimulation, label=label))
    return events


def assign_trials_to_events(events: Iterable[FlashEvent], trial_starts: np.ndarray) -> list[FlashEvent]:
    starts = np.asarray(trial_starts, dtype=int).ravel()
    if starts.size == 0:
        return list(events)

    assigned: list[FlashEvent] = []
    for event in events:
        trial_id = int(np.searchsorted(starts, event.sample, side="right") - 1)
        trial_id = max(0, min(trial_id, starts.size - 1))
        assigned.append(
            FlashEvent(
                sample=event.sample,
                duration=event.duration,
                stimulation=event.stimulation,
                label=event.label,
                trial_id=trial_id,
            )
        )
    return assigned


def split_trials(n_trials: int, training_trials: int = 15) -> tuple[np.ndarray, np.ndarray]:
    training_trials = min(training_trials, n_trials)
    train_indices = np.arange(training_trials)
    test_indices = np.arange(training_trials, n_trials)
    return train_indices, test_indices


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        return sio.loadmat(path, squeeze_me=True, simplify_cells=True)
    except TypeError:
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _extract_field(data: Any, key: str) -> Any:
    value = _extract_optional_field(data, key)
    if value is None:
        raise KeyError(f"Missing required field '{key}' in dataset")
    return value


def _extract_optional_field(data: Any, key: str) -> Any:
    if isinstance(data, dict) and key in data:
        return data[key]
    if hasattr(data, key):
        return getattr(data, key)
    if isinstance(data, np.ndarray) and data.dtype.names and key in data.dtype.names:
        return data[key]
    return None


def _normalize_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"hit", "1", "true", "yes"}:
            return 1
        if lowered in {"nohit", "0", "false", "no"}:
            return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _looks_structured(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype.names is not None


def _get_struct_value(row: Any, keys: list[Any], default: Any = None) -> Any:
    if isinstance(row, np.void) and row.dtype.names:
        for key in keys:
            if isinstance(key, str) and key in row.dtype.names:
                return row[key]
        if isinstance(keys[-1], int):
            try:
                return row[keys[-1]]
            except Exception:
                return default
    if isinstance(row, dict):
        for key in keys:
            if isinstance(key, str) and key in row:
                return row[key]
    if isinstance(row, (list, tuple, np.ndarray)):
        for key in keys:
            if isinstance(key, int) and key < len(row):
                return row[key]
    return default
