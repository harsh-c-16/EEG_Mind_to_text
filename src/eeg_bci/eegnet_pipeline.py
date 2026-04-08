from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from .config import BCIConfig
from .dataset import assign_trials_to_events, extract_flash_events, load_dataset, split_trials
from .decoding import decode_trial_letters, decoded_trials_to_words
from .preprocessing import extract_epochs, preprocess_eeg


def _torch_imports():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise ModuleNotFoundError(
            "PyTorch is required for EEGNet. Install with: pip install torch"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


@dataclass(slots=True)
class EEGNetConfig:
    dropout: float = 0.25
    kernel_length: int = 64
    F1: int = 8
    D: int = 2
    F2: int = 16
    batch_size: int = 64
    epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    random_state: int = 42


class EEGNetBinary:
    def __init__(self, n_channels: int, n_times: int, config: EEGNetConfig) -> None:
        torch, nn, _, _ = _torch_imports()
        self._torch = torch
        self._nn = nn
        self.config = config
        self.model = self._build_model(n_channels=n_channels, n_times=n_times)

    def _build_model(self, n_channels: int, n_times: int):
        torch = self._torch
        nn = self._nn
        cfg = self.config

        class _EEGNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(1, cfg.F1, kernel_size=(1, cfg.kernel_length), padding=(0, cfg.kernel_length // 2), bias=False),
                    nn.BatchNorm2d(cfg.F1),
                    nn.Conv2d(cfg.F1, cfg.F1 * cfg.D, kernel_size=(n_channels, 1), groups=cfg.F1, bias=False),
                    nn.BatchNorm2d(cfg.F1 * cfg.D),
                    nn.ELU(),
                    nn.AvgPool2d((1, 4)),
                    nn.Dropout(cfg.dropout),
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(
                        cfg.F1 * cfg.D,
                        cfg.F1 * cfg.D,
                        kernel_size=(1, 16),
                        padding=(0, 8),
                        groups=cfg.F1 * cfg.D,
                        bias=False,
                    ),
                    nn.Conv2d(cfg.F1 * cfg.D, cfg.F2, kernel_size=(1, 1), bias=False),
                    nn.BatchNorm2d(cfg.F2),
                    nn.ELU(),
                    nn.AvgPool2d((1, 8)),
                    nn.Dropout(cfg.dropout),
                )

                with torch.no_grad():
                    dummy = torch.zeros(1, 1, n_channels, n_times)
                    feat_dim = self.block2(self.block1(dummy)).reshape(1, -1).shape[1]
                self.classifier = nn.Linear(feat_dim, 1)

            def forward(self, x):
                x = self.block1(x)
                x = self.block2(x)
                x = x.reshape(x.shape[0], -1)
                return self.classifier(x).squeeze(-1)

        return _EEGNet()


def train_eegnet_and_save(
    data_path: str | Path,
    model_dir: str | Path,
    config: BCIConfig | None = None,
    eegnet_config: EEGNetConfig | None = None,
) -> dict[str, Any]:
    torch, nn, DataLoader, TensorDataset = _torch_imports()

    config = config or BCIConfig()
    eegnet_config = eegnet_config or EEGNetConfig()
    torch.manual_seed(eegnet_config.random_state)
    np.random.seed(eegnet_config.random_state)

    dataset = load_dataset(data_path, sfreq=config.sfreq)
    signal = preprocess_eeg(
        dataset.X,
        sfreq=config.sfreq,
        notch_freq=config.notch_freq,
        band_low=config.bandpass_low,
        band_high=config.bandpass_high,
    )

    flash_events = assign_trials_to_events(extract_flash_events(dataset), dataset.trial)
    valid_events = [event for event in flash_events if event.label is not None]
    if not valid_events:
        raise ValueError("No labeled flash events found")

    event_samples = np.asarray([event.sample for event in valid_events], dtype=int)
    event_labels = np.asarray([event.label for event in valid_events], dtype=int)
    event_trials = np.asarray([event.trial_id for event in valid_events], dtype=int)
    event_stimulations = np.asarray([event.stimulation for event in valid_events], dtype=int)

    epochs, valid_indices = extract_epochs(
        signal,
        event_samples,
        sfreq=config.sfreq,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
    )
    if epochs.size == 0:
        raise ValueError("No valid epochs extracted")

    event_labels = event_labels[valid_indices]
    event_trials = event_trials[valid_indices]
    event_stimulations = event_stimulations[valid_indices]

    unique_labels = np.unique(event_labels)
    if unique_labels.size != 2:
        raise ValueError(f"EEGNet binary setup expects 2 classes, got {unique_labels.tolist()}")
    negative_label, positive_label = int(unique_labels[0]), int(unique_labels[-1])
    y_binary = (event_labels == positive_label).astype(np.float32)

    train_trials, test_trials = split_trials(len(np.unique(dataset.trial)), training_trials=config.training_trials)
    train_mask = np.isin(event_trials, train_trials)
    test_mask = np.isin(event_trials, test_trials)
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError("Training or test split is empty")

    X_train = epochs[train_mask]
    y_train = y_binary[train_mask]
    X_test = epochs[test_mask]
    y_test = y_binary[test_mask]

    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train_t = torch.tensor(X_train[:, None, :, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test[:, None, :, :], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eegnet = EEGNetBinary(n_channels=X_train.shape[1], n_times=X_train.shape[2], config=eegnet_config)
    model = eegnet.model.to(device)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=eegnet_config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    pos = float(np.sum(y_train == 1.0))
    neg = float(np.sum(y_train == 0.0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=eegnet_config.learning_rate, weight_decay=eegnet_config.weight_decay)

    best_state = None
    best_loss = float("inf")
    no_improve = 0
    train_losses: list[float] = []
    for _ in range(eegnet_config.epochs):
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        train_losses.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= eegnet_config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t.to(device)).detach().cpu().numpy()
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_test, -40.0, 40.0)))
    y_pred_bin = (probabilities >= 0.5).astype(int)
    y_test_bin = y_test.astype(int)

    y_true_original = np.where(y_test_bin == 1, positive_label, negative_label)
    y_pred_original = np.where(y_pred_bin == 1, positive_label, negative_label)

    metrics = {
        "accuracy": float(accuracy_score(y_true_original, y_pred_original)),
        "f1": float(f1_score(y_test_bin, y_pred_bin, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_original, y_pred_original).tolist(),
        "classification_report": classification_report(y_true_original, y_pred_original, output_dict=True, zero_division=0),
        "roc_auc": float(roc_auc_score(y_test_bin, probabilities)),
    }

    decoded_trials = decode_trial_letters(
        trial_ids=event_trials[test_mask],
        stimulations=event_stimulations[test_mask],
        probabilities=probabilities,
        speller_matrix=config.speller_matrix,
    )
    decoded_words = decoded_trials_to_words(decoded_trials, letters_per_word=config.letters_per_word)

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "eegnet_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_channels": int(X_train.shape[1]),
            "n_times": int(X_train.shape[2]),
            "bci_config": asdict(config),
            "eegnet_config": asdict(eegnet_config),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "negative_label": negative_label,
            "positive_label": positive_label,
            "train_losses": train_losses,
            "metrics": metrics,
            "train_trials": train_trials.tolist(),
            "test_trials": test_trials.tolist(),
        },
        model_path,
    )

    summary = {
        "metrics": metrics,
        "decoded_words": decoded_words,
        "model_path": str(model_path),
        "device": str(device),
    }
    (output_dir / "training_summary_eegnet.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eegnet-p300", description="EEGNet pipeline for P300 decoding")
    parser.add_argument("--data", required=True, help="Path to input dataset (.mat/.npz/.pkl)")
    parser.add_argument("--model-dir", required=True, help="Output directory for EEGNet artifacts")
    parser.add_argument("--training-trials", type=int, default=15, help="Number of training trials")
    parser.add_argument("--epochs", type=int, default=35, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bci_config = BCIConfig(training_trials=args.training_trials)
    eegnet_config = EEGNetConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )
    result = train_eegnet_and_save(args.data, args.model_dir, config=bci_config, eegnet_config=eegnet_config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
