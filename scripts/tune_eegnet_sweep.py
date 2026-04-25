from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
from pathlib import Path

from scipy.io import loadmat

from eeg_bci.config import BCIConfig
from eeg_bci.eegnet_pipeline import EEGNetConfig, train_eegnet_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep EEGNet hyperparameters across all P300 subjects")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), help="Directory containing P300S*.mat files (defaults to DATA_DIR env var)")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR"), help="Sweep output directory (defaults to OUTPUT_DIR env var)")
    parser.add_argument("--epochs", default="20,30", help="Comma-separated epochs list")
    parser.add_argument("--batch-sizes", default="32,64", help="Comma-separated batch size list")
    parser.add_argument("--lrs", default="0.001,0.0005", help="Comma-separated learning rates")
    parser.add_argument("--dropouts", default="0.25,0.4", help="Comma-separated dropout values")
    parser.add_argument("--f1s", default="8,16", help="Comma-separated F1 filter counts")
    parser.add_argument("--ds", default="2", help="Comma-separated depth multipliers D")
    parser.add_argument("--f2s", default="16,32", help="Comma-separated F2 pointwise filter counts")
    parser.add_argument("--kernel-lengths", default="32,64", help="Comma-separated kernel lengths")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--max-combos", type=int, default=12, help="Maximum number of hyperparameter combos to run")
    return parser.parse_args()


def _int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _targets_from_word_field(mat_file: str) -> list[str]:
    data = loadmat(mat_file, squeeze_me=True, simplify_cells=True)
    word = str(data.get("Word", "")).strip()
    if not word:
        return []
    chunks = [word[i : i + 5] for i in range(0, len(word), 5)]
    return chunks[3:7] if len(chunks) >= 7 else chunks


def _word_accuracy(decoded_words: list[str], target_words: list[str]) -> float:
    if not target_words:
        return 0.0
    n = min(len(decoded_words), len(target_words))
    if n == 0:
        return 0.0
    return sum(int(decoded_words[i] == target_words[i]) for i in range(n)) / float(n)


def main() -> None:
    args = parse_args()
    if not args.data_dir:
        raise EnvironmentError("--data-dir must be provided or DATA_DIR environment variable must be set.")
    if not args.output_dir:
        raise EnvironmentError("--output-dir must be provided or OUTPUT_DIR environment variable must be set.")
    data_files = sorted(glob.glob(str(Path(args.data_dir) / "P300S*.mat")))
    if not data_files:
        raise FileNotFoundError(f"No files matched {Path(args.data_dir) / 'P300S*.mat'}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = list(
        itertools.product(
            _int_list(args.epochs),
            _int_list(args.batch_sizes),
            _float_list(args.lrs),
            _float_list(args.dropouts),
            _int_list(args.f1s),
            _int_list(args.ds),
            _int_list(args.f2s),
            _int_list(args.kernel_lengths),
        )
    )
    search_space = search_space[: max(1, args.max_combos)]

    all_runs: list[dict] = []
    best_run: dict | None = None

    for run_idx, (epochs, batch_size, lr, dropout, f1, d, f2, kernel_length) in enumerate(search_space, start=1):
        run_name = f"run_{run_idx:03d}_ep{epochs}_bs{batch_size}_lr{lr}_do{dropout}_f1{f1}_d{d}_f2{f2}_k{kernel_length}"
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        per_subject = []
        for data_file in data_files:
            subject = Path(data_file).stem
            subject_dir = run_dir / subject
            result = train_eegnet_and_save(
                data_path=data_file,
                model_dir=subject_dir,
                config=BCIConfig(training_trials=15),
                eegnet_config=EEGNetConfig(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    dropout=dropout,
                    F1=f1,
                    D=d,
                    F2=f2,
                    kernel_length=kernel_length,
                    patience=args.patience,
                ),
            )
            target_words = _targets_from_word_field(data_file)
            acc_words = _word_accuracy(result["decoded_words"], target_words)
            per_subject.append(
                {
                    "subject": subject,
                    "accuracy": result["metrics"]["accuracy"],
                    "f1": result["metrics"]["f1"],
                    "roc_auc": result["metrics"]["roc_auc"],
                    "word_accuracy": acc_words,
                    "decoded_words": result["decoded_words"],
                    "target_words": target_words,
                    "model_path": result["model_path"],
                    "device": result["device"],
                }
            )

        aggregate = {
            "subjects": len(per_subject),
            "mean_accuracy": statistics.mean([x["accuracy"] for x in per_subject]),
            "mean_f1": statistics.mean([x["f1"] for x in per_subject]),
            "mean_roc_auc": statistics.mean([x["roc_auc"] for x in per_subject]),
            "mean_word_accuracy": statistics.mean([x["word_accuracy"] for x in per_subject]),
        }

        run_summary = {
            "run_name": run_name,
            "params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "dropout": dropout,
                "F1": f1,
                "D": d,
                "F2": f2,
                "kernel_length": kernel_length,
                "patience": args.patience,
            },
            "aggregate": aggregate,
            "per_subject": per_subject,
        }

        (run_dir / "summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        all_runs.append(run_summary)

        if best_run is None:
            best_run = run_summary
        else:
            cur = run_summary["aggregate"]
            best = best_run["aggregate"]
            better = (cur["mean_word_accuracy"], cur["mean_roc_auc"], cur["mean_f1"]) > (
                best["mean_word_accuracy"],
                best["mean_roc_auc"],
                best["mean_f1"],
            )
            if better:
                best_run = run_summary

        print(
            f"[{run_idx}/{len(search_space)}] {run_name} -> "
            f"word_acc={aggregate['mean_word_accuracy']:.4f}, "
            f"auc={aggregate['mean_roc_auc']:.4f}, "
            f"f1={aggregate['mean_f1']:.4f}, "
            f"acc={aggregate['mean_accuracy']:.4f}"
        )

    overall = {
        "num_runs": len(all_runs),
        "best_run": best_run,
        "all_runs": [
            {
                "run_name": x["run_name"],
                "params": x["params"],
                "aggregate": x["aggregate"],
            }
            for x in all_runs
        ],
    }
    (output_dir / "sweep_results.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

    if best_run is not None:
        (output_dir / "best_config.json").write_text(
            json.dumps({"run_name": best_run["run_name"], "params": best_run["params"], "aggregate": best_run["aggregate"]}, indent=2),
            encoding="utf-8",
        )

    print("\nBest run:")
    print(json.dumps(best_run, indent=2))
    print(f"\nSaved sweep summary: {output_dir / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
