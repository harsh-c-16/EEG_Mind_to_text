"""Standalone runner: SVM with linear kernel on P300 EEG data.

Runs only SVM (Linear) across all subjects and prints results.
Uses the shared pipeline in src/eeg_bci.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_bci.classical import build_svm_linear
from eeg_bci.classical_pipeline import aggregate_by_model, evaluate_subject_models
from eeg_bci.config import BCIConfig
from eeg_bci.standalone_data import prepare_p300_file
from eeg_bci.standalone_metrics import json_safe

warnings.filterwarnings("ignore")

DATA_FOLDER = "/scratch/b24cm1027/P300/P300S*.mat"
OUTPUT_DIR = "/csehome/b24cm1027/PRML/outputs/classical_svm_linear"
TRAIN_TRIALS = 12
VAL_TRIALS = 3
FLASHES_PER_TRIAL = 120
RANDOM_STATE = 42


def main() -> None:
    files = sorted(glob.glob(DATA_FOLDER))
    if not files:
        print(f"No files found in {DATA_FOLDER}!")
        sys.exit(1)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BCIConfig(training_trials=TRAIN_TRIALS + VAL_TRIALS, random_state=RANDOM_STATE)
    models = {"SVM (Linear)": build_svm_linear(random_state=RANDOM_STATE)}
    all_results = {"SVM (Linear)": {"flash_level": [], "speller_level": []}}
    per_subject = []

    print(f"Running SVM (Linear) on {len(files)} subjects...")

    for filepath in tqdm(files, desc="Subjects"):
        prepared = prepare_p300_file(
            filepath,
            config=config,
            train_trials=TRAIN_TRIALS,
            val_trials=VAL_TRIALS,
            flashes_per_trial=FLASHES_PER_TRIAL,
        )
        subject_results = evaluate_subject_models(prepared, models, config)

        all_results["SVM (Linear)"]["flash_level"].append(subject_results["SVM (Linear)"]["flash_level"])
        all_results["SVM (Linear)"]["speller_level"].append(subject_results["SVM (Linear)"]["speller_level"])

        fl = subject_results["SVM (Linear)"]["flash_level"]
        print(
            f"{prepared.subject}: "
            f"BA={fl['balanced_accuracy']:.4f}, "
            f"F2={fl['f2']:.4f}"
        )
        per_subject.append({"subject": prepared.subject, "results": subject_results})

    output_data = {
        "pipeline": "classical_svm_linear",
        "aggregate_by_model": aggregate_by_model(all_results),
        "per_subject": per_subject,
    }
    output_file = output_dir / "summary_svm_linear.json"
    output_file.write_text(json.dumps(json_safe(output_data), indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
