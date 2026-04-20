from __future__ import annotations

import glob
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_bci.classical_pipeline import aggregate_by_model, build_classical_models, evaluate_subject_models
from eeg_bci.config import BCIConfig
from eeg_bci.standalone_data import prepare_p300_file
from eeg_bci.standalone_metrics import json_safe

warnings.filterwarnings("ignore")

DATA_FOLDER = "/scratch/b24cm1027/P300/P300S*.mat"
OUTPUT_DIR = "/csehome/b24cm1027/PRML/outputs/classical_aligned"
TRAIN_TRIALS = 12
VAL_TRIALS = 3
FLASHES_PER_TRIAL = 120
DOWNSAMPLE_FACTOR = 1
RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_seed()
    files = sorted(glob.glob(DATA_FOLDER))
    if not files:
        print(f"No files found in {DATA_FOLDER}!")
        sys.exit(1)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = BCIConfig(training_trials=TRAIN_TRIALS + VAL_TRIALS, random_state=RANDOM_STATE)
    models = build_classical_models(random_state=RANDOM_STATE)
    all_results = {name: {"flash_level": [], "speller_level": []} for name in models}
    per_subject = []

    print(f"Found {len(files)} files to process.")
    print(
        "Config: "
        f"notch={config.notch_freq}Hz  "
        f"bandpass={config.bandpass_low}-{config.bandpass_high}Hz  "
        f"epoch={config.epoch_tmin}-{config.epoch_tmax}s  "
        f"train_trials={TRAIN_TRIALS}  val_trials={VAL_TRIALS}"
    )

    for filepath in tqdm(files, desc="Subjects"):
        prepared = prepare_p300_file(
            filepath,
            config=config,
            train_trials=TRAIN_TRIALS,
            val_trials=VAL_TRIALS,
            flashes_per_trial=FLASHES_PER_TRIAL,
            downsample_factor=DOWNSAMPLE_FACTOR,
        )
        subject_results = evaluate_subject_models(prepared, models, config)

        for model_name, results in subject_results.items():
            all_results[model_name]["flash_level"].append(results["flash_level"])
            all_results[model_name]["speller_level"].append(results["speller_level"])

        per_subject.append(
            {
                "subject": prepared.subject,
                "results": subject_results,
                "primary_model": "LDA",
            }
        )

    print("AVERAGED RESULTS ACROSS SUBJECTS:\n")
    for model_name in models:
        print(f"Model: {model_name}")
        _print_metric_rows(all_results[model_name]["flash_level"])
        _print_metric_rows(all_results[model_name]["speller_level"])
        print()

    output_data = {
        "pipeline": "classical",
        "models": list(models.keys()),
        "aggregate_by_model": aggregate_by_model(all_results),
        "per_subject": per_subject,
    }
    output_file = output_dir / "summary_classical.json"
    output_file.write_text(json.dumps(json_safe(output_data), indent=2), encoding="utf-8")

    print(f"\nResults saved to {output_file}")
    print(f"Total subjects: {len(per_subject)}")


def _print_metric_rows(rows: list[dict[str, float | None]]) -> None:
    if not rows:
        return
    for key in rows[0].keys():
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    main()
