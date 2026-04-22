from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eeg_bci.config import BCIConfig
from eeg_bci.eegnet_standalone import BestEEGNetConfig, aggregate_eegnet_results, train_eegnet_subject
from eeg_bci.standalone_data import prepare_p300_file
from eeg_bci.standalone_metrics import json_safe

warnings.filterwarnings("ignore")

DATA_DIR = "/scratch/b24cm1027/P300"
OUTPUT_DIR = "/csehome/b24cm1027/PRML/outputs/eegnet_standalone"
THRESHOLD = 0.35
TRAIN_TRIALS = 12
VAL_TRIALS = 3
FLASHES_PER_TRIAL = 120


def main() -> None:
    data_files = sorted(glob.glob(f"{DATA_DIR}/P300S*.mat"))
    if not data_files:
        print(f"ERROR: No data files found in {DATA_DIR}")
        sys.exit(1)

    output_dir = Path(OUTPUT_DIR)
    bci_config = BCIConfig(training_trials=TRAIN_TRIALS + VAL_TRIALS)
    eeg_config = BestEEGNetConfig()

    print("EEGNET PIPELINE")
    print(f"Data files: {len(data_files)}")
    print(f"Output dir: {output_dir}")
    print(f"Threshold: {THRESHOLD}")

    per_subject = []
    for data_path in tqdm(data_files, desc="Subjects"):
        try:
            prepared = prepare_p300_file(
                data_path,
                config=bci_config,
                train_trials=TRAIN_TRIALS,
                val_trials=VAL_TRIALS,
                flashes_per_trial=FLASHES_PER_TRIAL,
            )
            result = train_eegnet_subject(
                prepared=prepared,
                output_dir=output_dir,
                config=eeg_config,
                threshold=THRESHOLD,
            )
            per_subject.append(result)

            flash = result["results"]["flash_level"]
            roc_auc = flash["roc_auc"] if flash["roc_auc"] is not None else float("nan")
            print(
                f"{prepared.subject}: "
                f"BA={flash['balanced_accuracy']:.4f}, "
                f"F2={flash['f2']:.4f}, "
                f"ROC-AUC={roc_auc:.4f}"
            )
        except Exception as exc:
            print(f"{Path(data_path).stem}: {exc}")

    summary = {
        "pipeline": "eegnet_standalone",
        "n_subjects": len(per_subject),
        "threshold": THRESHOLD,
        "aggregate": aggregate_eegnet_results(per_subject),
        "per_subject": per_subject,
    }

    print("AGGREGATED RESULTS")
    for key, stats in summary["aggregate"].items():
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is not None:
            print(f"{key}: {mean:.4f} +/- {std:.4f}")

    summary_file = output_dir / "summary_eegnet_standalone.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()
