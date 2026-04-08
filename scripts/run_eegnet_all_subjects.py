from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

from eeg_bci.config import BCIConfig
from eeg_bci.eegnet_pipeline import EEGNetConfig, train_eegnet_and_save


def main() -> None:
    files = sorted(glob.glob('/scratch/b24cm1027/P300/P300S*.mat'))
    base_out = Path('/csehome/b24cm1027/PRML/outputs/eegnet_all_subjects_gpu')
    base_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for fp in files:
        sid = Path(fp).stem
        out_dir = base_out / sid
        result = train_eegnet_and_save(
            data_path=fp,
            model_dir=out_dir,
            config=BCIConfig(training_trials=15),
            eegnet_config=EEGNetConfig(epochs=20, batch_size=64, learning_rate=1e-3, patience=7),
        )
        m = result['metrics']
        rows.append(
            {
                'subject': sid,
                'accuracy': m['accuracy'],
                'f1': m['f1'],
                'roc_auc': m['roc_auc'],
                'decoded_words': ' '.join(result['decoded_words']),
                'device': result.get('device', 'unknown'),
            }
        )
        print(f"{sid}: acc={m['accuracy']:.4f}, f1={m['f1']:.4f}, auc={m['roc_auc']:.4f}, words={rows[-1]['decoded_words']}")

    agg = {
        'subjects': len(rows),
        'mean_accuracy': statistics.mean([r['accuracy'] for r in rows]),
        'mean_f1': statistics.mean([r['f1'] for r in rows]),
        'mean_roc_auc': statistics.mean([r['roc_auc'] for r in rows]),
        'best_accuracy': max(r['accuracy'] for r in rows),
        'worst_accuracy': min(r['accuracy'] for r in rows),
    }

    summary = {'per_subject': rows, 'aggregate': agg}
    summary_path = base_out / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == '__main__':
    main()
