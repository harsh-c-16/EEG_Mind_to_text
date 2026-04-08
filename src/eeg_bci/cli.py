from __future__ import annotations

import argparse
import json

from .config import BCIConfig, FeatureConfig, ModelConfig
from .inference import predict_words
from .training import train_and_save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eeg-bci", description="P300 EEG mind-to-text pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the flash classifier")
    train_parser.add_argument("--data", required=True, help="Path to the P300 dataset file")
    train_parser.add_argument("--model-dir", required=True, help="Directory to store trained artifacts")
    train_parser.add_argument("--classifier", default="svm", choices=["svm", "rf", "logreg"], help="Classifier backend")
    train_parser.add_argument("--use-csp", action="store_true", help="Enable CSP features")
    train_parser.add_argument("--no-psd", action="store_true", help="Disable PSD features")
    train_parser.add_argument("--training-trials", type=int, default=15, help="Number of training trials")

    eval_parser = subparsers.add_parser("evaluate", help="Train and print metrics for the configured split")
    eval_parser.add_argument("--data", required=True, help="Path to the P300 dataset file")
    eval_parser.add_argument("--model-dir", required=True, help="Directory to store trained artifacts")
    eval_parser.add_argument("--classifier", default="svm", choices=["svm", "rf", "logreg"], help="Classifier backend")
    eval_parser.add_argument("--use-csp", action="store_true", help="Enable CSP features")
    eval_parser.add_argument("--no-psd", action="store_true", help="Disable PSD features")
    eval_parser.add_argument("--training-trials", type=int, default=15, help="Number of training trials")

    predict_parser = subparsers.add_parser("predict", help="Decode test-set words using a saved model")
    predict_parser.add_argument("--data", required=True, help="Path to the P300 dataset file")
    predict_parser.add_argument("--model-dir", required=True, help="Directory containing saved artifacts")
    predict_parser.add_argument("--training-trials", type=int, default=15, help="Number of training trials")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = BCIConfig(training_trials=args.training_trials)
    feature_config = FeatureConfig(use_psd=not getattr(args, "no_psd", False), use_csp=getattr(args, "use_csp", False))
    model_config = ModelConfig(classifier=getattr(args, "classifier", "svm"))

    if args.command in {"train", "evaluate"}:
        result = train_and_save(
            data_path=args.data,
            model_dir=args.model_dir,
            config=config,
            feature_config=feature_config,
            model_config=model_config,
        )
        print(json.dumps({"metrics": result.metrics, "decoded_words": result.decoded_words, "artifacts_path": str(result.artifacts_path)}, indent=2))
        return

    if args.command == "predict":
        words = predict_words(args.data, args.model_dir, config=config)
        print(json.dumps({"decoded_words": words}, indent=2))
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
