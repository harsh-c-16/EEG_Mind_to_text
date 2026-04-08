from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(slots=True)
class ModelArtifacts:
    model: Any
    feature_extractor: Any
    metadata: dict[str, Any]


def build_classifier(name: str = "svm", random_state: int = 42) -> Any:
    name = name.lower().strip()
    if name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=random_state)),
        ])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
        ])
    raise ValueError(f"Unknown classifier: {name}")


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X)
    probabilities = _positive_class_scores(model, X)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
    }
    if probabilities is not None:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
    return metrics


def save_artifacts(path: str | Path, artifacts: ModelArtifacts) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "model_artifacts.joblib"
    joblib.dump(artifacts, file_path)
    return file_path


def load_artifacts(path: str | Path) -> ModelArtifacts:
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / "model_artifacts.joblib"
    return joblib.load(file_path)


def _positive_class_scores(model: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores)
        return 1.0 / (1.0 + np.exp(-scores))
    return None
