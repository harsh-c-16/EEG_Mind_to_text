from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch

from .config import FeatureConfig


@dataclass
class CSPTransformer:
    n_components: int = 4
    reg: float = 1e-6
    filters_: np.ndarray | None = field(default=None, init=False)

    def fit(self, epochs: np.ndarray, y: np.ndarray) -> "CSPTransformer":
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("CSPTransformer requires binary labels")

        covariances = []
        for cls in classes:
            class_epochs = epochs[y == cls]
            if class_epochs.size == 0:
                raise ValueError(f"No epochs for class {cls}")
            class_covs = [_normalized_cov(epoch) for epoch in class_epochs]
            covariances.append(np.mean(class_covs, axis=0))

        composite = covariances[0] + covariances[1] + self.reg * np.eye(covariances[0].shape[0])
        eigenvalues, eigenvectors = eigh(covariances[0], composite)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        n_pairs = max(1, self.n_components // 2)
        selected = np.concatenate([eigenvectors[:, :n_pairs], eigenvectors[:, -n_pairs:]], axis=1)
        self.filters_ = selected.T
        return self

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        if self.filters_ is None:
            raise RuntimeError("CSPTransformer has not been fitted")

        transformed = []
        for epoch in epochs:
            projected = self.filters_ @ epoch
            variances = np.var(projected, axis=1)
            variances = np.maximum(variances, 1e-12)
            transformed.append(np.log(variances / variances.sum()))
        return np.asarray(transformed)


@dataclass
class FeatureExtractor:
    config: FeatureConfig = field(default_factory=FeatureConfig)
    csp: CSPTransformer | None = field(default=None, init=False)

    def fit(self, epochs: np.ndarray, y: np.ndarray) -> "FeatureExtractor":
        if self.config.use_csp:
            self.csp = CSPTransformer(n_components=self.config.csp_components).fit(epochs, y)
        return self

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        features = []
        if self.config.use_psd:
            features.append(self._psd_features(epochs))
        if self.config.use_csp:
            if self.csp is None:
                raise RuntimeError("CSP features requested before fitting")
            features.append(self.csp.transform(epochs))
        if not features:
            raise ValueError("FeatureExtractor needs at least one enabled feature family")
        return np.concatenate(features, axis=1) if len(features) > 1 else features[0]

    def fit_transform(self, epochs: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(epochs, y).transform(epochs)

    def _psd_features(self, epochs: np.ndarray) -> np.ndarray:
        band_features = []
        for epoch in epochs:
            channel_features = []
            for channel in epoch:
                freqs, psd = welch(channel, fs=250.0, nperseg=min(128, channel.shape[-1]))
                for low, high in self.config.bands:
                    mask = (freqs >= low) & (freqs < high)
                    power = np.trapezoid(psd[mask], freqs[mask]) if np.any(mask) else 0.0
                    channel_features.append(np.log1p(power))
            band_features.append(channel_features)
        return np.asarray(band_features)


def _normalized_cov(epoch: np.ndarray) -> np.ndarray:
    centered = epoch - epoch.mean(axis=1, keepdims=True)
    covariance = centered @ centered.T
    covariance /= max(centered.shape[1] - 1, 1)
    trace = np.trace(covariance)
    if trace <= 0:
        trace = 1.0
    return covariance / trace
