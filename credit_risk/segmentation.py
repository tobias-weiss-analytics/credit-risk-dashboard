"""Customer segmentation with k-means and a 2-D PCA projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data import ORDINAL_FEATURES


def encode_for_clustering(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Numeric features as they are; ordinal features as their rank on the scale.

    Categories outside the scale ("no checking account", "unknown savings") get the
    middle of the scale so they do not dominate the distance computation.
    """
    out = pd.DataFrame(index=X.index)
    for col in features:
        if col in ORDINAL_FEATURES:
            order, _ = ORDINAL_FEATURES[col]
            out[col] = X[col].map({c: i for i, c in enumerate(order)}).astype(float).fillna((len(order) - 1) / 2)
        else:
            out[col] = X[col].astype(float)
    return out


@dataclass
class Segmentation:
    features: list[str]
    scaler: StandardScaler
    kmeans: KMeans
    labels: np.ndarray
    coords: np.ndarray
    names: dict[int, str]

    def assign(self, X: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(self.scaler.transform(encode_for_clustering(X, self.features)))


def describe_clusters(centroids_z: np.ndarray, features: list[str]) -> dict[int, str]:
    """Name each cluster after its two most distinctive features (centroid z-scores)."""
    names = {}
    for i, centre in enumerate(centroids_z):
        top = np.argsort(-np.abs(centre))[:2]
        parts = [f"{'High' if centre[j] > 0 else 'Low'} {features[j].replace('_', ' ').lower()}" for j in top]
        names[i] = " · ".join(parts)
    # Guarantee unique names.
    seen: dict[str, int] = {}
    for i, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n} ({seen[n]})"
    return names


def segment(X: pd.DataFrame, features: list[str], k: int, random_state: int = 42) -> Segmentation:
    if not features:
        raise ValueError("Select at least one feature for the segmentation.")
    data = encode_for_clustering(X, features)
    scaler = StandardScaler().fit(data)
    z = scaler.transform(data)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(z)
    if z.shape[1] >= 2:
        coords = PCA(n_components=2, random_state=random_state).fit_transform(z)
    else:
        coords = np.column_stack([z[:, 0], np.zeros(len(z))])
    return Segmentation(features, scaler, kmeans, kmeans.labels_, coords,
                        describe_clusters(kmeans.cluster_centers_, features))
