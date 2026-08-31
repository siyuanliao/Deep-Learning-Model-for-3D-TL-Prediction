"""EOF-K subregion division and training-only geo-steady TL calculation."""

from dataclasses import dataclass
import numpy as np
from config import DATA, build_source_grid
from synthetic_data import generate_ssp_for_grid, generate_tl, sample_index_to_ymp


@dataclass
class EOFKResult:
    labels: np.ndarray
    eof_mean: np.ndarray
    eof_modes: np.ndarray
    explained_ratio: np.ndarray
    scores_mean: np.ndarray
    scores_std: np.ndarray
    centers: np.ndarray


def compute_training_mean_ssp(train_indices, cfg=DATA):
    """Compute T in R^[121,50] using training years/months only.

    For each source grid point, average its SSP over all training year-month
    combinations represented by train_indices.
    """
    year0, month0, _ = sample_index_to_ymp(train_indices, cfg)
    pairs = np.unique(np.column_stack([year0, month0]), axis=0)

    accum = np.zeros((cfg.points, cfg.ssp_levels), dtype=np.float64)
    for y, m in pairs:
        accum += generate_ssp_for_grid(int(y), int(m), cfg).astype(np.float64)
    return (accum / len(pairs)).astype(np.float32)


def eof_decompose(mean_ssp, variance_threshold=0.99):
    """EOF via centered SVD; retain smallest K with cumulative variance >= threshold."""
    x = np.asarray(mean_ssp, dtype=np.float64)
    eof_mean = x.mean(axis=0, keepdims=True)
    xc = x - eof_mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)

    eig = s ** 2
    explained = eig / np.maximum(eig.sum(), 1e-12)
    cumulative = np.cumsum(explained)
    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    k = min(k, vt.shape[0])

    # P = U Sigma, matching the EOF coefficient/principal-score description.
    scores = u[:, :k] * s[:k]
    modes = vt[:k].T
    return (
        scores.astype(np.float32),
        eof_mean.ravel().astype(np.float32),
        modes.astype(np.float32),
        explained[:k].astype(np.float32),
    )


def _kmeans(features, n_clusters, seed=42, max_iter=300, n_init=20):
    """Small NumPy K-means implementation to avoid adding scikit-learn dependency."""
    x = np.asarray(features, dtype=np.float64)
    if not 1 <= n_clusters <= len(x):
        raise ValueError("n_clusters must be between 1 and number of grid points")

    best_labels = None
    best_centers = None
    best_inertia = np.inf
    master = np.random.default_rng(seed)

    for _ in range(n_init):
        rng = np.random.default_rng(master.integers(0, 2**32 - 1))
        centers = x[rng.choice(len(x), n_clusters, replace=False)].copy()

        for _ in range(max_iter):
            dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist2.argmin(axis=1)
            new_centers = centers.copy()
            for k in range(n_clusters):
                members = x[labels == k]
                if len(members) == 0:
                    new_centers[k] = x[rng.integers(0, len(x))]
                else:
                    new_centers[k] = members.mean(axis=0)
            if np.allclose(new_centers, centers, rtol=0.0, atol=1e-7):
                centers = new_centers
                break
            centers = new_centers

        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist2.argmin(axis=1)
        inertia = dist2[np.arange(len(x)), labels].sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels.astype(np.int64), best_centers.astype(np.float32)


def fit_eof_k(train_indices, n_clusters=4, variance_threshold=0.99, seed=42, cfg=DATA):
    """Fit EOF-K using SSP information from the training split only.

    No water-depth feature is used: clustering features are normalized EOF
    coefficients only because bathymetry is unavailable in this dataset.
    """
    mean_ssp = compute_training_mean_ssp(train_indices, cfg)
    scores, eof_mean, eof_modes, explained = eof_decompose(mean_ssp, variance_threshold)

    score_mean = scores.mean(axis=0)
    score_std = scores.std(axis=0) + 1e-8
    zscores = (scores - score_mean) / score_std
    labels, centers = _kmeans(zscores, n_clusters=n_clusters, seed=seed)

    return EOFKResult(
        labels=labels,
        eof_mean=eof_mean,
        eof_modes=eof_modes,
        explained_ratio=explained,
        scores_mean=score_mean.astype(np.float32),
        scores_std=score_std.astype(np.float32),
        centers=centers,
    )


def haversine_distance_km(lon, lat, grid_lonlat):
    """Great-circle distance from one location to all grid points."""
    lon1 = np.deg2rad(float(lon))
    lat1 = np.deg2rad(float(lat))
    lon2 = np.deg2rad(grid_lonlat[:, 0].astype(np.float64))
    lat2 = np.deg2rad(grid_lonlat[:, 1].astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def location_to_cluster(lon, lat, labels, cfg=DATA):
    """Nearest-grid-point mapping followed by that grid point's EOF-K label."""
    grid = build_source_grid(cfg)
    nearest = int(np.argmin(haversine_distance_km(lon, lat, grid)))
    return int(labels[nearest])


def build_geo_steady_fields(train_indices, labels, n_clusters, batch_size=32, cfg=DATA):
    """Average training TL over years, months and source points in each cluster.

    This is streamed in batches, so the complete 15k+ training TL tensor is
    never stored in memory. Validation/test samples are never used.
    """
    sums = np.zeros((n_clusters, *cfg.field_shape), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    for start in range(0, len(train_indices), batch_size):
        batch_idx = train_indices[start:start + batch_size]
        fields = generate_tl(batch_idx, cfg).astype(np.float64)
        _, _, point0 = sample_index_to_ymp(batch_idx, cfg)
        batch_labels = labels[point0]

        for c in range(n_clusters):
            mask = batch_labels == c
            if np.any(mask):
                sums[c] += fields[mask].sum(axis=0)
                counts[c] += int(mask.sum())

    if np.any(counts == 0):
        raise RuntimeError(f"Empty EOF-K cluster(s): {np.where(counts == 0)[0].tolist()}")

    steady = sums / counts[:, None, None, None]
    return steady.astype(np.float32), counts
