import numpy as np
from config import DATA


def compute_long_term_ssp(data, train_years):
    """Compute the long-term mean SSP at each source grid point over all training years and months."""
    result = np.zeros((data.cfg.points, data.cfg.ssp_levels), dtype=np.float64)
    count = 0
    for year in train_years:
        for month in range(1, data.cfg.months + 1):
            points = np.arange(data.cfg.points)
            years = np.full(data.cfg.points, year)
            months = np.full(data.cfg.points, month)
            result += data.generate_ssp(years, months, points)
            count += 1
    return (result / count).astype(np.float32)


def eof_features(long_term_ssp, variance_threshold=0.99):
    """Perform EOF decomposition via SVD and retain components up to the specified cumulative variance threshold."""
    mean_profile = long_term_ssp.mean(axis=0, keepdims=True)
    centered = long_term_ssp - mean_profile
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s ** 2
    explained = variance / max(variance.sum(), 1e-12)
    cumulative = np.cumsum(explained)
    k = int(np.searchsorted(cumulative, variance_threshold) + 1)
    scores = u[:, :k] * s[:k]
    return {
        "mean": mean_profile.squeeze(0).astype(np.float32),
        "modes": vt[:k].astype(np.float32),
        "scores": scores.astype(np.float32),
        "explained_ratio": explained[:k].astype(np.float32),
        "k": k,
    }


def zscore_features(features):
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-8
    return ((features - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def kmeans(features, n_clusters=4, seed=42, max_iter=300, tol=1e-6):
    """Lightweight K-means implementation that avoids an additional scikit-learn dependency."""
    rng = np.random.default_rng(seed)
    n = len(features)
    if not 1 <= n_clusters <= n:
        raise ValueError("n_clusters must be between 1 and the number of samples")

    centers = np.empty((n_clusters, features.shape[1]), dtype=np.float32)
    centers[0] = features[rng.integers(n)]
    min_dist2 = ((features - centers[0]) ** 2).sum(axis=1)
    for c in range(1, n_clusters):
        probs = min_dist2 / max(min_dist2.sum(), 1e-12)
        centers[c] = features[rng.choice(n, p=probs)]
        d2 = ((features - centers[c]) ** 2).sum(axis=1)
        min_dist2 = np.minimum(min_dist2, d2)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist2 = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist2.argmin(axis=1)
        new_centers = centers.copy()
        for c in range(n_clusters):
            members = features[new_labels == c]
            if len(members):
                new_centers[c] = members.mean(axis=0)
        shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        labels, centers = new_labels, new_centers
        if shift < tol:
            break
    return labels, centers.astype(np.float32)


def build_eofk_partition(data, train_years, n_clusters=4, variance_threshold=0.99, seed=42):
    long_term_ssp = compute_long_term_ssp(data, train_years)
    eof = eof_features(long_term_ssp, variance_threshold)
    normalized, score_mean, score_std = zscore_features(eof["scores"])
    labels, centers = kmeans(normalized, n_clusters=n_clusters, seed=seed)
    eof.update({
        "long_term_ssp": long_term_ssp,
        "normalized_scores": normalized,
        "score_mean": score_mean,
        "score_std": score_std,
        "labels": labels,
        "centers": centers,
    })
    return eof


def haversine_distance(lon1, lat1, lon2, lat2):
    """Compute great-circle distance in kilometers."""
    radius = 6371.0088
    lon1, lat1 = np.deg2rad(lon1), np.deg2rad(lat1)
    lon2, lat2 = np.deg2rad(lon2), np.deg2rad(lat2)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def location_to_cluster(lon, lat, grid_lonlat, labels):
    """Map an arbitrary source location to the nearest grid point and return its EOF-K subregion label."""
    distances = haversine_distance(lon, lat, grid_lonlat[:, 0], grid_lonlat[:, 1])
    nearest = int(np.argmin(distances))
    return int(labels[nearest])


def fit_training_statistics_and_background(data, train_indices, labels, n_clusters, batch_size=32):
    """Compute TL normalization statistics and EOF-K geo-steady fields in one pass over the training set."""
    sums = np.zeros((n_clusters, DATA.channels, DATA.depth_bins, DATA.range_bins), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    scalar_sum = 0.0
    scalar_sumsq = 0.0
    scalar_count = 0

    for start in range(0, len(train_indices), batch_size):
        batch_idx = train_indices[start:start + batch_size]
        fields = data.generate_tl(batch_idx)
        _, _, point = data.decode_indices(batch_idx)
        batch_labels = labels[point]

        scalar_sum += float(fields.sum(dtype=np.float64))
        scalar_sumsq += float(np.square(fields, dtype=np.float64).sum(dtype=np.float64))
        scalar_count += fields.size

        for c in range(n_clusters):
            mask = batch_labels == c
            if np.any(mask):
                sums[c] += fields[mask].sum(axis=0, dtype=np.float64)
                counts[c] += int(mask.sum())

    if np.any(counts == 0):
        raise RuntimeError("At least one EOF-K subregion contains no training samples. Reduce the number of clusters or inspect the input features.")

    geo_steady = (sums / counts[:, None, None, None]).astype(np.float32)
    mean = scalar_sum / scalar_count
    var = max(scalar_sumsq / scalar_count - mean ** 2, 1e-12)
    std = float(np.sqrt(var))
    return geo_steady, counts, np.float32(mean), np.float32(std)
