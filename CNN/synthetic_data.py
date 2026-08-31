"""Synthetic SSP and 3-D transmission-loss generation.

The synthetic fields are designed only to make the public GitHub example
self-contained. They mimic several broad acoustic features (geometric
spreading, absorption, refraction-related variability, depth/azimuth
structure, interference-like oscillations and seasonal/interannual changes),
but they are not a replacement for Bellhop/RAM/PE simulations.
"""

import numpy as np
from config import DATA, build_source_grid


SSP_DEPTH_M = np.linspace(0.0, 1000.0, DATA.ssp_levels, dtype=np.float32)
RECEIVER_DEPTH_M = np.linspace(10.0, 1000.0, DATA.receiver_depths, dtype=np.float32)
RANGE_KM = np.linspace(1.0, 250.0, DATA.range_bins, dtype=np.float32)
AZIMUTH_DEG = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float32)


def sample_index_to_ymp(indices, cfg=DATA):
    """Global sample index -> zero-based (year, month, grid-point)."""
    idx = np.asarray(indices, dtype=np.int64)
    per_year = cfg.months * cfg.points
    year0 = idx // per_year
    rem = idx % per_year
    month0 = rem // cfg.points
    point0 = rem % cfg.points
    return year0, month0, point0


def split_indices_by_year(cfg=DATA, train_years=None, val_years=None, test_years=None):
    """Split sample indices by year. Year numbering starts from 1."""
    from config import TRAIN_YEARS, VAL_YEARS, TEST_YEARS

    train_years = TRAIN_YEARS if train_years is None else tuple(train_years)
    val_years = VAL_YEARS if val_years is None else tuple(val_years)
    test_years = TEST_YEARS if test_years is None else tuple(test_years)

    year_ids = np.repeat(np.arange(1, cfg.years + 1), cfg.months * cfg.points)
    train_idx = np.where(np.isin(year_ids, train_years))[0]
    val_idx = np.where(np.isin(year_ids, val_years))[0]
    test_idx = np.where(np.isin(year_ids, test_years))[0]
    return train_idx, val_idx, test_idx


def _ssp_core(lon, lat, month0, year0):
    """Vectorized 50-level SSP generator. Inputs are one-dimensional arrays."""
    lon = np.asarray(lon, dtype=np.float32)[:, None]
    lat = np.asarray(lat, dtype=np.float32)[:, None]
    month0 = np.asarray(month0, dtype=np.float32)[:, None]
    year0 = np.asarray(year0, dtype=np.float32)[:, None]
    z = SSP_DEPTH_M[None, :]

    # Smooth deep-ocean-like baseline: warm surface, thermocline minimum,
    # then pressure-related sound-speed increase at depth.
    baseline = 1485.0 + 35.0 * np.exp(-z / 120.0) + 0.017 * z

    lon_n = (lon - DATA.lon_min) / (DATA.lon_max - DATA.lon_min)
    lat_n = (lat - DATA.lat_min) / (DATA.lat_max - DATA.lat_min)
    season = np.sin(2.0 * np.pi * month0 / 12.0)
    season_q = np.cos(2.0 * np.pi * month0 / 12.0)

    # Geographically varying vertical structures. These are deliberately
    # smooth so EOF can recover meaningful dominant modes.
    mode1 = (2.4 * (lon_n - 0.5) - 1.8 * (lat_n - 0.5)) * np.exp(-z / 260.0)
    mode2 = (1.4 * np.sin(np.pi * lon_n) * np.cos(np.pi * lat_n)) * np.exp(-((z - 350.0) / 220.0) ** 2)
    seasonal = (3.2 * season + 1.1 * season_q * (lat_n - 0.5)) * np.exp(-z / 170.0)
    interannual = 0.35 * np.sin(2.0 * np.pi * year0 / 5.0 + 0.4 * lon_n) * np.exp(-z / 450.0)

    # Deterministic weak fine-scale structure; no random file is required and
    # each sample is exactly reproducible from its index.
    fine = 0.22 * np.sin(0.018 * z + 0.8 * month0 + 0.5 * year0 + 4.0 * lon_n + 3.0 * lat_n)

    return (baseline + mode1 + mode2 + seasonal + interannual + fine).astype(np.float32)


def generate_environment(indices, cfg=DATA):
    """Return [N,52] = lon, lat, 50-level SSP."""
    year0, month0, point0 = sample_index_to_ymp(indices, cfg)
    grid = build_source_grid(cfg)
    lon = grid[point0, 0]
    lat = grid[point0, 1]
    ssp = _ssp_core(lon, lat, month0, year0)
    return np.concatenate([lon[:, None], lat[:, None], ssp], axis=1).astype(np.float32)


def generate_ssp_for_grid(year0, month0, cfg=DATA):
    """Return SSPs for all 121 grid points: [121,50]."""
    grid = build_source_grid(cfg)
    n = cfg.points
    return _ssp_core(
        grid[:, 0], grid[:, 1],
        np.full(n, month0, dtype=np.int64),
        np.full(n, year0, dtype=np.int64),
    )


def generate_tl(indices, cfg=DATA):
    """Generate target TL fields with shape [N,36,4,250] in dB.

    The construction intentionally contains physically recognizable trends:
    20log10(r) spreading, weak absorption, source-environment dependence,
    vertical refraction structure, azimuth anisotropy, convergence/shadow-like
    modulation and interference-like oscillations.
    """
    indices = np.asarray(indices, dtype=np.int64)
    env = generate_environment(indices, cfg)
    year0, month0, _ = sample_index_to_ymp(indices, cfg)

    lon = env[:, 0]
    lat = env[:, 1]
    ssp = env[:, 2:]
    b = len(indices)

    r = RANGE_KM[None, None, None, :]                # [1,1,1,R]
    z = RECEIVER_DEPTH_M[None, :, None, None]        # [1,D,1,1]
    az = np.deg2rad(AZIMUTH_DEG)[None, None, :, None]# [1,1,A,1]

    lon_n = ((lon - cfg.lon_min) / (cfg.lon_max - cfg.lon_min))[:, None, None, None]
    lat_n = ((lat - cfg.lat_min) / (cfg.lat_max - cfg.lat_min))[:, None, None, None]
    month = month0[:, None, None, None].astype(np.float32)
    year = year0[:, None, None, None].astype(np.float32)

    # Broad TL level: 60 dB at 1 km from 20 log10(r[m]); absorption grows
    # approximately linearly with range for this illustrative narrowband case.
    spreading = 60.0 + 20.0 * np.log10(r)
    absorption = (0.014 + 0.005 * lat_n) * r

    ssp_surface = ssp[:, :5].mean(axis=1)
    ssp_mid = ssp[:, 15:30].mean(axis=1)
    ssp_deep = ssp[:, -8:].mean(axis=1)
    duct_index = (ssp_surface - ssp_mid)[:, None, None, None]
    deep_index = (ssp_deep - ssp_mid)[:, None, None, None]

    # Depth dependent correction linked to SSP shape.
    depth_shape = (
        2.8 * np.exp(-((z - 120.0) / 120.0) ** 2) * np.tanh(duct_index / 8.0)
        - 2.0 * np.exp(-((z - 650.0) / 260.0) ** 2) * np.tanh(deep_index / 8.0)
    )

    # Horizontal/azimuth anisotropy and seasonal directional variability.
    bearing_phase = 2.0 * np.pi * lon_n + 1.4 * lat_n
    anisotropy = (2.3 + 0.6 * np.sin(2.0 * np.pi * month / 12.0)) * np.cos(az - bearing_phase)
    anisotropy *= (1.0 - np.exp(-r / 45.0))

    # Smooth refractive bands, roughly emulating convergence/shadow structure.
    refractive = -4.2 * np.cos(2.0 * np.pi * r / (82.0 + 6.0 * np.tanh(duct_index / 5.0)) + 0.004 * z)
    refractive *= np.exp(-r / 330.0)

    # Weak interference-like texture; larger near/mid range and damped far out.
    interference = 2.2 * np.sin(
        2.0 * np.pi * r / (17.0 + 2.5 * lon_n)
        + 0.012 * z
        + 0.7 * az
        + 0.4 * np.sin(2.0 * np.pi * month / 12.0)
    ) * np.exp(-r / 210.0)

    # A geographically varying shadow-like loss region.
    shadow_center = 145.0 + 22.0 * (lat_n - 0.5) - 12.0 * (lon_n - 0.5)
    shadow = 5.0 * np.exp(-((r - shadow_center) / 34.0) ** 2) * np.exp(-((z - 600.0) / 380.0) ** 2)

    seasonal_offset = 1.0 * np.sin(2.0 * np.pi * month / 12.0 + 0.3 * lon_n)
    interannual_offset = 0.45 * np.sin(2.0 * np.pi * year / 5.0 + 0.2 * lat_n)

    # Deterministic small-scale residual, replacing arbitrary white noise with
    # a spatially coherent perturbation.
    texture = 0.45 * np.sin(0.13 * r + 0.017 * z + 1.7 * az + 0.31 * month + 0.19 * year)

    tl = (
        spreading + absorption + depth_shape + anisotropy + refractive
        + interference + shadow + seasonal_offset + interannual_offset + texture
    )
    return tl.astype(np.float32).reshape(b, *cfg.field_shape)
