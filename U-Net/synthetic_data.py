import numpy as np
from config import DATA


class SyntheticOceanData:
    """Generate synthetic environmental vectors and 3-D transmission-loss fields for this self-contained example."""

    def __init__(self, cfg=DATA):
        self.cfg = cfg
        lon_values = np.linspace(cfg.lon_min, cfg.lon_max, cfg.grid_side, dtype=np.float32)
        lat_values = np.linspace(cfg.lat_min, cfg.lat_max, cfg.grid_side, dtype=np.float32)
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        self.grid_lonlat = np.column_stack([lon_grid.ravel(), lat_grid.ravel()]).astype(np.float32)

        self.ssp_depth = np.linspace(0.0, 1000.0, cfg.ssp_levels, dtype=np.float32)
        self.receiver_depth = np.linspace(10.0, 1000.0, cfg.depth_bins, dtype=np.float32)
        self.range_km = np.linspace(1.0, 250.0, cfg.range_bins, dtype=np.float32)
        self.azimuth = np.deg2rad(np.arange(cfg.channels, dtype=np.float32) * 90.0)

        # These base fields depend only on the receiver grid and are precomputed to reduce repeated work during training.
        r = self.range_km[None, None, :]
        z = self.receiver_depth[None, :, None]
        theta = self.azimuth[:, None, None]
        self._spreading = 20.0 * np.log10(r)
        self._absorption = 0.0105 * r
        self._depth_shape = 3.3 * (z / 1000.0) + 1.7 * np.sin(np.pi * z / 1000.0)
        self._azimuth_pattern = np.cos(theta) + 0.45 * np.sin(2.0 * theta)
        self._interference = np.sin(2.0 * np.pi * r / 43.0 + z / 145.0 + 0.6 * theta)
        self._shadow = np.exp(-((r - 165.0) / 43.0) ** 2) * np.exp(-((z - 540.0) / 300.0) ** 2)

    def decode_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        per_year = self.cfg.months * self.cfg.points
        year = indices // per_year + 1
        rem = indices % per_year
        month = rem // self.cfg.points + 1
        point = rem % self.cfg.points
        return year, month, point

    def indices_for_years(self, years):
        years = np.asarray(tuple(years), dtype=np.int64)
        year_ids = np.repeat(np.arange(1, self.cfg.years + 1), self.cfg.months * self.cfg.points)
        return np.where(np.isin(year_ids, years))[0].astype(np.int64)

    def generate_ssp(self, year, month, point):
        """Return SSPs with shape [B, 50], including spatial, seasonal, and interannual variability."""
        year = np.atleast_1d(year).astype(np.float32)
        month = np.atleast_1d(month).astype(np.float32)
        point = np.atleast_1d(point).astype(np.int64)
        lon = self.grid_lonlat[point, 0]
        lat = self.grid_lonlat[point, 1]

        z = self.ssp_depth[None, :]
        lon_n = (lon - self.cfg.lon_min) / (self.cfg.lon_max - self.cfg.lon_min)
        lat_n = (lat - self.cfg.lat_min) / (self.cfg.lat_max - self.cfg.lat_min)
        season = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        season_q = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        interannual = (year - 7.0) / 6.0

        surface = 1517.0 + 2.5 * lat_n[:, None] - 1.4 * lon_n[:, None]
        thermocline = -22.0 / (1.0 + np.exp(-(z - (150.0 + 35.0 * lat_n[:, None])) / 48.0))
        deep = 0.017 * np.maximum(z - 500.0, 0.0)
        seasonal = (4.2 * season[:, None] * np.exp(-z / 210.0) +
                    1.3 * season_q[:, None] * np.exp(-((z - 260.0) / 180.0) ** 2))
        spatial_mode = (2.2 * np.sin(np.pi * lon_n)[:, None] * np.exp(-z / 420.0) +
                        1.7 * np.cos(np.pi * lat_n)[:, None] * (z / 1000.0))
        year_term = 0.75 * interannual[:, None] * np.exp(-z / 550.0)
        fine = 0.35 * np.sin(z / 82.0 + 2.3 * lon_n[:, None] + 1.5 * lat_n[:, None])

        return (surface + thermocline + deep + seasonal + spatial_mode + year_term + fine).astype(np.float32)

    def environment_vectors(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        year, month, point = self.decode_indices(indices)
        ssp = self.generate_ssp(year, month, point)
        lonlat = self.grid_lonlat[point]
        return np.concatenate([lonlat, ssp], axis=1).astype(np.float32)

    def generate_tl(self, indices):
        """Return physics-inspired TL fields with shape [B, 4, 36, 250] in dB."""
        indices = np.atleast_1d(indices).astype(np.int64)
        year, month, point = self.decode_indices(indices)
        ssp = self.generate_ssp(year, month, point)
        lon = self.grid_lonlat[point, 0]
        lat = self.grid_lonlat[point, 1]

        lon_n = (lon - self.cfg.lon_min) / (self.cfg.lon_max - self.cfg.lon_min)
        lat_n = (lat - self.cfg.lat_min) / (self.cfg.lat_max - self.cfg.lat_min)
        season = np.sin(2.0 * np.pi * (month.astype(np.float32) - 1.0) / 12.0)
        interannual = (year.astype(np.float32) - 7.0) / 6.0

        surface_ssp = ssp[:, 0]
        mid_ssp = ssp[:, self.cfg.ssp_levels // 2]
        deep_ssp = ssp[:, -1]
        stratification = surface_ssp - mid_ssp
        deep_gradient = deep_ssp - mid_ssp

        b = len(indices)
        field = np.broadcast_to(
            38.0 + self._spreading + self._absorption + self._depth_shape,
            (b, self.cfg.channels, self.cfg.depth_bins, self.cfg.range_bins),
        ).copy()

        spatial = 2.6 * (lon_n - 0.5) - 2.0 * (lat_n - 0.5)
        field += spatial[:, None, None, None]
        field += (1.4 + 0.07 * stratification)[:, None, None, None] * self._azimuth_pattern[None, :, :, :]
        field += (2.0 + 0.05 * deep_gradient)[:, None, None, None] * self._interference[None, :, :, :]
        field += (4.2 + 1.0 * lat_n)[:, None, None, None] * self._shadow[None, :, :, :]

        r = self.range_km[None, None, None, :]
        z = self.receiver_depth[None, None, :, None]
        refr_phase = (r / (37.0 + 5.0 * lat_n[:, None, None, None]) +
                      z / (210.0 + 25.0 * lon_n[:, None, None, None]))
        field += 1.8 * np.sin(2.0 * np.pi * refr_phase)

        seasonal_shape = np.exp(-self.receiver_depth[None, None, :, None] / 320.0)
        range_growth = np.sqrt(self.range_km[None, None, None, :] / 250.0)
        field += 1.6 * season[:, None, None, None] * seasonal_shape * range_growth
        field += 0.65 * interannual[:, None, None, None] * range_growth

        # Deterministic fine-scale texture introduces weak propagation variability while keeping each sample reproducible.
        phase = (indices.astype(np.float32) * 0.017)[:, None, None, None]
        texture = 0.45 * np.sin(
            phase + self.azimuth[None, :, None, None] +
            self.receiver_depth[None, None, :, None] / 71.0 +
            self.range_km[None, None, None, :] / 17.0
        )
        field += texture
        return field.astype(np.float32)
