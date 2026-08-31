from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DataConfig:
    years: int = 13
    months: int = 12
    grid_n_lon: int = 11
    grid_n_lat: int = 11
    lon_min: float = -133.0
    lon_max: float = -132.0
    lat_min: float = 36.0
    lat_max: float = 37.0
    ssp_levels: int = 50
    receiver_depths: int = 36
    azimuth_channels: int = 4
    range_bins: int = 250

    @property
    def points(self) -> int:
        return self.grid_n_lon * self.grid_n_lat

    @property
    def n_samples(self) -> int:
        return self.years * self.months * self.points

    @property
    def field_shape(self):
        # Target field layout, consistent with the original implementation: [36, 4, 250]
        return (self.receiver_depths, self.azimuth_channels, self.range_bins)


DATA = DataConfig()

# Chronological split: years 1-11 for training, year 12 for validation, year 13 for testing.
# EOF, K-means, geo-steady fields, and normalization statistics are fitted using TRAIN_YEARS only.
TRAIN_YEARS = tuple(range(1, 12))
VAL_YEARS = (12,)
TEST_YEARS = (13,)


def build_source_grid(cfg: DataConfig = DATA):
    """Build an 11 x 11 regular longitude-latitude grid with 121 source locations."""
    lons = np.linspace(cfg.lon_min, cfg.lon_max, cfg.grid_n_lon, dtype=np.float32)
    lats = np.linspace(cfg.lat_min, cfg.lat_max, cfg.grid_n_lat, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lons, lats, indexing="xy")
    return np.column_stack([lon2d.ravel(), lat2d.ravel()]).astype(np.float32)
