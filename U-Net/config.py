from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    years: int = 13
    months: int = 12
    grid_side: int = 11
    ssp_levels: int = 50
    channels: int = 4
    depth_bins: int = 36
    range_bins: int = 250
    lon_min: float = -133.0
    lon_max: float = -132.0
    lat_min: float = 36.0
    lat_max: float = 37.0

    @property
    def points(self):
        return self.grid_side * self.grid_side

    @property
    def n_samples(self):
        return self.years * self.months * self.points


DATA = DataConfig()

# Chronological extrapolation split: years 1-11 for training, year 12 for validation, and year 13 for testing.
# EOF, K-means, geo-steady fields, and normalization statistics are fitted using the training years only.
TRAIN_YEARS = tuple(range(1, 12))
VAL_YEARS = (12,)
TEST_YEARS = (13,)

# Default U-Net training parameters used by this repository.
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 20
DEFAULT_SEED = 42
DEFAULT_BASE_CHANNELS = 32
DEFAULT_COND_DIM = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_CLUSTERS = 4
