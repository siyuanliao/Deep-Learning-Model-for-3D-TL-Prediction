from dataclasses import dataclass
import numpy as np
import torch

from config import DATA
from synthetic_data import generate_environment, generate_tl


@dataclass
class Metrics:
    loss: float
    l1: float
    grad: float
    rmse: float


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_training_statistics(train_indices, batch_size=64, cfg=DATA):
    """Compute normalization statistics exclusively from the training split."""
    train_indices = np.asarray(train_indices, dtype=np.int64)

    # x1 is small enough to calculate exactly in one vectorized array.
    x1 = generate_environment(train_indices, cfg).astype(np.float64)
    x1_mean = x1.mean(axis=0).astype(np.float32)
    x1_std = (x1.std(axis=0) + 1e-6).astype(np.float32)

    # Stream target statistics to avoid allocating all 3-D fields at once.
    count = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for start in range(0, len(train_indices), batch_size):
        idx = train_indices[start:start + batch_size]
        y = generate_tl(idx, cfg).astype(np.float64)
        count += y.size
        sum_x += float(y.sum())
        sum_x2 += float(np.square(y).sum())

    mean = sum_x / count
    var = max(sum_x2 / count - mean * mean, 0.0)
    std = np.sqrt(var) + 1e-6
    return x1_mean, x1_std, np.float32(mean), np.float32(std)
