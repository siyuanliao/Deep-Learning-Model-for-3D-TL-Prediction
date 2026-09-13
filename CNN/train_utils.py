from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Metrics:
    loss: float
    l1: float
    grad: float
    rmse: float


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_environment_normalization(data, train_indices):
    """Compute normalization statistics for the 52-dimensional environmental input using the training set only."""
    x1 = data.environment_vectors(train_indices)
    mean = x1.mean(axis=0).astype(np.float32)
    std = (x1.std(axis=0) + 1e-6).astype(np.float32)
    return mean, std


def calculate_model_complexity(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model complexity")
    print("=" * 50)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Approximately: {total / 1e6:.2f} M parameters")
    return total, trainable
