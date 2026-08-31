import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA
from synthetic_data import generate_environment, generate_tl, sample_index_to_ymp


class SoundFieldDataset(Dataset):
    """On-the-fly dataset with EOF-K geo-steady background lookup."""

    def __init__(
        self,
        indices,
        cluster_labels,
        geo_steady_fields,
        x1_mean,
        x1_std,
        t_mean,
        t_std,
        cfg=DATA,
    ):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
        self.geo_steady_fields = np.asarray(geo_steady_fields, dtype=np.float32)
        self.x1_mean = np.asarray(x1_mean, dtype=np.float32)
        self.x1_std = np.asarray(x1_std, dtype=np.float32)
        self.t_mean = np.float32(t_mean)
        self.t_std = np.float32(t_std)
        self.cfg = cfg

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_idx = int(self.indices[i])

        x1 = generate_environment(np.array([global_idx]), self.cfg)[0]
        target = generate_tl(np.array([global_idx]), self.cfg)[0]  # [36,4,250]

        # For the regular synthetic grid, point0 is already the closest source
        # grid point. For arbitrary future locations, eofk.location_to_cluster
        # provides the great-circle nearest-neighbor implementation.
        _, _, point0 = sample_index_to_ymp(np.array([global_idx]), self.cfg)
        cluster = int(self.cluster_labels[int(point0[0])])
        background = self.geo_steady_fields[cluster]  # [36,4,250]

        x1 = (x1 - self.x1_mean) / self.x1_std
        background = (background - self.t_mean) / self.t_std
        target = (target - self.t_mean) / self.t_std

        # Keep the same transpose convention as the original cnn-net.py.
        x2 = np.transpose(background, (1, 0, 2)).copy()  # [4,36,250]
        y = np.transpose(target, (1, 0, 2)).copy()       # [4,36,250]

        return (
            torch.from_numpy(x1.astype(np.float32, copy=False)),
            torch.from_numpy(x2.astype(np.float32, copy=False)),
            torch.from_numpy(y.astype(np.float32, copy=False)),
        )
