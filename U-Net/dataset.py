import numpy as np
import torch
from torch.utils.data import Dataset


class SoundFieldDataset(Dataset):
    """Generate environmental inputs and target TL fields on demand and select the corresponding geo-steady field."""

    def __init__(self, data, indices, cluster_labels, geo_steady_fields,
                 x1_mean, x1_std, tl_mean, tl_std):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
        self.geo_steady_fields = np.asarray(geo_steady_fields, dtype=np.float32)
        self.x1_mean = np.asarray(x1_mean, dtype=np.float32)
        self.x1_std = np.asarray(x1_std, dtype=np.float32)
        self.tl_mean = np.float32(tl_mean)
        self.tl_std = np.float32(tl_std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        sample_idx = int(self.indices[item])
        x1 = self.data.environment_vectors([sample_idx])[0]
        target = self.data.generate_tl([sample_idx])[0]
        _, _, point = self.data.decode_indices([sample_idx])
        cluster = int(self.cluster_labels[int(point[0])])
        background = self.geo_steady_fields[cluster]

        x1 = (x1 - self.x1_mean) / self.x1_std
        background = (background - self.tl_mean) / self.tl_std
        target = (target - self.tl_mean) / self.tl_std

        return (
            torch.from_numpy(x1.astype(np.float32)),
            torch.from_numpy(background.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )
