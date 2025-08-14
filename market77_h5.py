# data/merket77_h5.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Market77SegDataset(Dataset):
    """
    Reads an HDF5 of shape:
      - seg_points: (N_samples, 20480, 3)
      - seg_colors: (N_samples, 20480, 3)
      - seg_labels: (N_samples, 20480, C)   # C=2 for binary, or C=10 for full 10-way
    and returns per-sample:
      - feats: torch.FloatTensor of shape (C_in, 20480)
      - label: torch.LongTensor of shape (20480,)
    where C_in = 3 (xyz) or 6 (xyz + rgb) depending on use_color.
    """
    def __init__(self, h5_path, split='train', transform=None, use_color=False):
        super().__init__()
        # load everything into memory for simplicity
        with h5py.File(h5_path, 'r') as f:
            points = f['seg_points'][:]    # (N, 20480, 3)
            colors = f['seg_colors'][:]    # (N, 20480, 3)
            labels = f['seg_labels'][:]         # (N, 20480, C)
        # convert one-hot to integer labels
        if labels.ndim == 3:
            labels = np.argmax(labels, axis=-1).astype('int64')
        else:
            labels = labels.astype('int64')
        # Optionally split into train/test
        n = points.shape[0]
        if split is not None and split in ['train', 'test']:
            split_idx = int(n * 0.8)
            if split == 'train':
                self.points = points[:split_idx]
                self.colors = colors[:split_idx]
                self.labels = labels[:split_idx]
            else:
                self.points = points[split_idx:]
                self.colors = colors[split_idx:]
                self.labels = labels[split_idx:]
        else:
            self.points = points
            self.colors = colors
            self.labels = labels
        self.transform = transform
        self.use_color = use_color

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        # (20480,3)
        pts = self.points[idx].astype('float32')
        # (20480,)
        lbl = self.labels[idx]
        if self.use_color:
            # (20480,3)
            col = self.colors[idx].astype('float32') / 255.0
            # concat → (20480,6)
            feats = np.concatenate([pts, col], axis=1)
        else:
            feats = pts

        if self.transform:
            feats, lbl = self.transform(feats, lbl)

        # to tensor: (C_in, N)
        feats = torch.from_numpy(feats).float().transpose(0, 1)
        # (N,)
        lbl = torch.from_numpy(lbl).long()
        return feats, lbl