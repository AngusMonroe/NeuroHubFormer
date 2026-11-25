"""Brain connectivity datasets and their 10-fold train/val/test splits.

Each dataset is a torch file (torch.save) holding a dict with
  "signals": a sequence of BOLD signals, one [N_rois, T] tensor per subject,
  "labels":  a 1-D tensor with one class label per subject.
Every subject is represented by the Pearson correlation matrix of its signals.
"""
import csv
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

# Dataset name -> file name inside --data_root.
DATASET_FILES = {
    "abide_schaefer100": "abide_full_schaefer100.pt",
    "adni_schaefer100": "adni_schaefer100.pt",
    "ppmi_schaefer100": "ppmi_schaefer100.pt",
    "hcpgender_schaefer100": "HCPGender_schaefer100.pt",
    "hcpage_schaefer100": "HCPAge_schaefer100.pt",
}

SPLIT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def correlation_matrix(bold):
    """Pearson correlation of a [N, T] signal array; NaNs (zero-variance ROIs) -> 0, unit diagonal."""
    corr = np.nan_to_num(np.corrcoef(bold), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return torch.from_numpy(corr).float()


class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, matrices, labels):
        self.matrices = matrices
        self.labels = labels

    def __getitem__(self, index):
        return self.matrices[index], self.labels[index]

    def __len__(self):
        return len(self.matrices)


class BrainDataset:
    """Connectivity matrices of one dataset with fold-wise train/val/test subsets.

    Splits are read from data/splits/<dataset>/{train,val,test}.index (one row of
    subject indices per fold). If they do not exist, they are created with a
    stratified K-fold (seed 42); within each fold, a stratified 11% of the
    non-test subjects forms the validation set.
    """

    def __init__(self, name, data_root="./datasets", data_path=None, fold=10):
        self.name = name.lower()
        self.fold = fold
        if data_path is None:
            if self.name not in DATASET_FILES:
                raise KeyError(f"Unknown dataset '{self.name}'. Pass --data_path or add it to DATASET_FILES.")
            data_path = os.path.join(data_root, DATASET_FILES[self.name])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}.")
        content = torch.load(data_path, map_location="cpu")
        signals, labels = content["signals"], content["labels"].clone()

        if self.name.startswith("adni"):
            labels[labels == 5] = 4

        # Subjects with a negative label are excluded from all splits.
        excluded = set()
        for i in range(len(signals)):
            if labels[i] < 0:
                labels[i] = 1e4
                excluded.add(i)
            # An all-zero ROI signal is replaced by noise so its correlations are defined.
            for j in range(signals[i].size(0)):
                if (signals[i][j] == 0).all():
                    signals[i][j] = torch.randn_like(signals[i][j])

        unique_labels = sorted(set(labels.tolist()))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        self.labels = [label_map[label] for label in labels.tolist()]
        self.num_classes = len(unique_labels)
        self.matrices = [correlation_matrix(s.numpy()) for s in signals]
        self.num_nodes = self.matrices[0].size(0)

        self.split_idx = self._load_splits()
        self.train, self.val, self.test = [], [], []
        for k in range(fold):
            for section, subsets in (("train", self.train), ("val", self.val), ("test", self.test)):
                idx = [i for i in self.split_idx[section][k] if i not in excluded]
                subsets.append(SubjectDataset([self.matrices[i] for i in idx], [self.labels[i] for i in idx]))
        print(f"Loaded {self.name}: {len(self.matrices)} subjects, {self.num_nodes} ROIs, "
              f"{self.num_classes} classes")

    def _load_splits(self):
        split_dir = os.path.join(SPLIT_ROOT, self.name.split("_")[0])
        if not os.path.exists(os.path.join(split_dir, "train.index")):
            print(f"Creating {self.fold}-fold splits in {split_dir}")
            os.makedirs(split_dir, exist_ok=True)
            folds = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=42)
            rows = {"train": [], "val": [], "test": []}
            for remain, test in folds.split(self.matrices, self.labels):
                local = list(range(len(remain)))
                train_local, val_local = train_test_split(
                    local, test_size=0.11, random_state=42, stratify=[self.labels[i] for i in remain]
                )
                rows["train"].append([int(remain[i]) for i in train_local])
                rows["val"].append([int(remain[i]) for i in val_local])
                rows["test"].append([int(i) for i in test])
            for section, section_rows in rows.items():
                with open(os.path.join(split_dir, f"{section}.index"), "w", newline="") as f:
                    csv.writer(f).writerows(section_rows)

        split_idx = {}
        for section in ("train", "val", "test"):
            with open(os.path.join(split_dir, f"{section}.index")) as f:
                split_idx[section] = [list(map(int, row)) for row in csv.reader(f)]
        if len(split_idx["test"]) != self.fold:
            raise ValueError(f"{split_dir} holds {len(split_idx['test'])} folds, expected {self.fold}.")
        return split_idx


def collate(samples):
    matrices, labels = zip(*samples)
    return torch.stack(matrices), torch.tensor(labels, dtype=torch.long)
