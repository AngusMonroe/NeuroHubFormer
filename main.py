"""Train and evaluate NeuroHubFormer with stratified 10-fold cross-validation.

For each fold, the test metrics are taken at the epoch with the highest validation
accuracy; the mean and standard deviation over folds are printed and appended to
--csv_path.
"""
import argparse
import csv
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import BrainDataset, collate
from model import NeuroHubFormer

METRICS = ("acc", "auc", "f1", "precision", "recall", "sensitivity", "specificity", "loss")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_metrics(logits, labels):
    y_true = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    y_pred = probs.argmax(axis=1)
    classes = list(range(probs.shape[1]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    try:
        if len(classes) == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=classes)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=classes).astype(np.float64)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fp + fn)
    sensitivity = tp / np.maximum(tp + fn, 1e-8)
    specificity = tn / np.maximum(tn + fp, 1e-8)
    if len(classes) == 2:
        sensitivity, specificity = sensitivity[1], specificity[1]
    else:
        sensitivity, specificity = sensitivity.mean(), specificity.mean()

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up, then cosine decay to min_lr_ratio x the peak; one step per epoch."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t))
        return [base_lr * factor for base_lr in self.base_lrs]


class Trainer:
    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.model = NeuroHubFormer(
            num_nodes=dataset.num_nodes,
            num_classes=dataset.num_classes,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            interaction_dim=args.interaction_dim,
            shrinkage=args.shrinkage,
            eigh_backend=args.eigh_backend,
            hub_score_init=args.hub_score_init,
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = WarmupCosineLR(
            self.optimizer,
            warmup_steps=max(1, int(args.epochs * args.warmup_ratio)),
            total_steps=args.epochs,
            min_lr_ratio=args.min_lr_ratio,
        )

    def compute_loss(self, logits, labels):
        """L_cls + lambda_ent L_ent + lambda_cov L_cov."""
        loss = self.criterion(logits, labels)
        if self.args.entropy_loss_weight > 0:
            loss = loss + self.args.entropy_loss_weight * self.model.hub_entropy.mean()
        if self.args.coverage_loss_weight > 0:
            coverage_loss = ((self.model.hub_coverage - self.args.target_coverage) ** 2).mean()
            loss = loss + self.args.coverage_loss_weight * coverage_loss
        return loss

    def run_epoch(self, loader, train):
        self.model.train(train)
        total_loss = torch.zeros((), dtype=torch.float64, device=self.device)
        all_logits, all_labels = [], []
        for matrices, labels in loader:
            matrices = matrices.to(self.device).float()
            labels = labels.to(self.device)
            with torch.set_grad_enabled(train):
                logits = self.model(matrices)
                loss = self.compute_loss(logits, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optimizer.step()
            total_loss += loss.detach().double()
            all_logits.append(logits.detach())
            all_labels.append(labels)
        metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels))
        metrics["loss"] = total_loss.item() / max(1, len(loader))
        return metrics

    def run_fold(self, fold_idx):
        loaders = {
            split: DataLoader(getattr(self.dataset, split)[fold_idx], batch_size=self.args.batch_size,
                              shuffle=(split == "train"), collate_fn=collate)
            for split in ("train", "val", "test")
        }
        best_val, best_test, best_epoch = -float("inf"), None, 0
        progress = tqdm(range(1, self.args.epochs + 1), desc=f"Fold {fold_idx + 1}")
        for epoch in progress:
            train_metrics = self.run_epoch(loaders["train"], train=True)
            val_metrics = self.run_epoch(loaders["val"], train=False)
            test_metrics = self.run_epoch(loaders["test"], train=False)
            self.scheduler.step()

            if val_metrics["acc"] > best_val:
                best_val, best_test, best_epoch = val_metrics["acc"], test_metrics, epoch
                if self.args.checkpoint_dir:
                    os.makedirs(self.args.checkpoint_dir, exist_ok=True)
                    torch.save({"model": self.model.state_dict(), "args": vars(self.args)},
                               os.path.join(self.args.checkpoint_dir, f"fold_{fold_idx + 1}.pt"))
            progress.set_postfix(
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                TrainLoss=f"{train_metrics['loss']:.4f}",
                ValAcc=f"{val_metrics['acc']:.4f}",
                TestAcc=f"{test_metrics['acc']:.4f}",
            )
        print(f"Fold {fold_idx + 1} best epoch: {best_epoch}, "
              f"test ACC: {best_test['acc']:.6f}, test AUC: {best_test['auc']:.6f}")
        return best_test


def parse_args():
    parser = argparse.ArgumentParser(description="NeuroHubFormer: 10-fold cross-validation.")
    # Data.
    parser.add_argument("--data_name", type=str, default="abide_schaefer100",
                        help="One of the datasets in data/dataset.py:DATASET_FILES.")
    parser.add_argument("--data_root", type=str, default="./datasets",
                        help="Folder with the dataset files.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to a dataset file (overrides --data_root).")
    parser.add_argument("--fold", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id; the CPU is used without CUDA.")
    # Optimization.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    # Architecture.
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--interaction_dim", type=int, default=64,
                        help="Width d_s of the interaction projection W_MI.")
    parser.add_argument("--shrinkage", type=float, default=0.1,
                        help="Shrinkage alpha toward the identity in the geometric view.")
    parser.add_argument("--eigh_backend", type=str, default="cpu", choices=["cpu", "native"],
                        help="'cpu': threaded LAPACK eigendecomposition on the host (fast for N > 32 "
                             "on CUDA); 'native': torch.linalg.eigh on the model's device.")
    parser.add_argument("--hub_score_init", type=str, default="random", choices=["random", "fixed"],
                        help="Initial weights of the hub logit: 'random' keeps PyTorch's default "
                             "initialization (used for the reported results); 'fixed' starts "
                             "a_u = a_rho = a_kappa = 1 and c = 0.")
    # Gate losses.
    parser.add_argument("--entropy_loss_weight", type=float, default=0.3)
    parser.add_argument("--coverage_loss_weight", type=float, default=2.0)
    parser.add_argument("--target_coverage", type=float, default=0.25)
    # Output.
    parser.add_argument("--csv_path", type=str, default="./results/summary.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="",
                        help="Save the best-validation model of every fold here (off when empty).")
    return parser.parse_args()


def write_summary(args, fold_metrics):
    means = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in METRICS}
    stds = {k: float(np.nanstd([m[k] for m in fold_metrics])) for k in METRICS}
    print(f"\n=== {args.fold}-fold summary ===")
    for key in METRICS:
        print(f"{key}: {means[key]:.6f} +/- {stds[key]:.6f}")

    config_keys = ["fold", "seed", "epochs", "batch_size", "lr", "weight_decay", "hidden_dim", "ff_dim",
                   "num_heads", "num_layers", "dropout", "interaction_dim", "shrinkage", "hub_score_init",
                   "entropy_loss_weight", "coverage_loss_weight", "target_coverage"]
    header = ["timestamp", "data_name", *METRICS, *config_keys]
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), args.data_name,
           *(f"{means[k]:.6f}+/-{stds[k]:.6f}" for k in METRICS),
           *(getattr(args, k) for k in config_keys)]
    os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(args.csv_path)
    with open(args.csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print("Saved summary row to:", args.csv_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    dataset = BrainDataset(args.data_name, data_root=args.data_root, data_path=args.data_path, fold=args.fold)
    fold_metrics = []
    for fold_idx in range(args.fold):
        set_seed(args.seed + fold_idx)
        fold_metrics.append(Trainer(args, dataset, device).run_fold(fold_idx))
    write_summary(args, fold_metrics)


if __name__ == "__main__":
    main()
