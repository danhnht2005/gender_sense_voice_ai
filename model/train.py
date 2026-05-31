"""
train.py - Stable training pipeline for gender voice classification.

Key safeguards:
- Group-based train/validation/test split to reduce leakage from repeated
  utterance variants.
- Explicit sample-id, group-id, and feature-hash overlap checks.
- Normalization fit on train only.
- AdamW + warmup cosine schedule + gradient clipping + early stopping.
- Best checkpoint saved by validation loss.
"""

import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import augment_dataset, build_dataset  # noqa: E402
from model_tcn_transformer_attention import (  # noqa: E402
    TCN_Transformer_Attention_Model,
    count_parameters,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return

        self.counter += 1
        print(f"  EarlyStopping: {self.counter}/{self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True


class ModelCheckpoint:
    def __init__(self, path):
        self.path = path
        self.best_val_loss = float("inf")

    def save_if_best(self, val_loss, payload):
        if val_loss >= self.best_val_loss:
            return False
        self.best_val_loss = val_loss
        torch.save(payload, self.path)
        return True


def make_group_split(y, groups, seed=42, train_size=0.70, val_size=0.15):
    """Split indices by group; no group may appear in more than one split."""
    indices = np.arange(len(y))
    temp_size = 1.0 - train_size

    first_split = GroupShuffleSplit(n_splits=1, test_size=temp_size, random_state=seed)
    train_idx, temp_idx = next(first_split.split(indices, y, groups=groups))

    relative_test_size = 0.5
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=relative_test_size, random_state=seed
    )
    val_rel_idx, test_rel_idx = next(
        second_split.split(temp_idx, y[temp_idx], groups=groups[temp_idx])
    )

    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    return train_idx, val_idx, test_idx


def feature_hashes(X):
    return np.array([hashlib.sha1(row.tobytes()).hexdigest() for row in X], dtype=str)


def split_overlap_report(name_a, idx_a, name_b, idx_b, sample_ids, group_ids, hashes):
    id_overlap = set(sample_ids[idx_a]).intersection(set(sample_ids[idx_b]))
    group_overlap = set(group_ids[idx_a]).intersection(set(group_ids[idx_b]))
    hash_overlap = set(hashes[idx_a]).intersection(set(hashes[idx_b]))
    print(
        f"  {name_a}/{name_b}: "
        f"id_overlap={len(id_overlap)}, group_overlap={len(group_overlap)}, "
        f"hash_overlap={len(hash_overlap)}"
    )
    return {
        "pair": f"{name_a}/{name_b}",
        "id_overlap": len(id_overlap),
        "group_overlap": len(group_overlap),
        "hash_overlap": len(hash_overlap),
        "sample_id_examples": sorted(list(id_overlap))[:10],
        "group_examples": sorted(list(group_overlap))[:10],
        "hash_examples": sorted(list(hash_overlap))[:10],
    }


def label_distribution(name, y):
    counts = {
        "male": int(np.sum(y == 0)),
        "female": int(np.sum(y == 1)),
        "total": int(len(y)),
    }
    female_ratio = counts["female"] / max(counts["total"], 1)
    print(
        f"{name:<6}: {counts['total']:>5} samples | "
        f"Male={counts['male']:>5} Female={counts['female']:>5} "
        f"FemaleRatio={female_ratio:.3f}"
    )
    return counts


def normalize_from_train(X_train, X_val, X_test):
    train_mean = np.mean(X_train, axis=(0, 1))
    train_std = np.std(X_train, axis=(0, 1))
    train_std[train_std < 1e-8] = 1e-8
    return (
        (X_train - train_mean) / train_std,
        (X_val - train_mean) / train_std,
        (X_test - train_mean) / train_std,
        train_mean,
        train_std,
    )


def smooth_targets(y, smoothing=0.03):
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    gradient_clip_norm=1.0,
    label_smoothing=0.03,
):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).unsqueeze(1)
        loss_targets = smooth_targets(batch_y, label_smoothing)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_X)
        loss = criterion(logits, loss_targets)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        optimizer.step()

        total_loss += loss.item() * batch_X.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / total, correct / total, float(grad_norm)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * batch_X.size(0)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def compute_binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def make_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.10):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_loader(X, y, batch_size, shuffle, device):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    pin_memory = device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_dir, "dataset")
    outputs_dir = os.path.join(project_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    seed = 42
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 1e-4
    epochs = 60
    warmup_epochs = 5
    patience = 15
    gradient_clip_norm = 1.0
    label_smoothing = 0.03

    input_dim = 60
    embed_dim = 64
    num_heads = 4
    tcn_channels = 64
    num_tcn = 3
    num_transformer = 1
    dropout = 0.45

    print("=" * 70)
    print("  STABLE TRAINING PIPELINE - Gender Voice Classification")
    print("=" * 70)
    set_seed(seed)
    device = get_device()

    print("\n[1] Load data/cache")
    X, y, sample_ids, group_ids = build_dataset(
        dataset_dir=dataset_dir,
        cache_dir=outputs_dir,
    )
    y = y.astype(np.float32)
    sample_ids = sample_ids.astype(str)
    group_ids = group_ids.astype(str)
    hashes = feature_hashes(X)

    print("\n[2] Group-based split and leakage audit")
    train_idx, val_idx, test_idx = make_group_split(y, group_ids, seed=seed)

    split_counts = {
        "train": label_distribution("Train", y[train_idx]),
        "val": label_distribution("Val", y[val_idx]),
        "test": label_distribution("Test", y[test_idx]),
    }
    print(
        f"Groups: train={len(np.unique(group_ids[train_idx]))}, "
        f"val={len(np.unique(group_ids[val_idx]))}, "
        f"test={len(np.unique(group_ids[test_idx]))}"
    )

    overlap_reports = [
        split_overlap_report("train", train_idx, "val", val_idx, sample_ids, group_ids, hashes),
        split_overlap_report("train", train_idx, "test", test_idx, sample_ids, group_ids, hashes),
        split_overlap_report("val", val_idx, "test", test_idx, sample_ids, group_ids, hashes),
    ]
    if any(r["id_overlap"] or r["group_overlap"] or r["hash_overlap"] for r in overlap_reports):
        raise RuntimeError("Leakage audit failed: split overlap detected.")

    split_report = {
        "split_strategy": "GroupShuffleSplit by canonical utterance id",
        "label_distribution": split_counts,
        "overlap_reports": overlap_reports,
        "train_sample_ids": sample_ids[train_idx].tolist(),
        "val_sample_ids": sample_ids[val_idx].tolist(),
        "test_sample_ids": sample_ids[test_idx].tolist(),
    }
    with open(os.path.join(outputs_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\n[3] Normalize with train statistics only")
    X_train_norm, X_val_norm, X_test_norm, train_mean, train_std = normalize_from_train(
        X_train, X_val, X_test
    )
    with open(os.path.join(outputs_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": train_mean.tolist(), "std": train_std.tolist()}, f, indent=2)

    print("\n[4] Augment train set only")
    X_train_aug, y_train_aug = augment_dataset(
        X_train_norm,
        y_train,
        noise_factor=0.003,
        time_shift_max=3,
        freq_mask_max=3,
        num_augmented=1,
    )

    train_loader = make_loader(X_train_aug, y_train_aug, batch_size, True, device)
    val_loader = make_loader(X_val_norm, y_val, batch_size, False, device)
    test_loader = make_loader(X_test_norm, y_test, batch_size, False, device)
    print(
        f"Loaders: train_batches={len(train_loader)}, "
        f"val_batches={len(val_loader)}, test_batches={len(test_loader)}, "
        f"batch_size={batch_size}"
    )

    print("\n[5] Model/optimizer/scheduler")
    model = TCN_Transformer_Attention_Model(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        tcn_channels=tcn_channels,
        num_tcn=num_tcn,
        num_transformer=num_transformer,
        dropout=dropout,
    ).to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")

    n_male = np.sum(y_train_aug == 0)
    n_female = np.sum(y_train_aug == 1)
    pos_weight_value = n_male / max(n_female, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    print(
        f"Class balance train_aug - Male={n_male}, Female={n_female}, "
        f"pos_weight={pos_weight.item():.4f}"
    )

    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        min_lr_ratio=0.10,
    )
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)
    checkpoint_path = os.path.join(outputs_dir, "best_model.pth")
    checkpoint = ModelCheckpoint(checkpoint_path)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "lr": [],
        "grad_norm": [],
    }
    hyperparameters = {
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "gradient_clip_norm": gradient_clip_norm,
        "label_smoothing": label_smoothing,
        "scheduler": "warmup + cosine LambdaLR",
        "dropout": dropout,
    }

    print("\n[6] Training")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, grad_norm = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            device,
            gradient_clip_norm=gradient_clip_norm,
            label_smoothing=label_smoothing,
        )
        val_loss, val_acc, val_preds, val_labels, _ = validate_one_epoch(
            model, val_loader, eval_criterion, device
        )
        val_metrics = compute_binary_metrics(val_labels, val_preds)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["lr"].append(float(current_lr))
        history["grad_norm"].append(float(grad_norm))

        saved_best = checkpoint.save_if_best(
            val_loss,
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_f1": val_metrics["f1"],
                "val_confusion_matrix": val_metrics["confusion_matrix"],
                "hyperparameters": hyperparameters,
            },
        )
        best_marker = " [BEST]" if saved_best else ""
        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"LR={current_lr:.6f} "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} "
            f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} ValF1={val_metrics['f1']:.4f} "
            f"GradNorm={grad_norm:.3f} "
            f"Time={time.time() - epoch_start:.1f}s{best_marker}"
        )

        scheduler.step()
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    with open(os.path.join(outputs_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Training completed in {total_time:.1f}s ({total_time / 60:.1f} min)")

    print("\n[7] Test evaluation with best checkpoint")
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_acc, test_preds, test_labels, test_probs = validate_one_epoch(
        model, test_loader, eval_criterion, device
    )
    test_metrics = compute_binary_metrics(test_labels, test_preds)
    test_report = classification_report(
        test_labels,
        test_preds,
        target_names=["Male (0)", "Female (1)"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    print(classification_report(
        test_labels,
        test_preds,
        target_names=["Male (0)", "Female (1)"],
        digits=4,
        zero_division=0,
    ))
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    np.save(os.path.join(outputs_dir, "test_preds.npy"), test_preds)
    np.save(os.path.join(outputs_dir, "test_labels.npy"), test_labels)
    np.save(os.path.join(outputs_dir, "test_probs.npy"), test_probs)

    evaluation_summary = {
        "test_loss": float(test_loss),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_report,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_loss": float(best_checkpoint["val_loss"]),
        "best_val_acc": float(best_checkpoint["val_acc"]),
        "best_val_f1": float(best_checkpoint["val_f1"]),
        "hyperparameters": hyperparameters,
        "split_strategy": split_report["split_strategy"],
    }
    with open(os.path.join(outputs_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    if history["val_acc"] and history["val_acc"][0] >= 0.995:
        print(
            "[WARNING] Validation accuracy is near 1.0 from epoch 1. "
            "After group/hash checks, the most likely cause is an easy or homogeneous "
            "dataset, not obvious sample overlap. Validate on external recordings."
        )

    print("=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best epoch: {best_checkpoint['epoch']}")
    print(f"Best val loss: {best_checkpoint['val_loss']:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Outputs saved to: {outputs_dir}")


if __name__ == "__main__":
    main()
