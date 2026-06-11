"""Training and bounded hyperparameter search for gender voice classification."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import augment_dataset, build_dataset  # noqa: E402
from model_tcn_transformer_attention import (  # noqa: E402
    TCN_Transformer_Attention_Model,
    count_parameters,
)


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    dropout: float = 0.5
    scheduler: str = "cosine"
    batch_size: int = 64
    epochs: int = 30
    early_stopping_patience: int = 4
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    embed_dim: int = 64
    num_heads: int = 4
    tcn_channels: int = 64
    num_tcn: int = 3
    num_transformer: int = 1
    augment: bool = True
    noise_factor: float = 0.005
    time_shift_max: int = 3
    freq_mask_max: int = 2
    seed: int = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(payload, checkpoint_path, retries=8):
    """Write a checkpoint without opening the destination file directly."""
    temp_path = (
        f"{checkpoint_path}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    torch.save(payload, temp_path)
    try:
        for attempt in range(retries):
            try:
                os.replace(temp_path, checkpoint_path)
                return checkpoint_path
            except OSError:
                if attempt == retries - 1:
                    fallback_path = (
                        f"{os.path.splitext(checkpoint_path)[0]}_"
                        f"{time.time_ns()}.pth"
                    )
                    os.replace(temp_path, fallback_path)
                    print(
                        "[WARNING] Checkpoint destination remained locked; "
                        f"saved to {fallback_path}"
                    )
                    return fallback_path
                time.sleep(0.25 * (attempt + 1))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def copy_file_atomic(source, destination, retries=8):
    """Copy a file through a temporary path before replacing its destination."""
    temp_path = (
        f"{destination}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    shutil.copy2(source, temp_path)
    try:
        for attempt in range(retries):
            try:
                os.replace(temp_path, destination)
                return
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


def make_group_split(y, groups, seed=42, train_size=0.70, val_size=0.15):
    """Return an approximate 5/1/1 stratified split with disjoint groups."""
    del train_size, val_size
    indices = np.arange(len(y))
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in splitter.split(indices, y, groups=groups)]
    return np.concatenate(folds[2:]), folds[0], folds[1]


def feature_hashes(X):
    return np.array([hashlib.sha1(row.tobytes()).hexdigest() for row in X], dtype=str)


def split_overlap_report(name_a, idx_a, name_b, idx_b, sample_ids, group_ids, hashes):
    id_overlap = set(sample_ids[idx_a]).intersection(sample_ids[idx_b])
    group_overlap = set(group_ids[idx_a]).intersection(group_ids[idx_b])
    hash_overlap = set(hashes[idx_a]).intersection(hashes[idx_b])
    print(
        f"  {name_a}/{name_b}: id_overlap={len(id_overlap)}, "
        f"group_overlap={len(group_overlap)}, hash_overlap={len(hash_overlap)}"
    )
    return {
        "pair": f"{name_a}/{name_b}",
        "id_overlap": len(id_overlap),
        "group_overlap": len(group_overlap),
        "hash_overlap": len(hash_overlap),
    }


def label_distribution(name, y):
    counts = {
        "male": int(np.sum(y == 0)),
        "female": int(np.sum(y == 1)),
        "total": int(len(y)),
    }
    print(
        f"{name:<6}: {counts['total']:>5} | "
        f"Male={counts['male']:>5} Female={counts['female']:>5}"
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


def make_loader(X, y, batch_size, shuffle, device):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def smooth_targets(y, smoothing):
    return y if smoothing <= 0 else y * (1.0 - smoothing) + 0.5 * smoothing


def compute_binary_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, config):
    model.train()
    total_loss = 0.0
    last_grad_norm = 0.0
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_X)
        loss = criterion(logits, smooth_targets(batch_y, config.label_smoothing))
        loss.backward()
        last_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(loader.dataset), float(last_grad_norm)


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True).unsqueeze(1)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            probs = torch.sigmoid(logits)
            total_loss += loss.item() * batch_X.size(0)
            all_probs.extend(probs.cpu().numpy().ravel())
            all_labels.extend(batch_y.cpu().numpy().ravel())

    labels = np.asarray(all_labels)
    probs = np.asarray(all_probs)
    preds = (probs >= 0.5).astype(np.float32)
    metrics = compute_binary_metrics(labels, preds)
    return total_loss / len(loader.dataset), metrics, preds, labels, probs


def is_better(candidate, incumbent, f1_epsilon=1e-4):
    """Rank by val F1, then val loss, then the train-validation F1 gap."""
    if incumbent is None:
        return True
    f1_delta = candidate["val_f1"] - incumbent["val_f1"]
    if f1_delta > f1_epsilon:
        return True
    if abs(f1_delta) <= f1_epsilon:
        loss_delta = candidate["val_loss"] - incumbent["val_loss"]
        if loss_delta < -1e-5:
            return True
        if abs(loss_delta) <= 1e-5:
            return candidate["f1_gap"] < incumbent["f1_gap"]
    return False


def make_cosine_warmup_scheduler(optimizer, epochs, warmup_ratio=0.1):
    warmup_epochs = max(1, int(round(epochs * warmup_ratio)))

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, config):
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
    if config.scheduler == "cosine":
        return make_cosine_warmup_scheduler(optimizer, config.epochs)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def plot_history(history, save_path, title):
    epochs = np.arange(1, len(history["val_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("train_loss", "val_loss", "Loss", "loss"),
        ("train_acc", "val_acc", "Accuracy", "accuracy"),
        ("train_f1", "val_f1", "F1", "f1"),
    ]
    for ax, (train_key, val_key, panel_title, ylabel) in zip(axes.flat[:3], panels):
        ax.plot(epochs, history[train_key], label=train_key)
        ax.plot(epochs, history[val_key], label=val_key)
        ax.set_title(panel_title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()

    axes[1, 1].plot(epochs, history["lr"], color="tab:purple")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("lr")
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_data(project_dir, outputs_dir, seed):
    dataset_dir = os.path.join(project_dir, "dataset")
    X, y, sample_ids, group_ids = build_dataset(dataset_dir, cache_dir=outputs_dir)
    y = y.astype(np.float32)
    sample_ids = sample_ids.astype(str)
    group_ids = group_ids.astype(str)
    hashes = feature_hashes(X)

    train_idx, val_idx, test_idx = make_group_split(y, group_ids, seed=seed)
    counts = {
        "train": label_distribution("Train", y[train_idx]),
        "val": label_distribution("Val", y[val_idx]),
        "test": label_distribution("Test", y[test_idx]),
    }
    overlaps = [
        split_overlap_report("train", train_idx, "val", val_idx, sample_ids, group_ids, hashes),
        split_overlap_report("train", train_idx, "test", test_idx, sample_ids, group_ids, hashes),
        split_overlap_report("val", val_idx, "test", test_idx, sample_ids, group_ids, hashes),
    ]
    if any(r["id_overlap"] or r["group_overlap"] or r["hash_overlap"] for r in overlaps):
        raise RuntimeError("Leakage audit failed.")

    X_train, X_val, X_test, mean, std = normalize_from_train(
        X[train_idx], X[val_idx], X[test_idx]
    )
    with open(os.path.join(outputs_dir, "norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    split_report = {
        "split_strategy": "StratifiedGroupKFold by canonical utterance id",
        "label_distribution": counts,
        "overlap_reports": overlaps,
        "train_sample_ids": sample_ids[train_idx].tolist(),
        "val_sample_ids": sample_ids[val_idx].tolist(),
        "test_sample_ids": sample_ids[test_idx].tolist(),
    }
    with open(os.path.join(outputs_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(split_report, f, indent=2)
    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y[train_idx],
        "X_val": X_val.astype(np.float32),
        "y_val": y[val_idx],
        "X_test": X_test.astype(np.float32),
        "y_test": y[test_idx],
        "input_dim": int(X.shape[2]),
        "time_steps": int(X.shape[1]),
        "split_strategy": split_report["split_strategy"],
    }


def make_trial_loaders(data, config, device):
    if config.augment:
        X_train, y_train = augment_dataset(
            data["X_train"],
            data["y_train"],
            noise_factor=config.noise_factor,
            time_shift_max=config.time_shift_max,
            freq_mask_max=min(config.freq_mask_max, data["input_dim"] - 1),
            num_augmented=1,
        )
    else:
        X_train, y_train = data["X_train"], data["y_train"]
    return {
        "train": make_loader(X_train, y_train, config.batch_size, True, device),
        "train_eval": make_loader(
            data["X_train"], data["y_train"], config.batch_size, False, device
        ),
        "val": make_loader(data["X_val"], data["y_val"], config.batch_size, False, device),
    }


def train_trial(data, config, device, trial_dir, trial_name):
    os.makedirs(trial_dir, exist_ok=True)
    set_seed(config.seed)
    loaders = make_trial_loaders(data, config, device)
    model = TCN_Transformer_Attention_Model(
        input_dim=data["input_dim"],
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        tcn_channels=config.tcn_channels,
        num_tcn=config.num_tcn,
        num_transformer=config.num_transformer,
        dropout=config.dropout,
    ).to(device)

    n_male = int(np.sum(data["y_train"] == 0))
    n_female = int(np.sum(data["y_train"] == 1))
    pos_weight = torch.tensor([n_male / max(n_female, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimizer, config)
    checkpoint_path = os.path.join(
        trial_dir, f"best_model_run_{os.getpid()}.pth"
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        "lr": [],
        "grad_norm": [],
    }
    best = None
    epochs_without_f1_improvement = 0
    started = time.time()

    print(
        f"\n[{trial_name}] lr={config.learning_rate:g} wd={config.weight_decay:g} "
        f"dropout={config.dropout:.2f} scheduler={config.scheduler} augment={config.augment}"
    )
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])
        _, grad_norm = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, config
        )
        train_loss, train_metrics, _, _, _ = evaluate_loader(
            model, loaders["train_eval"], criterion, device
        )
        val_loss, val_metrics, _, _, _ = evaluate_loader(
            model, loaders["val"], criterion, device
        )
        candidate = {
            "epoch": epoch,
            "val_loss": float(val_loss),
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "train_loss": float(train_loss),
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "f1_gap": max(0.0, train_metrics["f1"] - val_metrics["f1"]),
            "learning_rate": current_lr,
        }
        history["train_loss"].append(candidate["train_loss"])
        history["val_loss"].append(candidate["val_loss"])
        history["train_acc"].append(candidate["train_acc"])
        history["val_acc"].append(candidate["val_acc"])
        history["train_f1"].append(candidate["train_f1"])
        history["val_f1"].append(candidate["val_f1"])
        history["lr"].append(current_lr)
        history["grad_norm"].append(grad_norm)

        f1_improved = best is None or candidate["val_f1"] > best["val_f1"] + 1e-4
        improved = is_better(candidate, best)
        if improved:
            best = candidate.copy()
            payload = {
                **best,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hyperparameters": {
                    **asdict(config),
                    "input_dim": data["input_dim"],
                    "time_steps": data["time_steps"],
                },
                "model_parameters": count_parameters(model),
            }
            checkpoint_path = save_checkpoint(payload, checkpoint_path)
        epochs_without_f1_improvement = 0 if f1_improved else epochs_without_f1_improvement + 1

        if config.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        marker = " [BEST]" if improved else ""
        print(
            f"  epoch {epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} lr={current_lr:.2e} "
            f"time={time.time() - epoch_started:.1f}s{marker}"
        )
        if epochs_without_f1_improvement >= config.early_stopping_patience:
            print(
                f"  Early stopping: val_f1 did not improve for "
                f"{config.early_stopping_patience} epochs."
            )
            break

    with open(os.path.join(trial_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, os.path.join(trial_dir, "training_curves.png"), trial_name)
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    result = {
        "trial": trial_name,
        "config": asdict(config),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_loss": float(best_checkpoint["val_loss"]),
        "best_val_acc": float(best_checkpoint["val_acc"]),
        "best_val_f1": float(best_checkpoint["val_f1"]),
        "train_f1_at_best": float(best_checkpoint["train_f1"]),
        "f1_gap": float(best_checkpoint["f1_gap"]),
        "learning_rate_at_best": float(best_checkpoint["learning_rate"]),
        "epochs_ran": len(history["val_loss"]),
        "training_seconds": time.time() - started,
        "checkpoint_path": checkpoint_path,
        "history_path": os.path.join(trial_dir, "history.json"),
        "curves_path": os.path.join(trial_dir, "training_curves.png"),
    }
    with open(os.path.join(trial_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    del model, optimizer, scheduler, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def result_is_better(candidate, incumbent):
    if incumbent is None:
        return True
    return is_better(
        {
            "val_f1": candidate["best_val_f1"],
            "val_loss": candidate["best_val_loss"],
            "f1_gap": candidate["f1_gap"],
        },
        {
            "val_f1": incumbent["best_val_f1"],
            "val_loss": incumbent["best_val_loss"],
            "f1_gap": incumbent["f1_gap"],
        },
    )


def search_configs(max_trials, max_epochs, seed):
    """Deterministic bounded random search over regularization and LR schedules."""
    core = [
        TrainConfig(learning_rate=2e-4, weight_decay=5e-4, dropout=0.4, scheduler="plateau"),
        TrainConfig(learning_rate=3e-4, weight_decay=1e-4, dropout=0.3, scheduler="plateau"),
        TrainConfig(learning_rate=1e-4, weight_decay=5e-4, dropout=0.4, scheduler="plateau"),
        TrainConfig(learning_rate=5e-5, weight_decay=1e-4, dropout=0.3, scheduler="plateau"),
        TrainConfig(learning_rate=3e-4, weight_decay=1e-3, dropout=0.4, scheduler="cosine"),
        TrainConfig(learning_rate=2e-4, weight_decay=5e-4, dropout=0.3, scheduler="cosine"),
        TrainConfig(learning_rate=1e-4, weight_decay=1e-3, dropout=0.5, scheduler="cosine"),
        TrainConfig(
            learning_rate=2e-4,
            weight_decay=5e-4,
            dropout=0.4,
            scheduler="plateau",
            augment=False,
        ),
    ]
    for config in core:
        config.epochs = max_epochs
        config.seed = seed

    rng = random.Random(seed)
    candidates = []
    for lr in [3e-4, 2e-4, 1e-4, 5e-5]:
        for dropout in [0.2, 0.3, 0.4, 0.5]:
            for weight_decay in [1e-5, 1e-4, 5e-4, 1e-3]:
                for scheduler in ["plateau", "cosine"]:
                    for augment in [True, False]:
                        candidates.append(
                            TrainConfig(
                                learning_rate=lr,
                                weight_decay=weight_decay,
                                dropout=dropout,
                                scheduler=scheduler,
                                epochs=max_epochs,
                                augment=augment,
                                noise_factor=0.005,
                                time_shift_max=3,
                                freq_mask_max=2,
                                seed=seed,
                            )
                        )
    rng.shuffle(candidates)
    selected = core[:]
    selected.extend(c for c in candidates if c not in selected)
    return selected[:max_trials]


def print_ranking(results):
    ranked = sorted(
        results,
        key=lambda r: (-r["best_val_f1"], r["best_val_loss"], r["f1_gap"]),
    )
    print("\n" + "=" * 105)
    print("SEARCH RANKING")
    print("=" * 105)
    print(
        f"{'Rank':>4} {'Trial':<10} {'Val F1':>8} {'Val loss':>9} "
        f"{'Val acc':>8} {'Gap':>8} {'Epoch':>6} {'LR':>9} {'Drop':>6} {'WD':>9} {'Schedule':>9}"
    )
    for rank, result in enumerate(ranked, 1):
        config = result["config"]
        print(
            f"{rank:>4} {result['trial']:<10} {result['best_val_f1']:>8.4f} "
            f"{result['best_val_loss']:>9.4f} {result['best_val_acc']:>8.4f} "
            f"{result['f1_gap']:>8.4f} {result['best_epoch']:>6} "
            f"{config['learning_rate']:>9.1e} {config['dropout']:>6.2f} "
            f"{config['weight_decay']:>9.1e} {config['scheduler']:>9}"
        )
    return ranked


def promote_best_trial(best, outputs_dir):
    copies = [
        (best["checkpoint_path"], os.path.join(outputs_dir, "best_model.pth")),
        (best["history_path"], os.path.join(outputs_dir, "training_history.json")),
        (best["curves_path"], os.path.join(outputs_dir, "best_training_curves.png")),
        (best["curves_path"], os.path.join(outputs_dir, "training_curves.png")),
    ]
    for source, destination in copies:
        if os.path.abspath(source) != os.path.abspath(destination):
            copy_file_atomic(source, destination)
    best_config = {
        "trial": best["trial"],
        "config": best["config"],
        "best_epoch": best["best_epoch"],
        "best_val_loss": best["best_val_loss"],
        "best_val_acc": best["best_val_acc"],
        "best_val_f1": best["best_val_f1"],
        "f1_gap": best["f1_gap"],
        "learning_rate_at_best": best["learning_rate_at_best"],
    }
    with open(os.path.join(outputs_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)


def evaluate_best(data, device, outputs_dir):
    checkpoint_path = os.path.join(outputs_dir, "best_model.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hp = checkpoint["hyperparameters"]
    model = TCN_Transformer_Attention_Model(
        input_dim=hp["input_dim"],
        embed_dim=hp["embed_dim"],
        num_heads=hp["num_heads"],
        tcn_channels=hp["tcn_channels"],
        num_tcn=hp["num_tcn"],
        num_transformer=hp["num_transformer"],
        dropout=hp["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loader = make_loader(data["X_test"], data["y_test"], hp["batch_size"], False, device)
    n_male = int(np.sum(data["y_train"] == 0))
    n_female = int(np.sum(data["y_train"] == 1))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_male / max(n_female, 1)], device=device)
    )
    test_loss, metrics, preds, labels, probs = evaluate_loader(
        model, test_loader, criterion, device
    )
    report = classification_report(
        labels,
        preds,
        target_names=["Male (0)", "Female (1)"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    np.save(os.path.join(outputs_dir, "test_preds.npy"), preds)
    np.save(os.path.join(outputs_dir, "test_labels.npy"), labels)
    np.save(os.path.join(outputs_dir, "test_probs.npy"), probs)
    summary = {
        "test_loss": float(test_loss),
        "accuracy": metrics["accuracy"],
        "f1_score": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": report,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_f1": float(checkpoint["val_f1"]),
        "train_f1_at_best": float(checkpoint["train_f1"]),
        "f1_gap_at_best": float(checkpoint["f1_gap"]),
        "learning_rate_at_best": float(checkpoint["learning_rate"]),
        "model_parameters": count_parameters(model),
        "hyperparameters": hp,
        "split_strategy": data["split_strategy"],
    }
    with open(os.path.join(outputs_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nBest checkpoint test: loss={test_loss:.4f} "
        f"accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", action="store_true", help="Run bounded hyperparameter search.")
    parser.add_argument("--max-trials", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--search-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="cosine")
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.max_trials <= 30:
        raise ValueError("--max-trials must be between 1 and 30.")
    if not 1 <= args.max_epochs <= 30:
        raise ValueError("--max-epochs must be between 1 and 30.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_dir, "outputs")
    trials_dir = os.path.join(outputs_dir, "search_trials")
    os.makedirs(outputs_dir, exist_ok=True)
    device = get_device()
    set_seed(args.seed)
    data = prepare_data(project_dir, outputs_dir, args.seed)

    if args.search:
        os.makedirs(trials_dir, exist_ok=True)
        configs = search_configs(args.max_trials, args.max_epochs, args.seed)
    else:
        configs = [
            TrainConfig(
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                dropout=args.dropout,
                scheduler=args.scheduler,
                epochs=args.max_epochs,
                augment=not args.no_augment,
                seed=args.seed,
            )
        ]

    results = []
    best = None
    trials_without_global_improvement = 0
    for index, config in enumerate(configs, 1):
        trial_name = f"trial_{index:02d}" if args.search else "single"
        trial_dir = os.path.join(trials_dir, trial_name)
        result = train_trial(data, config, device, trial_dir, trial_name)
        results.append(result)
        if result_is_better(result, best):
            best = result
            trials_without_global_improvement = 0
        else:
            trials_without_global_improvement += 1
        if args.search and trials_without_global_improvement >= args.search_patience:
            print(
                f"\nGlobal early stop: no better val_f1 for "
                f"{args.search_patience} consecutive trials."
            )
            break

    ranked = print_ranking(results)
    best = ranked[0]
    promote_best_trial(best, outputs_dir)
    if args.search:
        with open(os.path.join(outputs_dir, "search_results.json"), "w", encoding="utf-8") as f:
            json.dump(ranked, f, indent=2)
    summary = evaluate_best(data, device, outputs_dir)
    print("\nBest config:")
    print(json.dumps(best["config"], indent=2))
    print(
        f"Best epoch={summary['best_epoch']} val_loss={summary['best_val_loss']:.4f} "
        f"val_acc={summary['best_val_acc']:.4f} val_f1={summary['best_val_f1']:.4f} "
        f"train-val F1 gap={summary['f1_gap_at_best']:.4f}"
    )


if __name__ == "__main__":
    main()
