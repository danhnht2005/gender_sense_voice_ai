"""Train CNN/TCN baselines under the same protocol as the main model."""

import json
import os
import sys
import time
from dataclasses import asdict

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # noqa: E402
    TrainConfig,
    build_scheduler,
    evaluate_loader,
    is_better,
    make_loader,
    make_trial_loaders,
    plot_history,
    prepare_data,
    set_seed,
    train_one_epoch,
)


class PureCNN(nn.Module):
    """Plain 1D CNN baseline with global average pooling."""

    def __init__(self, input_dim, channels=64, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x.permute(0, 2, 1))
        return self.classifier(self.pool(x).squeeze(-1))


class TCNBlock(nn.Module):
    """Dilated residual temporal convolution block."""

    def __init__(self, in_channels, out_channels, dilation=1, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.residual(x))


class PureTCN(nn.Module):
    """Pure TCN baseline with global average pooling."""

    def __init__(self, input_dim, channels=64, num_blocks=4, dropout=0.5):
        super().__init__()
        self.tcn = nn.Sequential(
            *[
                TCNBlock(
                    input_dim if index == 0 else channels,
                    channels,
                    dilation=2**index,
                    dropout=dropout,
                )
                for index in range(num_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.tcn(x.permute(0, 2, 1))
        return self.classifier(self.pool(x).squeeze(-1))


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(payload, checkpoint_path, retries=5):
    """Atomically replace checkpoints to avoid transient Windows file locks."""
    temp_path = f"{checkpoint_path}.{os.getpid()}.tmp"
    for attempt in range(retries):
        try:
            torch.save(payload, temp_path)
            os.replace(temp_path, checkpoint_path)
            return
        except (OSError, RuntimeError):
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if attempt == retries - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def train_baseline(model, model_name, data, config, device, output_dir):
    """Train one baseline with the same optimization protocol as the main model."""
    os.makedirs(output_dir, exist_ok=True)
    set_seed(config.seed)
    loaders = make_trial_loaders(data, config, device)
    test_loader = make_loader(
        data["X_test"], data["y_test"], config.batch_size, False, device
    )

    n_male = int(np.sum(data["y_train"] == 0))
    n_female = int(np.sum(data["y_train"] == 1))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([n_male / max(n_female, 1)], device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    checkpoint_path = os.path.join(output_dir, "best_model.pth")
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

    print(f"\nTraining {model_name}: {count_parameters(model):,} parameters")
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
        for key in ("train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1"):
            history[key].append(candidate[key])
        history["lr"].append(current_lr)
        history["grad_norm"].append(grad_norm)

        f1_improved = best is None or candidate["val_f1"] > best["val_f1"] + 1e-4
        improved = is_better(candidate, best)
        if improved:
            best = candidate.copy()
            save_checkpoint(
                {
                    **best,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hyperparameters": {
                        **asdict(config),
                        "input_dim": data["input_dim"],
                        "time_steps": data["time_steps"],
                    },
                    "model_parameters": count_parameters(model),
                },
                checkpoint_path,
            )
        epochs_without_f1_improvement = (
            0 if f1_improved else epochs_without_f1_improvement + 1
        )

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
                f"  Early stopping after {config.early_stopping_patience} "
                "epochs without val_f1 improvement."
            )
            break

    history_path = os.path.join(output_dir, "history.json")
    curves_path = os.path.join(output_dir, "training_curves.png")
    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    plot_history(history, curves_path, model_name)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics, y_pred, y_true, y_prob = evaluate_loader(
        model, test_loader, criterion, device
    )
    auc_score = float(roc_auc_score(y_true, y_prob))
    result = {
        "model": model_name,
        "config": asdict(config),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_val_acc": float(checkpoint["val_acc"]),
        "best_val_f1": float(checkpoint["val_f1"]),
        "train_f1_at_best": float(checkpoint["train_f1"]),
        "f1_gap": float(checkpoint["f1_gap"]),
        "learning_rate_at_best": float(checkpoint["learning_rate"]),
        "epochs_ran": len(history["val_loss"]),
        "test_loss": float(test_loss),
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "auc": auc_score,
        "params": count_parameters(model),
        "train_time": time.time() - started,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "curves_path": curves_path,
        "y_pred": y_pred,
        "y_true": y_true,
        "y_prob": y_prob,
    }
    serializable = {
        key: value
        for key, value in result.items()
        if key not in {"y_pred", "y_true", "y_prob"}
    }
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)
    print(
        f"  Best epoch={result['best_epoch']} val_f1={result['best_val_f1']:.4f} "
        f"test_acc={result['accuracy']:.4f} test_f1={result['f1']:.4f}"
    )
    return result


def load_main_model_result(outputs_dir):
    eval_path = os.path.join(outputs_dir, "evaluation_summary.json")
    paths = {
        "pred": os.path.join(outputs_dir, "test_preds.npy"),
        "label": os.path.join(outputs_dir, "test_labels.npy"),
        "prob": os.path.join(outputs_dir, "test_probs.npy"),
    }
    if not os.path.exists(eval_path) or not all(os.path.exists(path) for path in paths.values()):
        return None

    with open(eval_path, "r", encoding="utf-8") as file:
        summary = json.load(file)
    best_config_path = os.path.join(outputs_dir, "best_config.json")
    best_config = {}
    if os.path.exists(best_config_path):
        with open(best_config_path, "r", encoding="utf-8") as file:
            best_config = json.load(file)
    checkpoint_path = os.path.join(outputs_dir, "best_model.pth")
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    y_pred = np.load(paths["pred"])
    y_true = np.load(paths["label"])
    y_prob = np.load(paths["prob"])
    return {
        "model": "TCN+Transformer+Attention",
        "accuracy": float(summary["accuracy"]),
        "f1": float(summary["f1_score"]),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "params": summary.get(
            "model_parameters", checkpoint.get("model_parameters", "N/A")
        ),
        "train_time": "N/A",
        "best_epoch": summary.get("best_epoch", best_config.get("best_epoch")),
        "best_val_loss": summary.get(
            "best_val_loss",
            best_config.get("best_val_loss", summary.get("val_loss")),
        ),
        "best_val_acc": summary.get(
            "best_val_acc", best_config.get("best_val_acc")
        ),
        "best_val_f1": summary.get(
            "best_val_f1", best_config.get("best_val_f1")
        ),
        "y_pred": y_pred,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def plot_baseline_comparison(results, save_path):
    names = list(results)
    metrics = [("accuracy", "Accuracy"), ("f1", "F1 Score"), ("auc", "AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(names))
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    all_values = [results[name][metric] for name in names for metric, _ in metrics]
    lower_bound = max(0.0, min(all_values) - 0.05)

    for ax, (metric, label) in zip(axes, metrics):
        values = [results[name][metric] for name in names]
        bars = ax.bar(x, values, color=colors[: len(names)], alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.003,
                f"{value:.4f}",
                ha="center",
                fontsize=10,
            )
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(lower_bound, 1.01)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Synchronized Baseline Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(results, save_path):
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        matrix = confusion_matrix(results[name]["y_true"], results[name]["y_pred"])
        ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(name)
        ax.set_xticks([0, 1], labels=["Male", "Female"])
        ax.set_yticks([0, 1], labels=["Male", "Female"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=14,
                    fontweight="bold",
                )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_dir, "outputs")
    baseline_dir = os.path.join(outputs_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("SYNCHRONIZED PURE CNN / PURE TCN BASELINES")
    print("=" * 72)
    print(f"Device: {device}")

    config = TrainConfig()
    data = prepare_data(project_dir, outputs_dir, config.seed)
    print("Protocol:", json.dumps(asdict(config), indent=2))

    model_specs = [
        (
            "Pure CNN",
            "pure_cnn",
            lambda: PureCNN(
                data["input_dim"], channels=64, dropout=config.dropout
            ),
        ),
        (
            "Pure TCN",
            "pure_tcn",
            lambda: PureTCN(
                data["input_dim"],
                channels=32,
                num_blocks=3,
                dropout=config.dropout,
            ),
        ),
    ]
    results = {}
    for model_name, folder_name, model_factory in model_specs:
        set_seed(config.seed)
        model = model_factory().to(device)
        result = train_baseline(
            model,
            model_name,
            data,
            config,
            device,
            os.path.join(baseline_dir, folder_name),
        )
        results[model_name] = result

    main_result = load_main_model_result(outputs_dir)
    if main_result is not None:
        results[main_result["model"]] = main_result

    print("\n" + "=" * 96)
    print("COMPARISON")
    print("=" * 96)
    print(
        f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10} "
        f"{'Best val F1':>12} {'Best epoch':>11} {'Params':>10}"
    )
    for name, result in results.items():
        best_val_f1 = result.get("best_val_f1")
        best_epoch = result.get("best_epoch")
        print(
            f"{name:<30} {result['accuracy']:>10.4f} {result['f1']:>10.4f} "
            f"{result['auc']:>10.4f} "
            f"{best_val_f1 if best_val_f1 is not None else float('nan'):>12.4f} "
            f"{str(best_epoch):>11} {str(result['params']):>10}"
        )

    plot_baseline_comparison(
        results, os.path.join(outputs_dir, "baseline_comparison.png")
    )
    plot_confusion_matrices(
        results, os.path.join(outputs_dir, "baseline_confusion_matrices.png")
    )
    serializable_results = {
        name: {
            key: value
            for key, value in result.items()
            if key not in {"y_pred", "y_true", "y_prob"}
        }
        for name, result in results.items()
    }
    with open(
        os.path.join(outputs_dir, "baseline_results.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(serializable_results, file, indent=2)
    print(f"\nSaved baseline artifacts to: {baseline_dir}")


if __name__ == "__main__":
    main()
