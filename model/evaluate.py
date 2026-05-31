"""
evaluate.py - Metrics and visualization for the training pipeline.

The training curves show both raw values and moving averages. The smoothing is
only for readability; raw metrics remain visible and the JSON history is not
modified.
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def moving_average(values, window=5):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    if class_names is None:
        class_names = ["Male (0)", "Female (1)"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color=color,
                fontsize=15,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")
    return cm


def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#1f77b4")

    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        marker="o",
        s=90,
        color="crimson",
        zorder=5,
        label=f"Best threshold={optimal_threshold:.3f}",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")
    return roc_auc


def plot_training_curves(history, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    smooth_window = 5 if len(epochs) >= 10 else 3

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="royalblue", alpha=0.35, label="Train raw")
    ax.plot(epochs, history["val_loss"], color="crimson", alpha=0.35, label="Val raw")
    ax.plot(
        epochs,
        moving_average(history["train_loss"], smooth_window),
        color="royalblue",
        linewidth=2.2,
        label=f"Train MA{smooth_window}",
    )
    ax.plot(
        epochs,
        moving_average(history["val_loss"], smooth_window),
        color="crimson",
        linewidth=2.2,
        label=f"Val MA{smooth_window}",
    )
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    best_val_loss = float(np.min(history["val_loss"]))
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.55)
    ax.annotate(
        f"Best val loss\nepoch {best_epoch}, {best_val_loss:.4f}",
        xy=(best_epoch, best_val_loss),
        xytext=(min(best_epoch + 3, len(epochs)), best_val_loss + max(history["val_loss"]) * 0.12 + 1e-4),
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green",
        fontsize=9,
    )
    ax.set_title("Train / Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], color="royalblue", alpha=0.35, label="Train raw")
    ax.plot(epochs, history["val_acc"], color="crimson", alpha=0.35, label="Val raw")
    ax.plot(
        epochs,
        moving_average(history["train_acc"], smooth_window),
        color="royalblue",
        linewidth=2.2,
        label=f"Train MA{smooth_window}",
    )
    ax.plot(
        epochs,
        moving_average(history["val_acc"], smooth_window),
        color="crimson",
        linewidth=2.2,
        label=f"Val MA{smooth_window}",
    )
    min_acc = min(min(history["train_acc"]), min(history["val_acc"]))
    max_acc = max(max(history["train_acc"]), max(history["val_acc"]))
    ax.set_ylim(max(0.0, min_acc - 0.01), min(1.005, max_acc + 0.005))
    ax.set_title("Train / Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["lr"], color="forestgreen", linewidth=2, label="Learning Rate")
    ax.set_yscale("log")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if "grad_norm" in history:
        ax.plot(epochs, history["grad_norm"], color="purple", alpha=0.35, label="Grad raw")
        ax.plot(
            epochs,
            moving_average(history["grad_norm"], smooth_window),
            color="purple",
            linewidth=2.2,
            label=f"Grad MA{smooth_window}",
        )
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No gradient norm logged", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Gradient Norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Training Curves - TCN+Transformer+Attention", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    outputs_dir = os.path.join(project_dir, "outputs")

    print("=" * 70)
    print("  EVALUATION & METRICS VISUALIZATION")
    print("=" * 70)

    test_preds_path = os.path.join(outputs_dir, "test_preds.npy")
    test_labels_path = os.path.join(outputs_dir, "test_labels.npy")
    test_probs_path = os.path.join(outputs_dir, "test_probs.npy")
    history_path = os.path.join(outputs_dir, "training_history.json")

    if not os.path.exists(test_preds_path):
        print("[ERROR] test_preds.npy not found. Run train.py first.")
        sys.exit(1)

    y_pred = np.load(test_preds_path)
    y_true = np.load(test_labels_path)
    y_probs = np.load(test_probs_path)

    print(f"Loaded {len(y_true)} test samples")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Male (0)", "Female (1)"],
            digits=4,
            zero_division=0,
        )
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = plot_confusion_matrix(
        y_true, y_pred, os.path.join(outputs_dir, "confusion_matrix.png")
    )
    roc_auc = plot_roc_curve(y_true, y_probs, os.path.join(outputs_dir, "roc_curve.png"))

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(outputs_dir, "training_curves.png"))
    else:
        print("[WARNING] training_history.json not found; skipping curves.")

    summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": int(len(y_true)),
        "male_samples": int(np.sum(y_true == 0)),
        "female_samples": int(np.sum(y_true == 1)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Male (0)", "Female (1)"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }

    summary_path = os.path.join(outputs_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {roc_auc:.4f}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
