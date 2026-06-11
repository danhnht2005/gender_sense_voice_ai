"""evaluate.py - Metrics and simple training visualizations."""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


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





def plot_training_curves(history, save_path):
    """Save a 2x2 training-curve figure: Loss, Accuracy, F1, Learning Rate."""
    train_acc_key = "train_clean_acc" if "train_clean_acc" in history else "train_acc"
    train_loss_key = "train_clean_loss" if "train_clean_loss" in history else "train_loss"
    epochs = np.arange(len(history[train_loss_key]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: Loss
    axes[0, 0].plot(epochs, history[train_loss_key], label="train")
    axes[0, 0].plot(epochs, history["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    # Top-right: Accuracy
    axes[0, 1].plot(epochs, history[train_acc_key], label="train")
    axes[0, 1].plot(epochs, history["val_acc"], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    # Bottom-left: F1
    if "train_f1" in history and "val_f1" in history:
        axes[1, 0].plot(epochs, history["train_f1"], label="train")
        axes[1, 0].plot(epochs, history["val_f1"], label="val")
    axes[1, 0].set_title("F1")
    axes[1, 0].legend()

    # Bottom-right: Learning rate
    if "lr" in history:
        axes[1, 1].plot(epochs, history["lr"], label="lr")
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].legend()

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
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = plot_confusion_matrix(
        y_true, y_pred, os.path.join(outputs_dir, "confusion_matrix.png")
    )

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        plot_training_curves(history, os.path.join(outputs_dir, "training_curves.png"))
    else:
        print("[WARNING] training_history.json not found; skipping curves.")

    # Load val_loss from training history (last epoch or best)
    val_loss = None
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        if "val_loss" in history_data and history_data["val_loss"]:
            val_loss = min(history_data["val_loss"])  # best val_loss

    summary = {
        "accuracy": float(accuracy),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "f1_score": float(f1),
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
    if val_loss is not None:
        print(f"Val Loss:  {val_loss:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
