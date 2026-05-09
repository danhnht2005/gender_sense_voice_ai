"""
evaluate.py — Evaluation & Metrics Visualization
==================================================
Đánh giá model và tạo visualization cho báo cáo.

Outputs:
  - confusion_matrix.png    : Ma trận nhầm lẫn
  - roc_curve.png           : ROC Curve + AUC
  - training_curves.png     : Loss & Accuracy curves
  - evaluation_summary.json : Tổng hợp metrics
"""

import os
import sys
import json
import numpy as np

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score
)


# ==============================================================================
# 1. Confusion Matrix
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    """
    Vẽ và lưu Confusion Matrix.
    
    Args:
        y_true      : Labels thực
        y_pred      : Labels dự đoán
        save_path   : Đường dẫn lưu file PNG
        class_names : Tên các lớp (default: ['Male', 'Female'])
    """
    if class_names is None:
        class_names = ['Male (0)', 'Female (1)']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', pad=15)
    
    # Hiển thị số trong mỗi ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center", color=color,
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")
    
    return cm


# ==============================================================================
# 2. ROC Curve
# ==============================================================================

def plot_roc_curve(y_true, y_probs, save_path):
    """
    Vẽ và lưu ROC Curve + AUC score.
    
    Args:
        y_true    : Labels thực
        y_probs   : Xác suất dự đoán (probability)
        save_path : Đường dẫn lưu file PNG
    
    Returns:
        float: AUC score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # ROC curve
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    
    # Đường baseline (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', 
            label='Random Classifier')
    
    # Tô màu vùng dưới curve
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
    
    # Tìm và đánh dấu optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', s=100,
              color='red', zorder=5, label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve — Gender Classification', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")
    
    return roc_auc


# ==============================================================================
# 3. Training Curves
# ==============================================================================

def plot_training_curves(history, save_path):
    """
    Vẽ Training/Validation Loss & Accuracy Curves.
    
    Args:
        history   : Dict chứa train_loss, val_loss, train_acc, val_acc, lr
        save_path : Đường dẫn lưu file PNG
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Plot 1: Loss ---
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Đánh dấu best epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: epoch {best_epoch}\nloss={best_val_loss:.4f}',
                xy=(best_epoch, best_val_loss),
                xytext=(best_epoch + 2, best_val_loss + 0.05),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # --- Plot 2: Accuracy ---
    ax2 = axes[1]
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc', alpha=0.8)
    ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Đánh dấu best accuracy
    best_acc_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    ax2.axvline(x=best_acc_epoch, color='green', linestyle='--', alpha=0.5)
    ax2.annotate(f'Best: epoch {best_acc_epoch}\nacc={best_val_acc:.4f}',
                xy=(best_acc_epoch, best_val_acc),
                xytext=(best_acc_epoch + 2, best_val_acc - 0.05),
                fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # --- Plot 3: Learning Rate ---
    ax3 = axes[2]
    ax3.plot(epochs, history['lr'], 'g-', linewidth=2, label='Learning Rate', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.suptitle('Training Curves — TCN+Transformer+Attention', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ==============================================================================
# 4. Main Evaluation
# ==============================================================================

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
    
    print("=" * 70)
    print("  EVALUATION & METRICS VISUALIZATION")
    print("=" * 70)
    
    # ==== Load predictions ====
    print("\n[1] Loading predictions...")
    
    test_preds_path = os.path.join(OUTPUTS_DIR, "test_preds.npy")
    test_labels_path = os.path.join(OUTPUTS_DIR, "test_labels.npy")
    test_probs_path = os.path.join(OUTPUTS_DIR, "test_probs.npy")
    history_path = os.path.join(OUTPUTS_DIR, "training_history.json")
    
    if not os.path.exists(test_preds_path):
        print("[ERROR] test_preds.npy not found. Run train.py first!")
        sys.exit(1)
    
    y_pred = np.load(test_preds_path)
    y_true = np.load(test_labels_path)
    y_probs = np.load(test_probs_path)
    
    print(f"  Loaded {len(y_true)} test samples")
    
    # ==== Classification Report ====
    print("\n[2] Classification Report:")
    print("-" * 50)
    report = classification_report(
        y_true, y_pred, 
        target_names=['Male (0)', 'Female (1)'],
        digits=4
    )
    print(report)
    
    # ==== Metrics ====
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # ==== Confusion Matrix ====
    print("\n[3] Confusion Matrix...")
    cm = plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    )
    
    # ==== ROC Curve ====
    print("\n[4] ROC Curve...")
    roc_auc = plot_roc_curve(
        y_true, y_probs,
        save_path=os.path.join(OUTPUTS_DIR, "roc_curve.png")
    )
    print(f"  AUC: {roc_auc:.4f}")
    
    # ==== Training Curves ====
    print("\n[5] Training Curves...")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_curves(
            history,
            save_path=os.path.join(OUTPUTS_DIR, "training_curves.png")
        )
    else:
        print("  [WARNING] training_history.json not found, skipping curves")
    
    # ==== Save Summary ====
    print("\n[6] Saving evaluation summary...")
    
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
            y_true, y_pred,
            target_names=['Male (0)', 'Female (1)'],
            digits=4,
            output_dict=True
        )
    }
    
    summary_path = os.path.join(OUTPUTS_DIR, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {summary_path}")
    
    # ==== Final Summary ====
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {roc_auc:.4f}")
    print(f"\n  Outputs:")
    print(f"    - confusion_matrix.png")
    print(f"    - roc_curve.png")
    print(f"    - training_curves.png")
    print(f"    - evaluation_summary.json")
    
    # Check targets
    print(f"\n  Target Check:")
    print(f"    Accuracy >= 95%: {'PASS' if accuracy >= 0.95 else 'FAIL'} ({accuracy*100:.2f}%)")
    print(f"    F1 >= 0.95:      {'PASS' if f1 >= 0.95 else 'FAIL'} ({f1:.4f})")
    print(f"    AUC >= 0.98:     {'PASS' if roc_auc >= 0.98 else 'FAIL'} ({roc_auc:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
