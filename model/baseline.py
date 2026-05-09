"""
baseline.py — Baseline Models & So Sánh
=========================================
Train và đánh giá 3 baseline models, so sánh với model chính.

Baselines:
  B1. SVM (RBF kernel)     — Traditional ML, flatten MFCC
  B2. Random Forest        — Traditional ML, 200 trees
  B3. Simple 1D-CNN        — Deep Learning baseline

Output:
  - baseline_results.json     : Số liệu so sánh
  - baseline_comparison.png   : Bar chart so sánh
"""

import os
import sys
import json
import time
import numpy as np

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import build_dataset
from model_tcn_transformer_attention import TCN_Transformer_Attention_Model, count_parameters


# ==============================================================================
# 1. Simple 1D-CNN Baseline
# ==============================================================================

class SimpleCNN(nn.Module):
    """
    Simple 1D-CNN baseline cho so sánh.
    
    Architecture:
        Conv1d(60→32) → ReLU → Conv1d(32→64) → ReLU → Conv1d(64→64) → ReLU
        → AdaptiveAvgPool1d(1) → FC(64→1)
    
    ~15K params
    """
    
    def __init__(self, input_dim=60):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, T, F) — T=94, F=60
        Returns:
            (batch, 1)
        """
        # (batch, T, F) → (batch, F, T)
        x = x.permute(0, 2, 1)
        
        x = self.features(x)      # (batch, 64, T)
        x = self.pool(x)          # (batch, 64, 1)
        x = x.squeeze(-1)         # (batch, 64)
        x = self.classifier(x)    # (batch, 1)
        
        return x


# ==============================================================================
# 2. Train Simple CNN
# ==============================================================================

def train_simple_cnn(X_train, y_train, X_val, y_val, X_test, y_test, device):
    """
    Train Simple 1D-CNN baseline.
    
    Returns:
        dict: metrics {accuracy, f1, auc, params, train_time}
    """
    print("\n  Training Simple 1D-CNN...")
    
    model = SimpleCNN(input_dim=60).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {params:,}")
    
    # DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Loss & optimizer
    n_male = np.sum(y_train == 0)
    n_female = np.sum(y_train == 1)
    pos_weight = torch.tensor([n_male / n_female]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    start = time.time()
    
    for epoch in range(1, 51):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device).unsqueeze(1)
                out = model(bx)
                val_loss += criterion(out, by).item() * bx.size(0)
        val_loss /= len(val_ds)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    train_time = time.time() - start
    
    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(float))
            all_labels.extend(by.numpy().flatten())
    
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    
    print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    print(f"    Time: {train_time:.1f}s, Epochs: {epoch}")
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'auc': float(auc_score),
        'params': params,
        'train_time': float(train_time),
        'y_pred': y_pred,
        'y_true': y_true,
        'y_prob': y_prob
    }


# ==============================================================================
# 3. Train SVM Baseline
# ==============================================================================

def train_svm(X_train_flat, y_train, X_test_flat, y_test):
    """
    Train SVM (RBF kernel) baseline.
    
    Returns:
        dict: metrics
    """
    print("\n  Training SVM (RBF kernel)...")
    print(f"    Input shape: {X_train_flat.shape}")
    
    start = time.time()
    
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_flat, y_train)
    
    train_time = time.time() - start
    
    y_pred = svm.predict(X_test_flat)
    y_prob = svm.predict_proba(X_test_flat)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    print(f"    Time: {train_time:.1f}s")
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'auc': float(auc_score),
        'params': 'N/A',
        'train_time': float(train_time),
        'y_pred': y_pred,
        'y_true': y_test,
        'y_prob': y_prob
    }


# ==============================================================================
# 4. Train Random Forest Baseline
# ==============================================================================

def train_random_forest(X_train_flat, y_train, X_test_flat, y_test):
    """
    Train Random Forest baseline.
    
    Returns:
        dict: metrics
    """
    print("\n  Training Random Forest (200 trees)...")
    print(f"    Input shape: {X_train_flat.shape}")
    
    start = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, 
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_flat, y_train)
    
    train_time = time.time() - start
    
    y_pred = rf.predict(X_test_flat)
    y_prob = rf.predict_proba(X_test_flat)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    
    print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    print(f"    Time: {train_time:.1f}s")
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'auc': float(auc_score),
        'params': 'N/A',
        'train_time': float(train_time),
        'y_pred': y_pred,
        'y_true': y_test,
        'y_prob': y_prob
    }


# ==============================================================================
# 5. Plot So Sánh
# ==============================================================================

def plot_baseline_comparison(results, save_path):
    """
    Vẽ bar chart so sánh 4 models.
    
    Args:
        results   : Dict {model_name: {accuracy, f1, auc}}
        save_path : Đường dẫn lưu PNG
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1', 'auc']
    metric_labels = ['Accuracy', 'F1 Score', 'AUC']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(model_names))
    bar_width = 0.6
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        values = [results[name][metric] for name in model_names]
        
        bars = ax.bar(x, values, bar_width, color=colors[:len(model_names)], 
                      edgecolor='white', linewidth=1.5, alpha=0.85)
        
        # Giá trị trên mỗi bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
        ax.set_ylim([0.7, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.4, label='Target (0.95)')
        ax.legend(fontsize=9)
    
    plt.suptitle('Model Comparison — Gender Voice Classification',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved → {save_path}")


def plot_all_confusion_matrices(results, save_path):
    """
    Vẽ confusion matrix cho tất cả models trong 1 figure.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    class_names = ['Male', 'Female']
    
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        r = results[name]
        cm = confusion_matrix(r['y_true'], r['y_pred'])
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                       color=color, fontsize=14, fontweight='bold')
    
    plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ==============================================================================
# 6. Main
# ==============================================================================

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
    OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
    
    print("=" * 70)
    print("  BASELINE MODELS & COMPARISON")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ==== Load Data ====
    print("\n[1] Loading data...")
    X, y = build_dataset(dataset_dir=DATASET_DIR, cache_dir=OUTPUTS_DIR)
    
    # Stratified split (same as train.py)
    SEED = 42
    np.random.seed(SEED)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Normalize (same stats as training)
    train_mean = np.mean(X_train, axis=(0, 1))
    train_std = np.std(X_train, axis=(0, 1))
    train_std[train_std == 0] = 1e-8
    
    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    
    # Flatten for ML models: (N, 94, 60) → (N, 5640)
    X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
    X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)
    
    print(f"  Flattened shape: {X_train_flat.shape}")
    
    # ==== Train Baselines ====
    print("\n[2] Training baselines...")
    results = {}
    
    # B1: SVM
    print("\n" + "-" * 50)
    results['SVM (RBF)'] = train_svm(X_train_flat, y_train, X_test_flat, y_test)
    
    # B2: Random Forest
    print("\n" + "-" * 50)
    results['Random Forest'] = train_random_forest(X_train_flat, y_train, X_test_flat, y_test)
    
    # B3: Simple 1D-CNN
    print("\n" + "-" * 50)
    results['Simple 1D-CNN'] = train_simple_cnn(
        X_train_norm, y_train, X_val_norm, y_val, X_test_norm, y_test, device
    )
    
    # ==== Load Main Model Results ====
    print("\n[3] Loading main model results...")
    eval_path = os.path.join(OUTPUTS_DIR, "evaluation_summary.json")
    
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_summary = json.load(f)
        
        # Load test predictions
        test_preds = np.load(os.path.join(OUTPUTS_DIR, "test_preds.npy"))
        test_labels = np.load(os.path.join(OUTPUTS_DIR, "test_labels.npy"))
        test_probs = np.load(os.path.join(OUTPUTS_DIR, "test_probs.npy"))
        
        results['TCN+Trans+Attn'] = {
            'accuracy': eval_summary['accuracy'],
            'f1': eval_summary['f1_score'],
            'auc': eval_summary['auc'],
            'params': '~118K',
            'train_time': 'N/A',
            'y_pred': test_preds,
            'y_true': test_labels,
            'y_prob': test_probs
        }
        print(f"  Main model: Acc={eval_summary['accuracy']:.4f}, F1={eval_summary['f1_score']:.4f}")
    else:
        print("  [WARNING] evaluation_summary.json not found. Run train.py + evaluate.py first!")
        print("  Skipping main model in comparison.")
    
    # ==== Comparison Table ====
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Params':>10}")
    print(f"  {'-'*65}")
    for name, r in results.items():
        print(f"  {name:<25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f} {str(r['params']):>10}")
    print(f"  {'-'*65}")
    
    # ==== Plot ====
    print("\n[4] Generating comparison plots...")
    
    # Chuẩn bị results cho plot (loại bỏ numpy arrays)
    plot_results = {}
    for name, r in results.items():
        plot_results[name] = {
            'accuracy': r['accuracy'],
            'f1': r['f1'],
            'auc': r['auc']
        }
    
    plot_baseline_comparison(
        plot_results,
        save_path=os.path.join(OUTPUTS_DIR, "baseline_comparison.png")
    )
    
    plot_all_confusion_matrices(
        results,
        save_path=os.path.join(OUTPUTS_DIR, "baseline_confusion_matrices.png")
    )
    
    # ==== Save Results JSON ====
    print("\n[5] Saving results...")
    save_results = {}
    for name, r in results.items():
        save_results[name] = {
            'accuracy': r['accuracy'],
            'f1': r['f1'],
            'auc': r['auc'],
            'params': str(r['params']),
            'train_time': r.get('train_time', 'N/A')
        }
    
    results_path = os.path.join(OUTPUTS_DIR, "baseline_results.json")
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved → {results_path}")
    
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
