"""
train.py — Training Pipeline
=============================
Pipeline huấn luyện model TCN+Transformer+Attention cho phân loại giới tính.

Workflow:
  1. Load dữ liệu đã cache (X_raw.npy, y_raw.npy)
  2. Stratified split: train/val/test = 70/15/15
  3. Augment CHỈ train set
  4. Normalize features (mean/std từ train set) → lưu norm_stats.json
  5. Train với BCEWithLogitsLoss + pos_weight + ReduceLROnPlateau + Early Stopping
  6. Save best model → outputs/best_model.pth
  7. Save training history → outputs/training_history.json
  8. Evaluate trên test set → save predictions
"""

import os
import sys
import json
import time
import random
import numpy as np

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Import từ model package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_tcn_transformer_attention import TCN_Transformer_Attention_Model, count_parameters
from demo import build_dataset, augment_dataset


# ==============================================================================
# 1. Reproducibility
# ==============================================================================

def set_seed(seed=42):
    """
    Đặt seed cho tất cả random generators để đảm bảo reproducibility.
    
    Args:
        seed (int): Giá trị seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")


def get_device():
    """
    Auto-detect thiết bị tính toán (GPU/CPU).
    
    Returns:
        torch.device: cuda hoặc cpu
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using CPU")
    return device


# ==============================================================================
# 2. Early Stopping
# ==============================================================================

class EarlyStopping:
    """
    Early Stopping để tránh overfitting.
    
    Theo dõi validation loss, nếu không cải thiện sau 'patience' epochs → dừng.
    
    Args:
        patience  (int)  : Số epochs chờ trước khi dừng
        min_delta (float): Ngưỡng cải thiện tối thiểu
        verbose   (bool) : In thông báo
    """
    
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ==============================================================================
# 3. Training Loop
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train một epoch.
    
    Args:
        model      : PyTorch model
        dataloader : DataLoader cho train set
        criterion  : Loss function
        optimizer  : Optimizer
        device     : torch.device
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device).unsqueeze(1)  # (batch,) → (batch, 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)  # (batch, 1)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
        
        # Accuracy
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validate/Test trên một dataset.
    
    Args:
        model      : PyTorch model
        dataloader : DataLoader
        criterion  : Loss function
        device     : torch.device
    
    Returns:
        tuple: (avg_loss, accuracy, all_preds, all_labels, all_probs)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ==============================================================================
# 4. Main Training Pipeline
# ==============================================================================

def main():
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
    OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # ==== Hyperparameters ====
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 10  # Early stopping
    
    # Model hyperparameters
    INPUT_DIM = 60
    EMBED_DIM = 64
    NUM_HEADS = 4
    TCN_CHANNELS = 64
    NUM_TCN = 3
    NUM_TRANSFORMER = 1
    DROPOUT = 0.2
    
    print("=" * 70)
    print("  TRAINING PIPELINE — Gender Voice Classification")
    print("=" * 70)
    
    # ==== Setup ====
    set_seed(SEED)
    device = get_device()
    
    # ==== 1. Load Data ====
    print("\n" + "=" * 70)
    print("  STEP 1: Load Data")
    print("=" * 70)
    
    X, y = build_dataset(
        dataset_dir=DATASET_DIR,
        cache_dir=OUTPUTS_DIR
    )
    
    print(f"Dataset: X={X.shape}, y={y.shape}")
    print(f"Male (0): {np.sum(y == 0):.0f}, Female (1): {np.sum(y == 1):.0f}")
    
    # ==== 2. Stratified Split ====
    print("\n" + "=" * 70)
    print("  STEP 2: Train/Val/Test Split (70/15/15)")
    print("=" * 70)
    
    # Split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    
    # Split temp: 50/50 → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]} samples (Male: {np.sum(y_train==0):.0f}, Female: {np.sum(y_train==1):.0f})")
    print(f"Val:   {X_val.shape[0]} samples (Male: {np.sum(y_val==0):.0f}, Female: {np.sum(y_val==1):.0f})")
    print(f"Test:  {X_test.shape[0]} samples (Male: {np.sum(y_test==0):.0f}, Female: {np.sum(y_test==1):.0f})")
    
    # ==== 3. Augment Train Set ====
    print("\n" + "=" * 70)
    print("  STEP 3: Data Augmentation (Train Set Only)")
    print("=" * 70)
    
    X_train_aug, y_train_aug = augment_dataset(
        X_train, y_train, 
        noise_factor=0.005, 
        time_shift_max=5,
        freq_mask_max=5,
        num_augmented=1  # x2 train data
    )
    
    # ==== 4. Normalize ====
    print("\n" + "=" * 70)
    print("  STEP 4: Normalization (Train Set Statistics)")
    print("=" * 70)
    
    # Tính mean/std từ train set (trước augmentation cho stats ổn định hơn)
    train_mean = np.mean(X_train, axis=(0, 1))  # (60,)
    train_std = np.std(X_train, axis=(0, 1))     # (60,)
    train_std[train_std == 0] = 1e-8  # Tránh chia cho 0
    
    # Normalize tất cả sets
    X_train_aug = (X_train_aug - train_mean) / train_std
    X_val_norm = (X_val - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    
    # Lưu norm stats
    norm_stats = {
        "mean": train_mean.tolist(),
        "std": train_std.tolist()
    }
    norm_stats_path = os.path.join(OUTPUTS_DIR, "norm_stats.json")
    with open(norm_stats_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved norm_stats → {norm_stats_path}")
    
    # ==== 5. Create DataLoaders ====
    print("\n" + "=" * 70)
    print("  STEP 5: Create DataLoaders")
    print("=" * 70)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_aug),
        torch.FloatTensor(y_train_aug)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)} (batch_size={BATCH_SIZE})")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # ==== 6. Model Setup ====
    print("\n" + "=" * 70)
    print("  STEP 6: Model Setup")
    print("=" * 70)
    
    model = TCN_Transformer_Attention_Model(
        input_dim=INPUT_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        tcn_channels=TCN_CHANNELS,
        num_tcn=NUM_TCN,
        num_transformer=NUM_TRANSFORMER,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss với pos_weight cho class imbalance
    # pos_weight = num_negative / num_positive = num_male / num_female
    n_male = np.sum(y_train_aug == 0)
    n_female = np.sum(y_train_aug == 1)
    pos_weight = torch.tensor([n_male / n_female], dtype=torch.float32).to(device)
    print(f"Class weights — Male: {n_male}, Female: {n_female}, pos_weight: {pos_weight.item():.4f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    
    # ==== 7. Training Loop ====
    print("\n" + "=" * 70)
    print("  STEP 7: Training")
    print("=" * 70)
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUTS_DIR, "best_model.pth")
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch [{epoch:3d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}  |  "
              f"LR: {current_lr:.6f}  |  "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'hyperparameters': {
                    'input_dim': INPUT_DIM,
                    'embed_dim': EMBED_DIM,
                    'num_heads': NUM_HEADS,
                    'tcn_channels': TCN_CHANNELS,
                    'num_tcn': NUM_TCN,
                    'num_transformer': NUM_TRANSFORMER,
                    'dropout': DROPOUT,
                }
            }, best_model_path)
            print(f"  [BEST] Best model saved! (val_loss: {val_loss:.4f})")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # ==== 8. Save Training History ====
    history_path = os.path.join(OUTPUTS_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history → {history_path}")
    
    # ==== 9. Test Set Evaluation ====
    print("\n" + "=" * 70)
    print("  STEP 8: Test Set Evaluation")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\n{'='*40}")
    print(f"  TEST RESULTS")
    print(f"{'='*40}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*40}")
    
    # Save test predictions
    np.save(os.path.join(OUTPUTS_DIR, "test_preds.npy"), test_preds)
    np.save(os.path.join(OUTPUTS_DIR, "test_labels.npy"), test_labels)
    np.save(os.path.join(OUTPUTS_DIR, "test_probs.npy"), test_probs)
    print("Saved test predictions → outputs/test_preds.npy, test_labels.npy, test_probs.npy")
    
    # ==== Summary ====
    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Model:          {model.__class__.__name__}")
    print(f"  Parameters:     {count_parameters(model):,}")
    print(f"  Best Epoch:     {checkpoint['epoch']}")
    print(f"  Best Val Loss:  {checkpoint['val_loss']:.4f}")
    print(f"  Best Val Acc:   {checkpoint['val_acc']:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Total Time:     {total_time:.1f}s")
    print(f"\nOutputs saved to: {OUTPUTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
