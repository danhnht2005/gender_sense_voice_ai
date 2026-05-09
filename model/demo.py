"""
demo.py — Data Preprocessing Pipeline
======================================
Xử lý audio thô → trích xuất đặc trưng MFCC → xây dựng dataset.

Pipeline:
  1. xu_ly_nhieu()     : Normalize, DC offset, preemphasis
  2. tien_xu_ly()      : Resample → 16kHz, pad/trim → 3s
  3. trich_xuat_mfcc() : MFCC(20) + Delta + Delta² = 60 features
  4. build_dataset()   : Load toàn bộ dataset → cache X_raw.npy, y_raw.npy
  5. augment_dataset() : Feature-level augmentation (chỉ dùng cho train set)

Output: X_raw.npy shape (14196, 94, 60), y_raw.npy shape (14196,)
"""

import os
import sys
import numpy as np
import librosa

# Fix Windows console encoding for Vietnamese characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==============================================================================
# 1. Xử lý nhiễu (Noise Processing)
# ==============================================================================

def xu_ly_nhieu(y, sr):
    """
    Tiền xử lý tín hiệu audio:
      - Loại bỏ DC offset (trung bình tín hiệu)
      - Normalize biên độ về [-1, 1]
      - Áp dụng bộ lọc preemphasis (nhấn mạnh tần số cao)
    
    Args:
        y  (np.ndarray): Tín hiệu audio 1D
        sr (int)       : Sample rate
    
    Returns:
        np.ndarray: Tín hiệu đã xử lý
    """
    # Loại bỏ DC offset
    y = y - np.mean(y)
    
    # Normalize biên độ về [-1, 1]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    
    # Preemphasis filter: y[n] = y[n] - 0.97 * y[n-1]
    # Nhấn mạnh tần số cao, giúp MFCC capture được các đặc trưng giọng nói tốt hơn
    preemphasis_coeff = 0.97
    y = np.append(y[0], y[1:] - preemphasis_coeff * y[:-1])
    
    return y


# ==============================================================================
# 2. Tiền xử lý audio (Audio Preprocessing)
# ==============================================================================

def tien_xu_ly(file_path, sr=16000, duration=3.0):
    """
    Load và tiền xử lý file audio:
      - Load file với librosa
      - Resample về tần số chuẩn (16kHz)
      - Pad hoặc trim về độ dài cố định (3 giây)
      - Áp dụng xu_ly_nhieu()
    
    Args:
        file_path (str)  : Đường dẫn tới file .wav
        sr        (int)  : Sample rate mục tiêu (default: 16000)
        duration  (float): Độ dài mục tiêu tính bằng giây (default: 3.0)
    
    Returns:
        tuple: (y, sr) — tín hiệu đã xử lý và sample rate
    
    Raises:
        Exception: Nếu file không đọc được
    """
    # Load audio file, tự động resample về sr
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    
    # Tính số sample cần thiết cho duration giây
    target_length = int(sr * duration)
    
    # Pad (thêm silence) hoặc Trim (cắt bớt)
    if len(y) < target_length:
        # Pad với zeros (silence) ở cuối
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        # Trim lấy phần đầu
        y = y[:target_length]
    
    # Áp dụng xử lý nhiễu
    y = xu_ly_nhieu(y, sr)
    
    return y, sr


# ==============================================================================
# 3. Trích xuất MFCC (Feature Extraction)
# ==============================================================================

def trich_xuat_mfcc(y, sr, n_mfcc=20):
    """
    Trích xuất đặc trưng MFCC + Delta + Delta²:
      - MFCC: 20 hệ số Mel-Frequency Cepstral
      - Delta (velocity): Đạo hàm bậc 1 theo thời gian
      - Delta² (acceleration): Đạo hàm bậc 2 theo thời gian
      - Kết hợp: 20 + 20 + 20 = 60 features
    
    Args:
        y      (np.ndarray): Tín hiệu audio
        sr     (int)       : Sample rate
        n_mfcc (int)       : Số hệ số MFCC (default: 20)
    
    Returns:
        np.ndarray: Ma trận đặc trưng shape (T, 60) với T là số time frames
    """
    # Trích xuất MFCC — shape: (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Tính Delta (đạo hàm bậc 1)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Tính Delta² (đạo hàm bậc 2)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Kết hợp: stack theo chiều feature → (3*n_mfcc, T)
    combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
    
    # Transpose để có shape (T, features) = (T, 60)
    combined = combined.T
    
    return combined


# ==============================================================================
# 4. Xử lý một file (Helper cho multiprocessing)
# ==============================================================================

def _process_single_file(args):
    """
    Xử lý một file audio: load → tiền xử lý → trích xuất MFCC.
    Được thiết kế để chạy song song với ProcessPoolExecutor.
    
    Args:
        args (tuple): (file_path, label, sr, duration, n_mfcc)
    
    Returns:
        tuple: (mfcc_features, label) hoặc None nếu lỗi
    """
    file_path, label, sr, duration, n_mfcc = args
    try:
        y, sr = tien_xu_ly(file_path, sr=sr, duration=duration)
        mfcc = trich_xuat_mfcc(y, sr, n_mfcc=n_mfcc)
        return (mfcc, label)
    except Exception as e:
        print(f"[WARNING] Lỗi xử lý {file_path}: {e}")
        return None


# ==============================================================================
# 5. Xây dựng dataset (Build Dataset)
# ==============================================================================

def build_dataset(dataset_dir, sr=16000, duration=3.0, n_mfcc=20, 
                  max_workers=4, cache_dir=None):
    """
    Xây dựng dataset từ thư mục chứa audio files.
    
    Cấu trúc thư mục:
        dataset_dir/
        ├── male/      → label 0
        └── female/    → label 1
    
    Pipeline:
        1. Quét tất cả file .wav trong male/ và female/
        2. Song song: load → tiền xử lý → trích xuất MFCC
        3. Cache kết quả ra X_raw.npy và y_raw.npy
    
    Args:
        dataset_dir (str) : Đường dẫn thư mục dataset
        sr          (int) : Sample rate (16000)
        duration    (float): Độ dài audio (3.0s)
        n_mfcc      (int) : Số MFCC coefficients (20)
        max_workers (int) : Số workers cho multiprocessing
        cache_dir   (str) : Thư mục lưu cache (default: dataset_dir)
    
    Returns:
        tuple: (X, y) — X shape (N, T, 60), y shape (N,)
    """
    if cache_dir is None:
        cache_dir = dataset_dir
    
    cache_x = os.path.join(cache_dir, "X_raw.npy")
    cache_y = os.path.join(cache_dir, "y_raw.npy")
    
    # Kiểm tra cache
    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print(f"[INFO] Đọc từ cache: {cache_x}")
        X = np.load(cache_x)
        y = np.load(cache_y)
        print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    # Thu thập danh sách file
    file_list = []
    
    # Male = 0
    male_dir = os.path.join(dataset_dir, "male")
    if os.path.isdir(male_dir):
        for fname in os.listdir(male_dir):
            if fname.lower().endswith('.wav'):
                file_list.append((os.path.join(male_dir, fname), 0, sr, duration, n_mfcc))
    
    # Female = 1
    female_dir = os.path.join(dataset_dir, "female")
    if os.path.isdir(female_dir):
        for fname in os.listdir(female_dir):
            if fname.lower().endswith('.wav'):
                file_list.append((os.path.join(female_dir, fname), 1, sr, duration, n_mfcc))
    
    print(f"[INFO] Tổng số files: {len(file_list)}")
    print(f"  - Male:   {sum(1 for f in file_list if f[1] == 0)}")
    print(f"  - Female: {sum(1 for f in file_list if f[1] == 1)}")
    
    # Xử lý song song
    X_list = []
    y_list = []
    errors = 0
    
    print(f"[INFO] Đang xử lý với {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, args): args 
                   for args in file_list}
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Processing audio"):
            result = future.result()
            if result is not None:
                mfcc, label = result
                X_list.append(mfcc)
                y_list.append(label)
            else:
                errors += 1
    
    print(f"[INFO] Xử lý xong: {len(X_list)} thành công, {errors} lỗi")
    
    # Stack thành numpy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"[INFO] X shape: {X.shape}")  # Expected: (14196, 94, 60)
    print(f"[INFO] y shape: {y.shape}")  # Expected: (14196,)
    
    # Lưu cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    print(f"[INFO] Đã lưu cache: {cache_x}, {cache_y}")
    
    return X, y


# ==============================================================================
# 6. Data Augmentation (Feature-level)
# ==============================================================================

def augment_dataset(X, y, noise_factor=0.005, time_shift_max=5, 
                    freq_mask_max=5, num_augmented=1):
    """
    Tạo dữ liệu augmented ở mức feature (MFCC).
    CHỈ áp dụng cho training set (sau khi đã split).
    
    Kỹ thuật:
      1. Gaussian Noise: Thêm nhiễu ngẫu nhiên vào MFCC features
      2. Time Shift: Dịch chuyển theo trục thời gian (circular shift)
      3. Frequency Masking: Mask ngẫu nhiên một số frequency bins
    
    Args:
        X (np.ndarray)       : Features shape (N, T, F)
        y (np.ndarray)       : Labels shape (N,)
        noise_factor (float) : Hệ số nhiễu Gaussian
        time_shift_max (int) : Số frames dịch tối đa
        freq_mask_max (int)  : Số frequency bins mask tối đa
        num_augmented (int)  : Số bản augmented per sample
    
    Returns:
        tuple: (X_aug, y_aug) — Dữ liệu gốc + augmented
    """
    X_aug_list = [X]
    y_aug_list = [y]
    
    for i in range(num_augmented):
        X_copy = X.copy()
        
        # Kỹ thuật 1: Gaussian Noise
        noise = np.random.normal(0, noise_factor, X_copy.shape).astype(np.float32)
        X_noisy = X_copy + noise
        
        # Kỹ thuật 2: Time Shift (circular)
        shift = np.random.randint(-time_shift_max, time_shift_max + 1, size=X_copy.shape[0])
        X_shifted = np.zeros_like(X_noisy)
        for j in range(X_noisy.shape[0]):
            X_shifted[j] = np.roll(X_noisy[j], shift[j], axis=0)
        
        # Kỹ thuật 3: Frequency Masking
        for j in range(X_shifted.shape[0]):
            num_masks = np.random.randint(1, freq_mask_max + 1)
            f_start = np.random.randint(0, X_shifted.shape[2] - num_masks)
            X_shifted[j, :, f_start:f_start + num_masks] = 0
        
        X_aug_list.append(X_shifted)
        y_aug_list.append(y.copy())
    
    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    
    print(f"[INFO] Augmentation: {X.shape[0]} → {X_aug.shape[0]} samples "
          f"(x{num_augmented + 1})")
    
    return X_aug, y_aug


# ==============================================================================
# Main: Chạy trực tiếp để build dataset
# ==============================================================================

if __name__ == "__main__":
    # Đường dẫn dataset
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
    OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
    
    print("=" * 60)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Cache dir:   {OUTPUTS_DIR}")
    print()
    
    # Build dataset
    X, y = build_dataset(
        dataset_dir=DATASET_DIR,
        sr=16000,
        duration=3.0,
        n_mfcc=20,
        max_workers=4,
        cache_dir=OUTPUTS_DIR
    )
    
    print()
    print("=" * 60)
    print("  KẾT QUẢ")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Male (0):   {np.sum(y == 0):.0f}")
    print(f"Female (1): {np.sum(y == 1):.0f}")
    print(f"Dtype X: {X.dtype}, Dtype y: {y.dtype}")
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}")
    print("=" * 60)
