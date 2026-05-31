"""
demo.py - Audio preprocessing and feature cache.

This module builds MFCC features and, importantly, stores sample metadata
next to the feature cache so train/validation/test splits can be audited for
data leakage.
"""

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
from tqdm import tqdm


if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def xu_ly_nhieu(y, sr):
    """Remove DC offset, normalize amplitude, and apply preemphasis."""
    y = y - np.mean(y)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    preemphasis_coeff = 0.97
    return np.append(y[0], y[1:] - preemphasis_coeff * y[:-1])


def tien_xu_ly(file_path, sr=16000, duration=3.0):
    """Load audio, resample to sr, pad/trim to duration, then preprocess."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    target_length = int(sr * duration)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]

    return xu_ly_nhieu(y, sr), sr


def trich_xuat_mfcc(y, sr, n_mfcc=20):
    """Extract MFCC + delta + delta2 features with shape (T, 3*n_mfcc)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0).T


def canonical_group_id(file_path):
    """
    Group repeated CMU Arctic utterances such as arctic_a0001(3).wav.

    The label folder is intentionally not part of the group key. If both
    male/female folders contain arctic_a0001 variants, all of those variants
    stay in the same split so the model cannot benefit from seeing the same
    spoken sentence in train and validation/test.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0].lower()
    return re.sub(r"\(\d+\)$", "", stem)


def _process_single_file(args):
    file_path, label, sr, duration, n_mfcc, sample_id, group_id = args
    try:
        y, sr = tien_xu_ly(file_path, sr=sr, duration=duration)
        mfcc = trich_xuat_mfcc(y, sr, n_mfcc=n_mfcc)
        return mfcc, label, sample_id, group_id
    except Exception as exc:
        print(f"[WARNING] Failed to process {file_path}: {exc}")
        return None


def _collect_files(dataset_dir, sr, duration, n_mfcc):
    file_list = []
    for class_name, label in (("male", 0), ("female", 1)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(class_dir, fname)
            sample_id = os.path.relpath(path, dataset_dir).replace("\\", "/")
            file_list.append(
                (path, label, sr, duration, n_mfcc, sample_id, canonical_group_id(path))
            )
    return file_list


def build_dataset(
    dataset_dir,
    sr=16000,
    duration=3.0,
    n_mfcc=20,
    max_workers=4,
    cache_dir=None,
    force_rebuild=False,
):
    """
    Build or load cached features.

    Returns:
        X, y, sample_ids, group_ids
    """
    if cache_dir is None:
        cache_dir = dataset_dir

    cache_x = os.path.join(cache_dir, "X_raw.npy")
    cache_y = os.path.join(cache_dir, "y_raw.npy")
    cache_ids = os.path.join(cache_dir, "sample_ids.npy")
    cache_groups = os.path.join(cache_dir, "group_ids.npy")

    cache_complete = all(
        os.path.exists(path) for path in (cache_x, cache_y, cache_ids, cache_groups)
    )
    if cache_complete and not force_rebuild:
        print(f"[INFO] Loading cache: {cache_x}")
        X = np.load(cache_x)
        y = np.load(cache_y)
        sample_ids = np.load(cache_ids, allow_pickle=False)
        group_ids = np.load(cache_groups, allow_pickle=False)
        print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
        print(f"[INFO] Unique groups: {len(np.unique(group_ids))}")
        return X, y, sample_ids, group_ids

    if os.path.exists(cache_x) and os.path.exists(cache_y) and not cache_complete:
        print("[WARNING] Old cache has no sample metadata; rebuilding cache.")

    file_list = _collect_files(dataset_dir, sr, duration, n_mfcc)
    print(f"[INFO] Total files: {len(file_list)}")
    print(f"  - Male:   {sum(1 for f in file_list if f[1] == 0)}")
    print(f"  - Female: {sum(1 for f in file_list if f[1] == 1)}")

    ordered_results = [None] * len(file_list)
    errors = 0
    print(f"[INFO] Processing audio with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_file, args): idx
            for idx, args in enumerate(file_list)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio"):
            result = future.result()
            if result is None:
                errors += 1
                continue
            ordered_results[futures[future]] = result

    X_list, y_list, id_list, group_list = [], [], [], []
    for result in ordered_results:
        if result is None:
            continue
        mfcc, label, sample_id, group_id = result
        X_list.append(mfcc)
        y_list.append(label)
        id_list.append(sample_id)
        group_list.append(group_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    sample_ids = np.array(id_list, dtype=str)
    group_ids = np.array(group_list, dtype=str)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_x, X)
    np.save(cache_y, y)
    np.save(cache_ids, sample_ids)
    np.save(cache_groups, group_ids)

    print(f"[INFO] Processed: {len(X_list)} success, {errors} errors")
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[INFO] Unique groups: {len(np.unique(group_ids))}")
    print(f"[INFO] Saved cache: {cache_dir}")
    return X, y, sample_ids, group_ids


def augment_dataset(
    X,
    y,
    noise_factor=0.003,
    time_shift_max=3,
    freq_mask_max=3,
    num_augmented=1,
):
    """Feature-level augmentation, intended for the training set only."""
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(num_augmented):
        X_copy = X.copy()
        noise = np.random.normal(0, noise_factor, X_copy.shape).astype(np.float32)
        X_noisy = X_copy + noise

        shift = np.random.randint(-time_shift_max, time_shift_max + 1, size=X_copy.shape[0])
        X_shifted = np.zeros_like(X_noisy)
        for j in range(X_noisy.shape[0]):
            X_shifted[j] = np.roll(X_noisy[j], shift[j], axis=0)

        for j in range(X_shifted.shape[0]):
            num_masks = np.random.randint(1, freq_mask_max + 1)
            f_start = np.random.randint(0, X_shifted.shape[2] - num_masks)
            X_shifted[j, :, f_start:f_start + num_masks] = 0

        X_aug_list.append(X_shifted)
        y_aug_list.append(y.copy())

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    print(f"[INFO] Augmentation: {X.shape[0]} -> {X_aug.shape[0]} samples")
    return X_aug, y_aug


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_dir, "dataset")
    outputs_dir = os.path.join(project_dir, "outputs")

    print("=" * 60)
    print("  DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Dataset dir: {dataset_dir}")
    print(f"Cache dir:   {outputs_dir}")

    X, y, sample_ids, group_ids = build_dataset(
        dataset_dir=dataset_dir,
        sr=16000,
        duration=3.0,
        n_mfcc=20,
        max_workers=4,
        cache_dir=outputs_dir,
    )

    print("=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"sample_ids shape: {sample_ids.shape}")
    print(f"unique groups: {len(np.unique(group_ids))}")
    print(f"Male (0):   {np.sum(y == 0):.0f}")
    print(f"Female (1): {np.sum(y == 1):.0f}")
    print(f"X min: {X.min():.4f}, max: {X.max():.4f}")
