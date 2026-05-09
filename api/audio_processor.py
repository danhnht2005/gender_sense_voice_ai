"""
audio_processor.py — Xử lý Audio Upload
=========================================
Chịu trách nhiệm nhận file audio từ người dùng upload,
lưu tạm, tiền xử lý và trích xuất đặc trưng MFCC.

Pipeline:
  1. Lưu file upload vào thư mục tạm (temp/)
  2. Gọi hàm tiền xử lý từ model/demo.py
  3. Trích xuất MFCC features
  4. Xóa file tạm
  5. Trả về ma trận MFCC (numpy array)
"""

import os
import sys
import uuid
import tempfile
import numpy as np

# Thêm thư mục gốc project vào path để import từ model/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.demo import tien_xu_ly, trich_xuat_mfcc


# Thư mục lưu file tạm
TEMP_DIR = os.path.join(PROJECT_DIR, "api", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def process_audio_file(file_path: str, sr: int = 16000,
                       duration: float = 3.0, n_mfcc: int = 20) -> np.ndarray:
    """
    Xử lý file audio đã lưu trên đĩa và trích xuất đặc trưng MFCC.

    Args:
        file_path (str)  : Đường dẫn tuyệt đối tới file audio
        sr        (int)  : Sample rate mục tiêu (16000)
        duration  (float): Độ dài cắt/pad (3.0s)
        n_mfcc    (int)  : Số hệ số MFCC (20)

    Returns:
        np.ndarray: Ma trận MFCC shape (T, 60) — T~94 frames, 60 features
    """
    # Bước 1: Tiền xử lý audio (resample, pad/trim, khử nhiễu)
    y, sr_out = tien_xu_ly(file_path, sr=sr, duration=duration)

    # Bước 2: Trích xuất MFCC + Delta + Delta²
    mfcc = trich_xuat_mfcc(y, sr_out, n_mfcc=n_mfcc)

    return mfcc


async def process_upload(upload_file) -> np.ndarray:
    """
    Xử lý file audio từ UploadFile của FastAPI.

    Quy trình:
      1. Lưu nội dung file upload vào thư mục tạm với tên ngẫu nhiên (UUID)
      2. Gọi process_audio_file() để trích xuất MFCC
      3. Xóa file tạm ngay sau khi xử lý xong
      4. Trả về ma trận MFCC

    Args:
        upload_file: đối tượng UploadFile từ FastAPI

    Returns:
        np.ndarray: Ma trận MFCC features shape (T, 60)

    Raises:
        Exception: Nếu file audio không hợp lệ hoặc không đọc được
    """
    # Tạo tên file tạm duy nhất (tránh xung đột khi nhiều request đồng thời)
    file_ext = os.path.splitext(upload_file.filename or "audio.wav")[1] or ".wav"
    temp_filename = f"{uuid.uuid4().hex}{file_ext}"
    temp_path = os.path.join(TEMP_DIR, temp_filename)

    try:
        # Đọc nội dung file upload và ghi ra đĩa
        content = await upload_file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Trích xuất MFCC từ file tạm
        mfcc = process_audio_file(temp_path)

        return mfcc

    finally:
        # Luôn luôn xóa file tạm, kể cả khi có lỗi xảy ra
        if os.path.exists(temp_path):
            os.remove(temp_path)
