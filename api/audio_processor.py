"""
audio_processor.py — Xử lý Audio Upload
=========================================
Chịu trách nhiệm nhận file audio từ người dùng upload,
lưu tạm, tiền xử lý và trích xuất đặc trưng MFCC.

Pipeline:
  1. Lưu file upload vào thư mục tạm (temp/)
  2. Convert sang WAV nếu cần (webm, mp3, ogg → wav)
  3. Gọi hàm tiền xử lý từ model/demo.py
  4. Trích xuất MFCC features
  5. Xóa file tạm
  6. Trả về ma trận MFCC (numpy array)

Định dạng hỗ trợ: .wav, .mp3, .flac, .ogg, .webm, .m4a, .aac
"""

import os
import sys
import uuid
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

# Các định dạng mà librosa đọc trực tiếp được (không cần convert)
NATIVE_FORMATS = {".wav", ".flac", ".aiff", ".aif"}

# Tìm đường dẫn ffmpeg từ imageio-ffmpeg (đã cài qua pip)
def _get_ffmpeg_path():
    """Lấy đường dẫn tới ffmpeg binary từ imageio-ffmpeg package."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # Fallback: hy vọng ffmpeg có trong PATH


def convert_to_wav(input_path: str, output_path: str) -> str:
    """
    Convert file audio bất kỳ sang WAV 16kHz mono bằng ffmpeg.

    Args:
        input_path  : Đường dẫn file đầu vào (.webm, .mp3, .ogg...)
        output_path : Đường dẫn file WAV đầu ra

    Returns:
        str: Đường dẫn file WAV đã convert
    """
    import subprocess

    ffmpeg_path = _get_ffmpeg_path()

    cmd = [
        ffmpeg_path,
        "-y",              # Ghi đè nếu file đã tồn tại
        "-i", input_path,  # File đầu vào
        "-ar", "16000",    # Resample về 16kHz
        "-ac", "1",        # Mono channel
        "-f", "wav",       # Output format WAV
        output_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30  # Timeout 30 giây
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg convert thất bại: {error_msg}")

    return output_path


def process_audio_file(file_path: str, sr: int = 16000,
                       duration: float = 3.0, n_mfcc: int = 20) -> np.ndarray:
    """
    Xử lý file audio đã lưu trên đĩa và trích xuất đặc trưng MFCC.

    Nếu file không phải WAV/FLAC, sẽ tự động convert sang WAV trước.

    Args:
        file_path (str)  : Đường dẫn tuyệt đối tới file audio
        sr        (int)  : Sample rate mục tiêu (16000)
        duration  (float): Độ dài cắt/pad (3.0s)
        n_mfcc    (int)  : Số hệ số MFCC (20)

    Returns:
        np.ndarray: Ma trận MFCC shape (T, 60) — T~94 frames, 60 features
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    wav_path = None

    try:
        # Nếu file không phải native format → convert sang WAV
        if file_ext not in NATIVE_FORMATS:
            wav_path = file_path.rsplit(".", 1)[0] + "_converted.wav"
            print(f"[AudioProcessor] Converting {file_ext} -> .wav")
            convert_to_wav(file_path, wav_path)
            processing_path = wav_path
        else:
            processing_path = file_path

        # Bước 1: Tiền xử lý audio (resample, pad/trim, khử nhiễu)
        y, sr_out = tien_xu_ly(processing_path, sr=sr, duration=duration)

        # Bước 2: Trích xuất MFCC + Delta + Delta²
        mfcc = trich_xuat_mfcc(y, sr_out, n_mfcc=n_mfcc)

        return mfcc

    finally:
        # Xóa file WAV đã convert (nếu có)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


async def process_upload(upload_file) -> np.ndarray:
    """
    Xử lý file audio từ UploadFile của FastAPI.

    Quy trình:
      1. Lưu nội dung file upload vào thư mục tạm với tên ngẫu nhiên (UUID)
      2. Convert sang WAV nếu cần (webm, mp3, ogg...)
      3. Gọi process_audio_file() để trích xuất MFCC
      4. Xóa file tạm ngay sau khi xử lý xong
      5. Trả về ma trận MFCC

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

        # Trích xuất MFCC từ file tạm (tự động convert nếu cần)
        mfcc = process_audio_file(temp_path)

        return mfcc

    finally:
        # Luôn luôn xóa file tạm, kể cả khi có lỗi xảy ra
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Xóa cả file converted nếu còn sót
        converted_path = temp_path.rsplit(".", 1)[0] + "_converted.wav"
        if os.path.exists(converted_path):
            os.remove(converted_path)
