"""
main.py — FastAPI Application
===============================
Entry point cho Backend API của hệ thống phân loại giới tính qua giọng nói.

Endpoints:
  GET  /            → Welcome message
  GET  /health      → Health check (kiểm tra server + model)
  GET  /model-info  → Thông tin chi tiết về mô hình
  POST /predict     → Upload file audio → dự đoán giới tính

Chạy:
  uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Fix encoding cho Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from schemas import PredictionResponse, ModelInfoResponse, HealthResponse
from model_loader import model_service
from audio_processor import process_upload


# ======================================================================
# Khởi tạo FastAPI App
# ======================================================================

app = FastAPI(
    title="Gender Voice Classification API",
    description=(
        "API phân loại giới tính qua giọng nói sử dụng mô hình "
        "TCN + Transformer + Attention. Upload file audio (.wav) "
        "và nhận kết quả dự đoán Nam/Nữ kèm theo độ tin cậy."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ======================================================================
# CORS Middleware — cho phép Frontend gọi API từ domain khác
# ======================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Cho phép tất cả origins (dev mode)
    allow_credentials=True,
    allow_methods=["*"],            # Cho phép tất cả HTTP methods
    allow_headers=["*"],            # Cho phép tất cả headers
)


# ======================================================================
# Startup Event — Load model 1 lần khi server khởi động
# ======================================================================

@app.on_event("startup")
async def startup_event():
    """Nạp mô hình vào bộ nhớ khi server bắt đầu chạy."""
    print("=" * 60)
    print("  STARTING SERVER — Loading AI Model...")
    print("=" * 60)
    try:
        model_service.load()
        print("=" * 60)
        print("  SERVER READY!")
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise


# ======================================================================
# Endpoints
# ======================================================================

@app.get("/", tags=["General"])
async def root():
    """Welcome message — xác nhận API đang hoạt động."""
    return {
        "message": "Gender Voice Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /health",
            "model_info": "GET /model-info",
            "predict": "POST /predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Kiểm tra trạng thái hoạt động của server và model."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "unhealthy",
        model_loaded=model_service.is_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Trả về thông tin chi tiết của mô hình đang chạy."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model chưa được load.")

    info = model_service.model_info
    return ModelInfoResponse(
        model_name=info["model_name"],
        total_parameters=info["total_parameters"],
        best_epoch=info["best_epoch"],
        training_accuracy=info["training_accuracy"],
        device=info["device"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="File audio (.wav, .mp3, .flac...)")):
    """
    Dự đoán giới tính từ file audio upload.

    **Cách sử dụng:**
    - Upload một file audio (khuyến nghị .wav, hỗ trợ thêm .mp3, .flac, .ogg)
    - API sẽ tự động xử lý (resample 16kHz, pad/trim 3s) và trích xuất MFCC
    - Trả về kết quả dự đoán Nam/Nữ kèm xác suất và thời gian xử lý

    **Lưu ý:**
    - File audio nên có thời lượng tối thiểu 1 giây
    - Kích thước file tối đa: 10MB
    """
    # Kiểm tra model đã sẵn sàng
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng. Vui lòng thử lại sau.")

    # Kiểm tra file hợp lệ
    if not file.filename:
        raise HTTPException(status_code=400, detail="Không có file nào được upload.")

    # Kiểm tra kích thước (tối đa 10MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn. Tối đa 10MB.")
    # Reset file position để process_upload có thể đọc lại
    await file.seek(0)

    # Bắt đầu tính thời gian xử lý
    start_time = time.time()

    try:
        # Bước 1: Xử lý audio → MFCC features
        mfcc = await process_upload(file)

        # Bước 2: Dự đoán
        result = model_service.predict(mfcc)

        # Tính thời gian xử lý
        processing_time_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            probability_male=result["probability_male"],
            probability_female=result["probability_female"],
            processing_time_ms=round(processing_time_ms, 2),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý file audio: {str(e)}"
        )
