"""
schemas.py — Pydantic Models (Request / Response)
===================================================
Định nghĩa cấu trúc dữ liệu JSON cho API.

- PredictionResponse : Kết quả dự đoán giới tính
- ModelInfoResponse  : Thông tin mô hình
- HealthResponse     : Trạng thái server
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    """Kết quả dự đoán giới tính từ file audio."""

    label: str = Field(
        ...,
        description="Giới tính dự đoán: 'Male' hoặc 'Female'",
        examples=["Male"]
    )
    confidence: float = Field(
        ...,
        description="Độ tin cậy của dự đoán (0.0 - 1.0)",
        ge=0.0, le=1.0,
        examples=[0.9876]
    )
    probability_male: float = Field(
        ...,
        description="Xác suất là giọng Nam (0.0 - 1.0)",
        ge=0.0, le=1.0,
        examples=[0.9876]
    )
    probability_female: float = Field(
        ...,
        description="Xác suất là giọng Nữ (0.0 - 1.0)",
        ge=0.0, le=1.0,
        examples=[0.0124]
    )
    processing_time_ms: float = Field(
        ...,
        description="Thời gian xử lý tính bằng mili-giây",
        examples=[45.3]
    )


class ModelInfoResponse(BaseModel):
    """Thông tin chi tiết về mô hình đang được sử dụng."""

    model_name: str = Field(
        ...,
        description="Tên kiến trúc mô hình",
        examples=["TCN_Transformer_Attention_Model"]
    )
    total_parameters: int = Field(
        ...,
        description="Tổng số tham số trainable",
        examples=[117826]
    )
    best_epoch: int = Field(
        ...,
        description="Epoch có kết quả tốt nhất trong quá trình training",
        examples=[21]
    )
    training_accuracy: float = Field(
        ...,
        description="Độ chính xác trên tập Validation tại best epoch",
        examples=[1.0]
    )
    features: str = Field(
        default="MFCC(20) + Delta + Delta2 = 60 features",
        description="Loại đặc trưng âm thanh được sử dụng"
    )
    input_shape: str = Field(
        default="(batch, 94, 60)",
        description="Kích thước đầu vào của mô hình"
    )
    device: str = Field(
        ...,
        description="Thiết bị đang chạy inference (cpu hoặc cuda)",
        examples=["cpu"]
    )


class HealthResponse(BaseModel):
    """Trạng thái hoạt động của API server."""

    status: str = Field(
        ...,
        description="Trạng thái server",
        examples=["healthy"]
    )
    model_loaded: bool = Field(
        ...,
        description="Mô hình đã được nạp thành công hay chưa",
        examples=[True]
    )
