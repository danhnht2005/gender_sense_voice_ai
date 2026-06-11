"""
model_loader.py — Model Service (Singleton)
=============================================
Nạp mô hình Deep Learning một lần duy nhất khi server khởi động.
Cung cấp hàm predict() để dự đoán giới tính từ MFCC features.

Sử dụng Singleton pattern để đảm bảo model chỉ được load 1 lần
vào bộ nhớ, tất cả các request sẽ dùng chung cùng một instance.
"""

import os
import sys
import json
import numpy as np
import torch

# Thêm thư mục gốc project vào path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.model_tcn_transformer_attention import (
    TCN_Transformer_Attention_Model,
    count_parameters,
    infer_model_hyperparameters,
)


class ModelService:
    """
    Singleton service quản lý mô hình Deep Learning.

    Attributes:
        model       : PyTorch model đã load trọng số
        device      : torch.device (cpu hoặc cuda)
        norm_mean   : numpy array mean cho normalization
        norm_std    : numpy array std cho normalization
        model_info  : dict chứa metadata của model (epoch, accuracy...)
        _is_loaded  : flag đánh dấu model đã được nạp thành công
    """

    def __init__(self):
        self.model = None
        self.device = None
        self.norm_mean = None
        self.norm_std = None
        self.input_dim = None
        self.time_steps = None
        self.model_info = {}
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Kiểm tra model đã sẵn sàng chưa."""
        return self._is_loaded

    def load(self, model_path: str = None, norm_stats_path: str = None):
        """
        Nạp trọng số mô hình và thông số chuẩn hóa từ đĩa.
        Chỉ nên gọi 1 lần duy nhất khi server startup.

        Args:
            model_path      : Đường dẫn tới best_model.pth
            norm_stats_path : Đường dẫn tới norm_stats.json
        """
        OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

        if model_path is None:
            model_path = os.path.join(OUTPUTS_DIR, "best_model.pth")
        if norm_stats_path is None:
            norm_stats_path = os.path.join(OUTPUTS_DIR, "norm_stats.json")

        # --- 1. Chọn thiết bị ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelService] Device: {self.device}")

        # --- 2. Load checkpoint ---
        print(f"[ModelService] Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # --- 3. Khởi tạo model với hyperparameters từ checkpoint ---
        hp = infer_model_hyperparameters(checkpoint)
        self.input_dim = int(hp["input_dim"])
        self.time_steps = int(hp.get("time_steps", 94))
        self.model = TCN_Transformer_Attention_Model(
            input_dim=self.input_dim,
            embed_dim=hp.get("embed_dim", 64),
            num_heads=hp.get("num_heads", 4),
            tcn_channels=hp.get("tcn_channels", 64),
            num_tcn=hp.get("num_tcn", 3),
            num_transformer=hp.get("num_transformer", 1),
            dropout=hp.get("dropout", 0.2),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()  # Chuyển sang chế độ inference (tắt dropout, batchnorm eval)

        # --- 4. Lưu metadata ---
        self.model_info = {
            "model_name": "TCN_Transformer_Attention_Model",
            "total_parameters": count_parameters(self.model),
            "best_epoch": checkpoint.get("epoch", -1),
            "training_accuracy": checkpoint.get("val_acc", -1),
            "device": str(self.device),
        }

        print(f"[ModelService] Model loaded successfully!")
        print(f"  - Parameters: {self.model_info['total_parameters']:,}")
        print(f"  - Best epoch: {self.model_info['best_epoch']}")
        print(f"  - Val accuracy: {self.model_info['training_accuracy']:.4f}")

        # --- 5. Load normalization stats ---
        print(f"[ModelService] Loading norm stats from {norm_stats_path}")
        with open(norm_stats_path, "r") as f:
            norm_stats = json.load(f)
        self.norm_mean = np.array(norm_stats["mean"], dtype=np.float32)
        self.norm_std = np.array(norm_stats["std"], dtype=np.float32)
        expected_shape = (self.input_dim,)
        if self.norm_mean.shape != expected_shape or self.norm_std.shape != expected_shape:
            raise ValueError(
                "Normalization stats do not match the model input: "
                f"model expects {self.input_dim} features, "
                f"mean shape={self.norm_mean.shape}, std shape={self.norm_std.shape}."
            )
        if np.any(self.norm_std <= 0):
            raise ValueError("Normalization std must contain only positive values.")

        self.model_info.update(
            {
                "features": f"MFCC({self.input_dim}) = {self.input_dim} features",
                "input_shape": f"(batch, {self.time_steps}, {self.input_dim})",
            }
        )

        self._is_loaded = True
        print("[ModelService] Ready for inference!")

    def predict(self, mfcc: np.ndarray) -> dict:
        """
        Dự đoán giới tính từ ma trận MFCC features.

        Args:
            mfcc (np.ndarray): MFCC features shape (T, input_dim)

        Returns:
            dict: {
                'label': 'Male' hoặc 'Female',
                'confidence': float,
                'probability_male': float,
                'probability_female': float,
            }

        Raises:
            RuntimeError: Nếu model chưa được load
        """
        if not self._is_loaded:
            raise RuntimeError("Model chưa được load! Gọi load() trước.")

        # Bước 1: Normalize features (dùng mean/std từ training set)
        if mfcc.ndim != 2 or mfcc.shape[1] != self.input_dim:
            raise ValueError(
                f"Invalid MFCC shape {mfcc.shape}; expected (time, {self.input_dim})."
            )
        mfcc_norm = (mfcc - self.norm_mean) / self.norm_std

        # Bước 2: Chuyển sang PyTorch tensor — shape: (1, T, input_dim)
        x = torch.FloatTensor(mfcc_norm).unsqueeze(0).to(self.device)

        # Bước 3: Inference (tắt tính gradient để tăng tốc)
        with torch.no_grad():
            logit = self.model(x)  # (1, 1)
            prob_female = torch.sigmoid(logit).item()

        # Bước 4: Tính kết quả
        prob_male = 1.0 - prob_female
        label = "Female" if prob_female >= 0.5 else "Male"
        confidence = max(prob_female, prob_male)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probability_male": round(prob_male, 4),
            "probability_female": round(prob_female, 4),
        }


# ======================================================================
# Singleton instance — toàn bộ API sẽ dùng chung instance này
# ======================================================================
model_service = ModelService()
