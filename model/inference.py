"""
inference.py — Standalone Inference
====================================
Dự đoán giới tính từ file audio đơn lẻ.

Sử dụng:
    python inference.py path/to/audio.wav
    python inference.py path/to/audio.wav --model outputs/best_model.pth

Output:
    Prediction: Male/Female (confidence: XX.X%)
"""

import os
import sys
import json
import argparse
import numpy as np

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch

# Import từ model package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_tcn_transformer_attention import TCN_Transformer_Attention_Model
from demo import tien_xu_ly, trich_xuat_mfcc


# ==============================================================================
# Inference Pipeline
# ==============================================================================

class GenderPredictor:
    """
    Lớp dự đoán giới tính từ file audio.
    
    Attributes:
        model       : PyTorch model đã load weights
        device      : torch.device (cpu/cuda)
        norm_mean   : Mean normalization (từ training)
        norm_std    : Std normalization (từ training)
    """
    
    def __init__(self, model_path=None, norm_stats_path=None, device=None):
        """
        Khởi tạo predictor.
        
        Args:
            model_path      : Đường dẫn tới best_model.pth
            norm_stats_path : Đường dẫn tới norm_stats.json
            device          : torch.device (None = auto-detect)
        """
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
        OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
        
        if model_path is None:
            model_path = os.path.join(OUTPUTS_DIR, "best_model.pth")
        if norm_stats_path is None:
            norm_stats_path = os.path.join(OUTPUTS_DIR, "norm_stats.json")
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        print(f"[INFO] Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Lấy hyperparameters từ checkpoint
        hp = checkpoint.get('hyperparameters', {})
        self.model = TCN_Transformer_Attention_Model(
            input_dim=hp.get('input_dim', 60),
            embed_dim=hp.get('embed_dim', 64),
            num_heads=hp.get('num_heads', 4),
            tcn_channels=hp.get('tcn_channels', 64),
            num_tcn=hp.get('num_tcn', 3),
            num_transformer=hp.get('num_transformer', 1),
            dropout=hp.get('dropout', 0.2)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[INFO] Model loaded (epoch {checkpoint.get('epoch', '?')}, "
              f"val_acc={checkpoint.get('val_acc', '?'):.4f})")
        
        # Load normalization stats
        print(f"[INFO] Loading norm stats from {norm_stats_path}")
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        self.norm_mean = np.array(norm_stats['mean'], dtype=np.float32)
        self.norm_std = np.array(norm_stats['std'], dtype=np.float32)
        
        print(f"[INFO] Predictor ready on {self.device}")
    
    def predict(self, audio_path, sr=16000, duration=3.0, n_mfcc=20):
        """
        Dự đoán giới tính từ file audio.
        
        Args:
            audio_path : Đường dẫn file .wav
            sr         : Sample rate (16000)
            duration   : Độ dài audio (3.0s)
            n_mfcc     : Số MFCC coefficients (20)
        
        Returns:
            dict: {
                'gender': 'Male' hoặc 'Female',
                'confidence': float (0-1),
                'probability_female': float (0-1),
                'probability_male': float (0-1)
            }
        """
        # 1. Tiền xử lý audio
        y, sr = tien_xu_ly(audio_path, sr=sr, duration=duration)
        
        # 2. Trích xuất MFCC
        mfcc = trich_xuat_mfcc(y, sr, n_mfcc=n_mfcc)  # (T, 60)
        
        # 3. Normalize
        mfcc = (mfcc - self.norm_mean) / self.norm_std
        
        # 4. Chuyển sang tensor: (1, T, 60)
        x = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        
        # 5. Inference
        with torch.no_grad():
            logit = self.model(x)  # (1, 1)
            prob_female = torch.sigmoid(logit).item()
        
        prob_male = 1.0 - prob_female
        gender = "Female" if prob_female >= 0.5 else "Male"
        confidence = max(prob_female, prob_male)
        
        return {
            'gender': gender,
            'confidence': confidence,
            'probability_female': prob_female,
            'probability_male': prob_male
        }
    
    def predict_from_array(self, y, sr, n_mfcc=20):
        """
        Dự đoán từ numpy array (dùng cho API).
        
        Args:
            y      : numpy array audio signal
            sr     : sample rate
            n_mfcc : số MFCC coefficients
        
        Returns:
            dict: Kết quả dự đoán
        """
        from demo import xu_ly_nhieu, trich_xuat_mfcc
        
        # Pad/trim to 3s
        target_length = int(sr * 3.0)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Xử lý nhiễu
        y = xu_ly_nhieu(y, sr)
        
        # Trích xuất MFCC
        mfcc = trich_xuat_mfcc(y, sr, n_mfcc=n_mfcc)
        
        # Normalize
        mfcc = (mfcc - self.norm_mean) / self.norm_std
        
        # Inference
        x = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x)
            prob_female = torch.sigmoid(logit).item()
        
        prob_male = 1.0 - prob_female
        gender = "Female" if prob_female >= 0.5 else "Male"
        confidence = max(prob_female, prob_male)
        
        return {
            'gender': gender,
            'confidence': confidence,
            'probability_female': prob_female,
            'probability_male': prob_male
        }


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dự đoán giới tính từ file audio"
    )
    parser.add_argument("audio_path", type=str, help="Đường dẫn file .wav")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn model .pth")
    parser.add_argument("--norm-stats", type=str, default=None, help="Đường dẫn norm_stats.json")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"[ERROR] File not found: {args.audio_path}")
        sys.exit(1)
    
    # Khởi tạo predictor
    predictor = GenderPredictor(
        model_path=args.model,
        norm_stats_path=args.norm_stats
    )
    
    # Dự đoán
    print(f"\n{'='*50}")
    print(f"  File: {args.audio_path}")
    print(f"{'='*50}")
    
    result = predictor.predict(args.audio_path)
    
    print(f"\n  Prediction:  {result['gender']}")
    print(f"  Confidence:  {result['confidence']*100:.1f}%")
    print(f"  P(Male):     {result['probability_male']*100:.1f}%")
    print(f"  P(Female):   {result['probability_female']*100:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
